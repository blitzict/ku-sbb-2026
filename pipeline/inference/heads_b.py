"""Early saved linear heads (group B): expert fusion -> harmonic -> seasonal -> mechanism -> trend -> extended -> hard state tree.

Each head is a fixed readout saved in new weights/<name>_candidate/weights.json. The test logit is
parent test logit + design(test) @ w, with design rebuilt from the new upstream test outputs in the
prediction workspace. Rank encodings use the quantile tables saved with the weights. Nothing is refitted:
the training scripts' fit/fit_head/fit_views are never called. Train/OOF archives, result reports and
weights are copied verbatim from the trained workspace; submission.csv and replay inputs are recomputed.
"""
import hashlib
import importlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from .common import destination, require, save_npz

NW = 'new weights'
WS = 'tune_out/weight_space'
FUSION = 'expert_fusion_candidate'


def _mod(name):
    return importlib.import_module(name)


def _ids(work, part):
    return pd.read_parquet(require(Path(work) / f'clean/features_{part}.parquet'), columns=[]).index


def _recipe(model_root, folder):
    return json.loads(require(Path(model_root) / NW / folder / 'weights.json').read_text())


def _parent(work, folder, tids):
    sub = pd.read_csv(require(Path(work) / NW / folder / 'submission.csv'), dtype={'id': str}).set_index('id')
    return sub.loc[tids, 'break_probability'].to_numpy()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _copy(stage, model_root, work, skip=()):
    #train-side outputs (weights, oof archives, reports) are reused unchanged
    written = []
    for rel in stage['outputs']:
        if rel in skip:
            continue
        src = Path(model_root) / rel
        if src.is_file():
            target = destination(work, rel)
            shutil.copy2(src, target)
            written.append(target)
    return written


def _submission(work, rel, p, tids):
    #same checks/format as finalize.write_submission; row order follows the workspace sample submission
    tids = pd.Index(tids).astype(str)
    if p.shape != (len(tids),) or not np.isfinite(p).all() or not ((0 <= p) & (p <= 1)).all():
        raise ValueError(f'{rel}: invalid test probabilities')
    sample = Path(work) / 'data/sample_submission.csv'
    if sample.is_file():
        sub = pd.read_csv(sample, dtype={'id': str})
        if not sub.id.is_unique or set(sub.id) != set(tids):
            raise ValueError('sample_submission ids differ from the test ids')
        sub = sub[['id']].copy()
    else:
        sub = pd.DataFrame({'id': np.asarray(tids)})
    sub['break_probability'] = pd.Series(p, index=tids).loc[sub.id].to_numpy()
    target = destination(work, rel)
    sub.to_csv(target, index=False)
    return sub, target


def _head(stage, model_root, work, folder, tids, z, f, w, bias=None):
    #parent logit + design @ saved weights; writes submission and the packaged replay files the stage declares
    base = f'{NW}/{folder}/'
    p = expit(z + f @ w if bias is None else z + f @ w + bias)
    sub, target = _submission(work, base + 'submission.csv', p, tids)
    written = [target]
    outs = set(stage['outputs'])
    if base + 'replay_inputs.npz' in outs:
        order = pd.Index(tids).get_indexer(sub.id)
        path = destination(work, base + 'replay_inputs.npz')
        save_npz(path, ids=np.asarray(tids).astype(str), order=order, parent_logits=z, features=f)
        written.append(path)
    if base + 'submission_replay.csv' in outs:
        path = destination(work, base + 'submission_replay.csv')
        shutil.copyfile(target, path)
        written.append(path)
    if base + 'validation.json' in outs and (Path(model_root) / base / 'validation.json').is_file():
        report = json.loads((Path(model_root) / base / 'validation.json').read_text())
        if 'submission_sha256' in report:
            report['submission_sha256'] = _sha(target)
        if 'rows' in report:
            report['rows'] = len(tids)
        path = destination(work, base + 'validation.json')
        path.write_text(json.dumps(report, indent=2))
        written.append(path)
    done = {base + n for n in ('submission.csv', 'replay_inputs.npz', 'submission_replay.csv', 'validation.json')}
    return written + _copy(stage, model_root, work, skip=done)


def _fusion_inputs(work):
    #test side of expert_fusion.read_data without the training-time hash checks
    ef = _mod('expert_fusion')
    work = Path(work)
    tids = _ids(work, 'test')
    item = json.loads(require(work / NW / 'ensemble.json').read_text())
    names = json.loads(require(work / 'submissions/sub_seqw5_stack.json').read_text())['members']
    mats = []
    for part in ('oof', 'test'):
        paths = [ef.OLD / f'{part}_head_v2_bag.npy' if n == 'MLPHV_bag_w' else ef.O / f'{part}_{n}.npy' for n in names]
        mats.append(np.column_stack([logit(np.clip(np.load(require(p)), 1e-06, 1 - 1e-06)) for p in paths]).astype(float))
    archives = {t: ef.OLD / f'ver_head_v2_{t}.npz' for t in ['MLPHV_big', 'MLPHV_big_o1', 'MLPHV_big_o2']}
    _, VT, labels, _, _ = ef.expand(names, *mats, archives)
    E = mats[1][:, [names.index(n) for n in ef.EXPERTS]]
    blocks = []
    for spec in (ef.CONTEXT, ef.EVIDENCE):
        for stem, cols in spec.items():
            b = pd.read_parquet(require(work / f'{stem}_test.parquet'), columns=cols).loc[tids, cols]
            if not (b.index.equals(tids) and b.index.is_unique):
                raise ValueError(f'{stem}: test ids differ')
            blocks.append(b)
    D = np.column_stack(blocks)
    nc = sum(len(c) for c in ef.CONTEXT.values())
    if not np.isfinite(E).all():
        raise ValueError('non-finite expert logits')
    theta = np.array(item['coefficients_and_intercept'])
    z = VT[:, [labels.index(n) for n in item['feature_order']]] @ theta[:-1] + theta[-1]
    return ef, tids, z, E, D, nc


def _fusion_z(model_root, work):
    #expert fusion test logit and its context ranks (parent of the harmonic head)
    ef, tids, z, E, D, nc = _fusion_inputs(work)
    art = _recipe(model_root, FUSION)
    context = ef.ranks(D, np.array(art['quantiles']))
    return ef, tids, ef.adjust(art['mode'], E, context, z, nc, np.array(art['weights'])), context


def expert_fusion(stage, model_root, work):
    _, tids, z, _ = _fusion_z(model_root, work)
    sub, target = _submission(work, f'{NW}/{FUSION}/submission.csv', expit(z), tids)
    return [target] + _copy(stage, model_root, work, skip={f'{NW}/{FUSION}/submission.csv'})


def residual_trial(stage, model_root, work):
    rt = _mod('residual_trial')
    tag = stage['argv'][stage['argv'].index('--tag') + 1]
    folder = f'{tag}_candidate'
    art = _recipe(model_root, folder)
    ef, tids, z, context = _fusion_z(model_root, work)
    tab = rt.frame(tag, 'test', tids)
    if list(tab.columns) != art['feature_order']:
        raise ValueError(f'{tag}: test columns differ from the saved head')
    f = rt.matrix(art['mode'], ef.ranks(tab.to_numpy(), np.array(art['quantiles'])), z, context)
    theta = np.array(art['coefficients'])
    return _head(stage, model_root, work, folder, tids, z, f, theta[:-1], theta[-1])


def seasonal_head(stage, model_root, work):
    sh = _mod('seasonal_head')
    seq = '--sequence' in stage['argv']
    folder = 'seasonal' + ('_seq' if seq else '') + '_candidate'
    art = _recipe(model_root, folder)
    if bool(art['sequence']) != seq:
        raise ValueError(f'{folder}: sequence flag differs from the saved head')
    ids, tids = _ids(work, 'train'), _ids(work, 'test')
    _, x, g, _ = sh.read_tables((None, ids, tids), seq)
    orig, harm = (_parent(work, p, tids) for p in (FUSION, 'harmonic_candidate'))
    g = g[len(ids):]
    z = sh.base(art['mode'], logit(orig), logit(harm) - logit(orig), g)
    f = sh.features(art['mode'], x[art['cols']].to_numpy()[len(ids):], g, z, np.array(art['quantiles']))
    return _head(stage, model_root, work, folder, tids, z, f, np.array(art['weights']))


def _test_tables(tags, tids):
    rt = _mod('residual_trial')
    return {t: rt.frame(t, 'test', tids) for t in tags}


def _season_gate(tids):
    #seasonal_head.read_tables gate, test rows only
    p = _test_tables(['periodic'], tids)['periodic']
    return p.gate.to_numpy() * expit((p.period.to_numpy() - 9.0) / 1.5) * expit((38.0 - p.period.to_numpy()) / 2.0)


def mechanism_edit(stage, model_root, work):
    me = _mod('mechanism_edit')
    mode = stage['argv'][stage['argv'].index('--mode') + 1]
    folder = f'mechanism_{mode}_candidate'
    art = _recipe(model_root, folder)
    if art['mode'] != mode:
        raise ValueError(f'{folder}: mode differs from the saved head')
    tids = _ids(work, 'test')
    g = _season_gate(tids)
    t = _test_tables(me.inputs(mode), tids)
    z = logit(_parent(work, 'seasonal_seq_candidate', tids))
    f = me.encode(mode, t, z, g, np.arange(len(tids)))
    return _head(stage, model_root, work, folder, tids, z, f, np.array(art['weights']))


def trend_870(stage, model_root, work):
    trend = _mod('trend_870')
    mode = stage['argv'][stage['argv'].index('--mode') + 1]
    folder = f'trend870_{mode}_candidate'
    art = _recipe(model_root, folder)
    if art['mode'] != mode:
        raise ValueError(f'{folder}: mode differs from the saved head')
    tids = _ids(work, 'test')
    t = _test_tables(trend.names(mode), tids)
    z = logit(_parent(work, 'mechanism_phase_ar_mix_transfer_candidate', tids))
    f = trend.design(mode, t, z, np.arange(len(tids)))
    return _head(stage, model_root, work, folder, tids, z, f, np.array(art['weights']))


def extend_readout(stage, model_root, work):
    er = _mod('extend_readout')
    spec = stage['argv'][stage['argv'].index('--spec') + 1]
    folder = 'extended_' + spec.replace('.', '_').replace('+', '_') + '_candidate'
    art = _recipe(model_root, folder)
    if art['spec'] != spec:
        raise ValueError(f'{folder}: spec differs from the saved head')
    tids = _ids(work, 'test')
    mods = er.modules(spec)
    t = _test_tables(list(dict.fromkeys(tag for m, mode in mods for tag in m.tags(mode))), tids)
    z = logit(_parent(work, 'trend870_net_candidate', tids))
    f = er.features(mods, t, z, np.arange(len(tids)))
    return _head(stage, model_root, work, folder, tids, z, f, np.array(art['weights']))


def hard_expert_trial(stage, model_root, work):
    kind = stage['argv'][stage['argv'].index('--kind') + 1]
    if '--verify' not in stage['argv']:
        #unverified trial exports only its cv report and oof archive
        return _copy(stage, model_root, work)
    he = _mod('hard_expert_trial')
    folder = f'hard_{kind}_candidate'
    art = _recipe(model_root, folder)
    if art['kind'] != kind:
        raise ValueError(f'{folder}: kind differs from the saved head')
    tids = _ids(work, 'test')
    X, _ = he.sources(kind, 'test', tids)
    st = pd.read_parquet(require(Path(work) / WS / 'state_test.parquet')).loc[tids]
    z = logit(_parent(work, Path(he.PARENT).parent.name, tids))
    f = he.features(X, z, np.arange(len(tids)), st)
    return _head(stage, model_root, work, folder, tids, z, f, np.array(art['weights']))


ADAPTERS = {
    'expert_fusion.py': expert_fusion, 'residual_trial.py': residual_trial, 'seasonal_head.py': seasonal_head,
    'mechanism_edit.py': mechanism_edit, 'trend_870.py': trend_870, 'extend_readout.py': extend_readout,
    'hard_expert_trial.py': hard_expert_trial,
}


def supported(stage):
    return Path(stage['argv'][0]).name in ADAPTERS


def predict_stage(stage, model_root, work):
    return ADAPTERS[Path(stage['argv'][0]).name](stage, Path(model_root), Path(work))
