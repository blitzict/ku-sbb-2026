"""Saved white-noise readout heads (group E1): parent cache, consult, stage, frozen stage, view contrasts,
family agreement, early opinions and their frozen caches.

Every head is test = expit(zt + design(test) @ w). zt (the parent's test logits) and design(test) are rebuilt
from the upstream test outputs already written into the prediction workspace; w, the outer law priors and the
local-rank centering grid come from the saved weights.json in model_root. Train/OOF arrays and train-only
reports are copied verbatim. Nothing is refitted: white_outer_priors.fit_on, white_nested.fit and
white_local_rank_weights.weights are never called.
"""
import argparse
import importlib
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.special import expit, logit, logsumexp

from .common import destination, require, save_npz, save_table

WS = Path('tune_out/weight_space')
PACK = Path('new weights')


def _mod(name):
    return importlib.import_module(name)


def _sha(path):
    return _mod('expert_fusion').sha(path)


def _copy(model_root, work, relative):
    dst = destination(work, relative)
    temporary = dst.with_name(dst.name + '.prediction-partial')
    shutil.copyfile(require(Path(model_root) / relative), temporary)
    temporary.replace(dst)
    return dst


def _copy_rest(stage, model_root, work, written):
    #train-only outputs (reports, oof archives, nested fold caches) are kept verbatim
    done = {Path(p).resolve() for p in written}
    out = list(written)
    for rel in stage['outputs']:
        if (Path(work) / rel).resolve() not in done:
            out.append(_copy(model_root, work, rel))
    return out


def _parent(work):
    with np.load(require(Path(work) / WS / 'white_loop_parent.npz')) as q:
        return pd.Index(q['ids']), pd.Index(q['tids']), q['zt'].copy()


def _law_test(work, tids, priors):
    #white_outer_priors.fit_on test rows, with the law/null/change mixtures it saved (no refit)
    head = _mod('white_law_head')
    with np.load(require(Path(work) / WS / 'wnlaw_physical_test.npz')) as a:
        if not np.array_equal(a['ids'], tids):
            raise ValueError('wnlaw_physical test ids differ from the parent query ids')
        M = a['values'].astype(float)
    law = np.array(priors['law'])
    _, L, F = head.unpack(M, law)
    lr = logsumexp(L + np.log(np.array(priors['change'])), axis=1) - logsumexp(L + np.log(np.array(priors['null'])), axis=1)
    return np.column_stack([lr, F]).clip(-80, 80)


def _candidate(model_root, work, name, zt, F, tids, order_keys):
    #write submission.csv / replay_inputs.npz of wn_<name>_candidate with the saved weights
    folder = PACK / f'wn_{name}_candidate'
    recipe = require(Path(model_root) / folder / 'weights.json')
    spec = json.loads(recipe.read_text())
    w = np.array(spec['weights'])
    if F.shape != (len(tids), len(w)):
        raise ValueError(f'{name}: test design {F.shape} does not match {len(w)} saved weights')
    written = [_copy(model_root, work, folder / 'weights.json')]
    csv = destination(work, folder / 'submission.csv')
    sub = _mod('expert_fusion').write_submission(expit(zt + F @ w), tids, csv)
    written.append(csv)
    values = dict(parent_logits=zt, features=F, ids=tids.to_numpy(dtype=str), order=tids.get_indexer(sub.id))
    replay = destination(work, folder / 'replay_inputs.npz')
    save_npz(replay, **{k: values[k] for k in order_keys})
    written.append(replay)
    source = Path(model_root) / folder / 'validation.json'
    if source.is_file():
        v = json.loads(source.read_text())
        v.update(recipe_sha256=_sha(written[0]), submission_sha256=_sha(csv))
        dst = destination(work, folder / 'validation.json')
        dst.write_text(json.dumps(v, indent=2))
        written.append(dst)
    return spec, written


def parent_cache(stage, model_root, work):
    #white_loop.prepare: zt is logit of the locked parent's (hard_state_tree) test submission
    tids = pd.read_parquet(require(Path(work) / 'clean/features_test.parquet'), columns=[]).index
    p = pd.read_csv(require(Path(work) / PACK / 'hard_state_tree_candidate/submission.csv'), dtype={'id': str})
    p = p.set_index('id').loc[tids, 'break_probability']
    with np.load(require(Path(model_root) / WS / 'white_loop_parent.npz'), allow_pickle=True) as q:
        values = {k: q[k] for k in q.files}
    values.update(tids=tids.to_numpy(dtype=str), zt=logit(p.to_numpy()))
    path = destination(work, WS / 'white_loop_parent.npz')
    save_npz(path, **values)
    return _copy_rest(stage, model_root, work, [path])


def consult(stage, model_root, work):
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('base', 'short', 'full'))
    p.add_argument('--extra', nargs='*', default=[])
    p.add_argument('--seeds', type=int, nargs='+')
    p.add_argument('--tag', default='physics')
    p.add_argument('--physical', nargs='*', default=[])
    a = p.parse_args(stage['argv'][1:])
    name = f'consult_{a.mode}_{a.tag}'
    spec = json.loads(require(Path(model_root) / PACK / f'wn_{name}_candidate/weights.json').read_text())
    wc = _mod('white_consult')
    if spec['models'] != wc.CORE + a.extra:
        raise ValueError(f'{name}: saved member list differs from the stage arguments')
    ids, tids, zt = _parent(work)
    build, _ = wc.compile_inputs(ids, tids, spec['models'], a.physical)
    te = len(ids) + np.arange(len(tids))
    F = build(_law_test(work, tids, spec['priors']), te, zt, a.mode)
    _, written = _candidate(model_root, work, name, zt, F, tids, ('parent_logits', 'features', 'ids', 'order'))
    return _copy_rest(stage, model_root, work, written)


def stage_head(stage, model_root, work):
    tag = stage['argv'][1]
    name = 'stage_' + tag
    spec = json.loads(require(Path(model_root) / PACK / f'wn_{name}_candidate/weights.json').read_text())
    wc, ws = _mod('white_consult'), _mod('white_stage')
    ids, tids, zt = _parent(work)
    base, _ = wc.compile_inputs(ids, tids, spec['models'], ['volatility'])
    side, _ = ws.new_inputs(spec['tables'], ids, tids)
    te = len(ids) + np.arange(len(tids))
    F = np.column_stack([base(_law_test(work, tids, spec['priors']), te, zt, 'full'), side[te]])
    _, written = _candidate(model_root, work, name, zt, F, tids, ('ids', 'parent_logits', 'features', 'order'))
    return _copy_rest(stage, model_root, work, written)


def _frozen_full(model_root, work, pack, relative):
    #full.npz: test logits = replayed parent logits + features @ saved weights; train fields verbatim
    spec = json.loads(require(Path(work) / pack / 'weights.json').read_text())
    with np.load(require(Path(work) / pack / 'replay_inputs.npz')) as r:
        tids = r['ids'].copy()
        test = r['parent_logits'] + r['features'] @ np.array(spec['weights'])
    _, parent_ids, _ = _parent(work)
    if not np.array_equal(tids, parent_ids.to_numpy(dtype=str)):
        raise ValueError(f'{pack}: replay ids differ from the parent query ids')
    with np.load(require(Path(model_root) / relative), allow_pickle=True) as a:
        values = {k: a[k] for k in a.files}
    values.update(test_ids=tids, test=test)
    path = destination(work, relative)
    save_npz(path, **values)
    return path


def frozen_stage(stage, model_root, work):
    tag = stage['argv'][1]
    path = _frozen_full(model_root, work, PACK / f'wn_stage_{tag}_candidate', WS / f'frozen_stage_{tag}/full.npz')
    return _copy_rest(stage, model_root, work, [path])


def freeze_agreement(stage, model_root, work):
    name = stage['argv'][1] if len(stage['argv']) > 1 else 'local_rank_family_agreement'
    pack = PACK / f'wn_{name}_candidate'
    path = _frozen_full(model_root, work, pack, WS / f'frozen_{name}/full.npz')
    with np.load(path) as a:
        tids, p = a['test_ids'].copy(), a['test'].copy()
    ref = pd.read_csv(require(Path(work) / pack / 'submission.csv'), dtype={'id': str}).set_index('id').loc[tids, 'break_probability']
    if np.max(np.abs(expit(p) - ref.to_numpy())) >= 1e-12:
        raise ValueError(f'{name}: frozen test logits do not replay the candidate submission')
    return _copy_rest(stage, model_root, work, [path])


def view_contrasts(stage, model_root, work):
    #white_view_contrasts.generate, query rows only (pure transform of new view scores, nothing learned)
    tag, runs = stage['argv'][1], stage['argv'][2:]
    work = Path(work)
    out, root = work / WS, work
    _, tids, _ = _parent(work)
    ids = tids
    groups, sources = {}, {}
    for name in runs:
        path = require(root / f'tune_out/ver_{name}.npz')
        sources[path.name] = _sha(path)
        with np.load(path) as q:
            views = tuple(q['kinds'].astype(str))
            assert views[0] == 'orig' and len(set(views)) == len(views)
            value = q['test']
            assert value.shape == (len(views), len(tids))
            groups.setdefault(views, []).append(logit(value.clip(1e-06, 0.999999)))

    def table(part, ids, v, name):
        base = out / f'view_{v}' if v else out
        if name == 'n':
            path = base / f'clean/features_{part}.parquet' if v else root / f'clean/features_{part}.parquet'
            return pd.read_parquet(require(path), columns=['n_pre', 'n_post']).loc[ids].to_numpy()
        return pd.read_parquet(require(base / f'{name}_{part}.parquet')).loc[ids]
    part = 'test'
    columns = {}
    n = table(part, ids, '', 'n')
    size = 1 / (1 / n[:, 0] + 1 / n[:, 1])
    wk = table(part, ids, '', 'innov_v2').pre_walk.to_numpy()
    left = table(part, ids, '', 'volatility_law').pre_cluster.to_numpy()
    right = table(part, ids, '', 'volatility_law').post_cluster.to_numpy()
    white = table(part, ids, '', 'innov_v2').pre_white.to_numpy()
    for views, members in groups.items():
        scores = np.mean(members, 0)
        orig = scores[0]
        prefix = '_'.join(views)
        crops = [(i, v) for i, v in enumerate(views) if v.startswith('c')]
        resid, fractions, asym = [], [], []
        for i, v in crops:
            dims = table(part, ids, v, 'n')
            ratio = 1 / (1 / dims[:, 0] + 1 / dims[:, 1]) / size
            fractions.append(ratio)
            asym.append(dims[:, 0] / n[:, 0] - dims[:, 1] / n[:, 1])
            resid.append((scores[i] - orig).clip(-5, 5))
        if resid:
            residual = np.stack(resid)
            frac = np.stack(fractions)
            small = residual.mean(0)
            centered = small - orig * (np.sqrt(frac.mean(0)).clip(0.25, 1) - 1)
            for key, z in (('crop', small), ('size_adjusted', centered), ('imbalance', (residual * np.stack(asym)).mean(0)), ('abs', abs(small)), ('disagreement', residual.std(0)), ('direction', np.tanh(orig / 2) * small)):
                columns[prefix + '_' + key] = z
        if 'rev' in views:
            r = (scores[views.index('rev')] - orig).clip(-6, 6)
            rev_white = table(part, ids, 'rev', 'innov_v2').pre_white.to_numpy()
            for key, z in (('reversal', r), ('length', r * (n[:, 0] - n[:, 1]) / n.sum(1)), ('vol', r * (left - right)), ('white', r * (white - rev_white)), ('walk', r * wk), ('absolute', abs(r)), ('confidence', r * np.tanh(orig / 2))):
                columns[prefix + '_' + key] = z
    df = pd.DataFrame(columns, index=pd.Index(ids, name='id')).astype(np.float32)
    assert np.isfinite(df).all().all()
    train = pd.read_parquet(require(Path(model_root) / WS / f'{tag}_train.parquet'))
    if not df.columns.equals(train.columns):
        raise ValueError(f'{tag}: query columns differ from the saved train table')
    path = destination(work, WS / f'{tag}_test.parquet')
    save_table(path, df)
    js = destination(work, WS / f'{tag}_sources.json')
    js.write_text(json.dumps(sources, indent=2))
    return _copy_rest(stage, model_root, work, [path, js])


def _local_rank(stage, model_root, work, name, builder):
    #white_local_rank_weights.run export branch with the saved weights and centering grid
    rank = _mod('white_local_rank_weights')
    spec = json.loads(require(Path(model_root) / PACK / f'wn_{name}_candidate/weights.json').read_text())
    ids, tids, _ = _parent(work)
    with np.load(require(Path(work) / WS / f"frozen_stage_{spec['parent']}/full.npz")) as a:
        if not np.array_equal(tids, a['test_ids']):
            raise ValueError(f'{name}: frozen parent test ids differ')
        zt = a['test'].copy()
    make, _ = builder(ids, tids, spec['models'], spec['physics'])
    te = len(ids) + np.arange(len(tids))
    v, _ = rank.centered(make(te, zt), zt, np.array(spec['centers']))
    _, written = _candidate(model_root, work, name, zt, v, tids, ('ids', 'order', 'parent_logits', 'features'))
    return _copy_rest(stage, model_root, work, written)


def family_agreement(stage, model_root, work):
    args = stage['argv'][1:]
    parent, tag = args[0], args[1]
    extras = args[args.index('--new') + 1:] if '--new' in args else []
    ea = _mod('white_expert_agreement')
    name = 'local_rank_' + tag
    spec = json.loads(require(Path(model_root) / PACK / f'wn_{name}_candidate/weights.json').read_text())
    models = [*ea.GROUPS['crop_trees'][:3], *ea.GROUPS['rank_trees'], 'lawforecast', 'lawretrieval', *extras]
    if spec['models'] != models or spec['parent'] != parent:
        raise ValueError(f'{name}: saved spec differs from the stage arguments')
    #constructor wraps the unpatched white_local_rank_weights.inputs, as in the script's __main__
    return _local_rank(stage, model_root, work, name, ea.constructor(extras))


def _read_data_no_hash(work):
    #the slice of expert_fusion.read_data white_baseline_opinions uses (ids, tids, member oof logits, context
    #descriptors); the ensemble.json input-hash check is skipped since query member files are new by design
    ef = _mod('expert_fusion')
    root = Path(work)
    ids, tids = [pd.read_parquet(require(root / f'clean/features_{p}.parquet'), columns=[]).index for p in ('train', 'test')]
    names = json.loads(require(root / 'submissions/sub_seqw5_stack.json').read_text())['members']
    paths = [ef.OLD / 'oof_head_v2_bag.npy' if n == 'MLPHV_bag_w' else ef.O / f'oof_{n}.npy' for n in names]
    oof = np.column_stack([logit(np.clip(np.load(require(p)), 1e-06, 1 - 1e-06)) for p in paths]).astype(float)
    sides = []
    for part, index in (('train', ids), ('test', tids)):
        blocks = [pd.read_parquet(require(root / f'{stem}_{part}.parquet'), columns=cols).loc[index, cols] for stem, cols in ef.CONTEXT.items()]
        assert all((b.index.equals(index) and b.index.is_unique for b in blocks))
        sides.append(np.column_stack(blocks))
    context = np.vstack(sides)
    data = [None] * 13
    data[1], data[2], data[5], data[11], data[12] = ids, tids, oof, context, context.shape[1]
    return tuple(data)


def early_opinions(stage, model_root, work):
    tag, extra = stage['argv'][1], stage['argv'][2:]
    bo = _mod('white_baseline_opinions')
    name = 'local_rank_' + tag
    spec = json.loads(require(Path(model_root) / PACK / f'wn_{name}_candidate/weights.json').read_text())
    models = [*bo.GROUPS['crop_trees'][:3], *bo.GROUPS['rank_trees'], 'lawforecast', 'lawretrieval', *extra]
    if spec['models'] != models or spec['parent'] != 'reversible_rank':
        raise ValueError(f'{name}: saved spec differs from the stage arguments')
    data = _read_data_no_hash(work)
    with patch.object(bo.ef, 'read_data', lambda: data):
        return _local_rank(stage, model_root, work, name, bo.inputs)


ADAPTERS = {
    'white_loop.py': parent_cache, 'white_consult.py': consult, 'white_stage.py': stage_head,
    'white_frozen_stage.py': frozen_stage, 'white_view_contrasts.py': view_contrasts,
    'white_expert_agreement.py': family_agreement, 'white_freeze_agreement.py': freeze_agreement,
    'white_baseline_opinions.py': early_opinions,
}


def supported(stage):
    return Path(stage['argv'][0]).name in ADAPTERS


def predict_stage(stage, model_root, work):
    return ADAPTERS[Path(stage['argv'][0]).name](stage, Path(model_root), Path(work))
