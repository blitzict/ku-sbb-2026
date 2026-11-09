"""Saved late heads, group E2: channel replays, reliable signals, local-rank readouts and the final layer.

Every adapter keeps the train/OOF side of its outputs verbatim from model_root and recomputes only the query (test)
side from the new upstream test outputs in work. Learned readout coefficients come from the saved
new weights/wn_<name>_candidate/weights.json; sequence networks come from model_root's checkpoints. Design matrices
are rebuilt with the training source's own functions; their train-row statistics (quantiles, centering grids,
eigenvectors, residualisation) are deterministic functions of the verbatim training rows. Nothing is refitted:
white_nested.fit / white_local_rank_weights.weights / physical_870.fit are never called.
"""
import hashlib
import importlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from .common import destination, require, replace_test_fields, save_npz, save_table

WS = Path('tune_out/weight_space')
NW = Path('new weights')
SEEDS = (7, 29, 0, 123, 2024, 11041)



def _copy(model_root, work, relative):
    #train-side artifact or saved model state: verbatim copy
    target = destination(work, relative)
    temporary = target.with_name(target.name + '.prediction-partial')
    shutil.copyfile(require(Path(model_root) / relative), temporary)
    temporary.replace(target)
    return target


def _spec(model_root, name):
    return json.loads(require(Path(model_root) / NW / f'wn_{name}_candidate/weights.json').read_text())


def _load(work):
    #white_loop.load without its train-only diagnostics (labels, cohorts, weights): split rows and identifiers
    wl = importlib.import_module('white_loop')
    with np.load(require(Path(work) / WS / 'white_loop_parent.npz')) as cache:
        q = {k: cache[k] for k in cache.files}
    return wl, q, pd.Index(q['ids']), pd.Index(q['tids'])


def _package(model_root, work, wl, name, tids, logits, features, coefficients):
    """white_loop/fixed_final_readouts save(): new submission + replay inputs, saved recipe and reports verbatim."""
    folder = NW / f'wn_{name}_candidate'
    written = [_copy(model_root, work, WS / f'{name}.json'), _copy(model_root, work, WS / f'{name}_oof.npz')]
    written += [_copy(model_root, work, folder / f) for f in ('weights.json', 'validation.json', 'replay.py')]
    csv = destination(work, folder / 'submission.csv')
    temporary = csv.with_name('submission.prediction-partial.csv')
    sub = wl.ef.write_submission(expit(logits + features @ coefficients), tids, temporary)
    temporary.replace(csv)
    save_npz(destination(work, folder / 'replay_inputs.npz'), ids=tids.to_numpy(dtype=str),
             order=tids.get_indexer(sub.id), parent_logits=logits, features=features)
    return written + [csv, Path(work) / folder / 'replay_inputs.npz']


def _test_scores(work, model_root, name, tids):
    #white_paired_branches.test_scores on the replay inputs written in work, saved weights from model_root
    with np.load(require(Path(work) / NW / f'wn_{name}_candidate/replay_inputs.npz')) as a:
        if not np.array_equal(a['ids'], tids):
            raise ValueError(f'{name}: replay query identifiers differ')
        return a['parent_logits'] + a['features'] @ _spec(model_root, name)['weights']


#white_channel_replay.py

def _check_reproduction(pred, orig):
    #periodic_replay's own guard: replayed original-series logits must match the sequence models' saved test logits
    err = abs(pred - orig)
    assert err.mean() < 0.002 and err.max() < 0.05, (err.mean(), err.max())


def _fold_models(model_root, seq, fold):
    import torch
    manifest = json.loads(require(Path(model_root) / NW / 'model_manifest.json').read_text())
    hashes = {Path(row['file']): row['sha256'] for row in manifest['files']}
    models = []
    for run, (hq, seed) in seq.RUNS.items():
        seq.a.hq = hq
        model = seq.Net().eval().to('cpu')
        rel = Path('models') / run / f'fold{fold}_seed{seed}.pt'
        ck = require(Path(model_root) / NW / rel)
        if hashlib.sha256(ck.read_bytes()).hexdigest() != hashes[rel]:
            raise ValueError(f'{ck}: checkpoint hash differs from manifest')
        model.load_state_dict(torch.load(ck, weights_only=True, map_location='cpu'))
        models.append((hq, model))
    return models


def _fold_scores(replay, models, gpu, gate, orig, rows):
    """periodic_replay.run per-fold query block: [original mean, common, separate] logits for the given rows."""
    import torch
    selected = torch.as_tensor(rows, device='cpu')
    before = orig.mean(1).T
    scores = [before]
    for i, x in enumerate(gpu[:3]):
        pred = replay.predict(models, x[selected], gpu[3][selected], gpu[4][selected])
        if i == 0:
            _check_reproduction(pred, orig)
        else:
            result = before.copy()
            active = gate[rows] > 0.075
            result[active] = pred.mean(1).T[active]
            scores.append(result)
    return np.column_stack(scores)


def channels(stage, model_root, work):
    import torch
    torch.set_num_threads(1)
    tag = stage['argv'][1]
    replay = importlib.import_module('periodic_replay')
    seq = replay.seq
    tids, gpu, gate = replay.streams('test', tag)
    every = np.arange(len(tids))
    written = []
    for fold in range(5):
        relative = WS / f'{tag}_replay/fold{fold}.npz'
        with np.load(require(Path(model_root) / relative)) as a:
            saved = {k: a[k] for k in a.files}
        models = _fold_models(model_root, seq, fold)
        orig = []
        for run in seq.RUNS:
            with np.load(require(Path(work) / f'tune_out/seqpart/{run}_f{fold}.npz')) as a:
                orig.append(a['pt'])
        test = _fold_scores(replay, models, gpu, gate, np.stack(orig), every)
        target = destination(work, relative)
        temporary = target.with_name(target.name + '.prediction-partial')
        with temporary.open('wb') as stream:
            np.savez(stream, ids=saved['ids'], tids=tids.to_numpy(dtype=str), val=saved['val'], test=test)
        temporary.replace(target)
        written.append(target)
    return written + _assemble(model_root, work, tag, tids, seq)


def _assemble(model_root, work, tag, tids, seq):
    written = []
    #periodic_replay.run with every archive present: query = mean of the fold archives (train side verbatim)
    out = Path(work) / WS
    test = np.zeros((len(tids), 3 * len(seq.RUNS)))
    for fold in range(5):
        with np.load(out / f'{tag}_replay/fold{fold}.npz') as a:
            if not np.array_equal(a['tids'], tids):
                raise ValueError(f'{tag}: fold archive query identifiers differ')
            test = test + a['test'] / 5
    cols = [prefix + '_' + t for prefix in ('original', 'common', 'separate') for t in seq.RUNS]
    if not np.isfinite(test).all():
        raise ValueError(f'{tag}: replay scores are not finite')
    save_table(destination(work, WS / f'{tag}_seq_test.parquet'), pd.DataFrame(test, index=tids, columns=cols))
    #white_channel_replay.run, query part
    seq_tab = pd.read_parquet(out / f'{tag}_seq_test.parquet')
    pulse = pd.read_parquet(out / 'pulse_seq_test.parquet').loc[seq_tab.index]
    gate = pd.read_parquet(out / f'{tag}_test.parquet').loc[seq_tab.index, 'gate'].to_numpy()
    old = pulse.filter(regex='^separate_').to_numpy()
    a = seq_tab.filter(regex='^common_').to_numpy()
    b = seq_tab.filter(regex='^separate_').to_numpy()
    assert old.shape == a.shape == b.shape and a.shape[1] > 0
    d, e = ((a - old).clip(-8, 8), (b - old).clip(-8, 8))
    means = [d.mean(1), e.mean(1), abs(d).mean(1), abs(e).mean(1), (a - b).mean(1)]
    features = np.column_stack([*means, *(gate * v for v in means)])
    tab = pd.DataFrame(features, index=seq_tab.index, columns=[f'v{i}' for i in range(features.shape[1])])
    assert np.isfinite(features).all()
    save_table(destination(work, WS / f'{tag}_channel_effect_test.parquet'), tab)
    scores = np.column_stack([expit(old.mean(1) + gate * delta) for delta in means[:2]])
    for part in ('seq', 'channel_effect'):
        written.append(_copy(model_root, work, WS / f'{tag}_{part}_train.parquet'))
        written.append(out / f'{tag}_{part}_test.parquet')
    for k, label in enumerate(('core', 'other')):
        relative = WS / f'law_{tag}_{label}.npz'
        replace_test_fields(model_root, work, relative, tids=seq_tab.index.to_numpy(dtype=str), test=scores[:, k])
        written.append(out / f'law_{tag}_{label}.npz')
    return written


#white_reliable_signals.py

def reliable(stage, model_root, work):
    #white_reliable_signals.run, query part only (fixed transforms, no learned state)
    _, _, _, ix = _load(work)
    out = Path(work) / WS
    get = lambda name: pd.read_parquet(out / f'{name}_test.parquet').loc[ix]
    s = get('mode_residual')
    h = get('mode_residual_channel_effect')
    edge = get('tail_edges')
    tail = get('tail_channels')
    a = get('tail_channels_channel_effect')
    gate = ((s.gate.to_numpy() - 0.5) / 0.45).clip(0, 1)
    gtail = ((tail.gate.to_numpy() - 0.65) / 0.32).clip(0, 1)
    output = {}
    d, e = (h.v0.to_numpy(), h.v1.to_numpy())
    for name, v in (('within_mode', d), ('occupancy', e), ('agree', np.sign(d + e) * np.minimum(abs(d), abs(e))), ('within_given_motion', d * np.tanh(abs(s.residual_change) / 2)), ('visits_given_counts', e * np.tanh(abs(s.occupancy_change) / 2))):
        output['mode_' + name] = gate * v
    for name, v in (('core', a.v0), ('bursts', a.v1), ('core_given_shift', a.v0 * np.tanh(abs(tail.core_location) / 2))):
        output['tail_' + name] = gtail * v
    for side in ('up', 'down'):
        band = edge.filter(like='normalized_' + side + '_').copy()
        band.columns = band.columns.str.removeprefix('normalized_' + side + '_')
        g = ((np.maximum(band.pre_bounded, band.post_bounded) - 0.55) / 0.4).clip(0, 1)
        output['edge_' + side + '_move'] = g * np.arcsinh(band.bounded_move.clip(-6, 15))
        output['edge_' + side + '_run'] = g * np.arcsinh(band.tail_run.clip(-6, 15))
    df = pd.DataFrame(output, index=ix).rename_axis('id')
    if not np.isfinite(df).all().all():
        raise ValueError('reliable signals are not finite')
    save_table(destination(work, WS / 'reliable_signals_test.parquet'), df)
    return [_copy(model_root, work, WS / 'reliable_signals_train.parquet'), Path(work) / WS / 'reliable_signals_test.parquet']


#white_signal_audit.py: OOF ablation report only

def audit(stage, model_root, work):
    return [_copy(model_root, work, rel) for rel in stage['outputs']]


#white_signal_package.py

def package(stage, model_root, work):
    parent, name, signal = stage['argv'][1:4]
    wl, q, ids, tids = _load(work)
    centered = importlib.import_module('white_local_rank_weights').centered
    table, cols = importlib.import_module('white_signal_audit').terms(signal)
    tag = 'local_rank_' + name
    spec = _spec(model_root, tag)
    if spec['columns'] != cols or spec['signal'] != signal or spec['parent'] != parent:
        raise ValueError(f'{tag}: saved recipe does not match the stage')
    out = Path(work) / WS
    a = pd.read_parquet(out / f'{table}_test.parquet', columns=cols).loc[tids]
    h = np.arcsinh(a.to_numpy().clip(-24, 55))
    g = pd.read_parquet(out / 'innov_v2_test.parquet', columns=['pre_white', 'pre_walk']).loc[tids]
    features = np.column_stack([h, 0.6 * h * g.pre_white.to_numpy()[:, None], 0.6 * h * g.pre_walk.to_numpy()[:, None]])
    with np.load(require(out / f'frozen_{parent}/full.npz')) as r:
        if not np.array_equal(tids, r['test_ids']):
            raise ValueError(f'{parent}: frozen query identifiers differ')
        v = r['test']
    val, _ = centered(features, v, np.array(spec['centers']))
    return _package(model_root, work, wl, tag, tids, v, val, np.array(spec['weights']))


#white_fold_vote.py

def fold_vote(stage, model_root, work):
    tag, method, *members = stage['argv'][1:]
    module = importlib.import_module('white_fold_vote')
    read_member = importlib.import_module('white_joint').read_member
    wl, q, ids, tids = _load(work)
    n = len(ids)
    name = 'local_rank_' + tag
    spec = _spec(model_root, name)
    if spec['members'] != members or spec['method'] != method:
        raise ValueError(f'{name}: saved recipe does not match the stage')
    #frozen parent: validation folds are train side, full.npz gets the parent's new query logits
    cache = WS / ('frozen_' + module.PARENT)
    written = [_copy(model_root, work, cache / f's{s}_f{f}.npz') for s in SEEDS for f in range(5)]
    v = _test_scores(work, model_root, module.PARENT, tids)
    replace_test_fields(model_root, work, cache / 'full.npz', test_ids=tids.to_numpy(dtype=str), test=v)
    written.append(Path(work) / cache / 'full.npz')
    x = np.column_stack([read_member(m, ids, tids)[0][n:, 0] for m in members])
    reg = pd.read_parquet(Path(work) / WS / 'innov_v2_test.parquet', columns=['pre_white', 'pre_walk']).loc[tids].to_numpy()
    a, b, c = spec['parent_quantiles']
    lo, med, hi = (np.array(u) for u in spec['member_quantiles'])
    calibrated = (x - med) * ((c - a) / (hi - lo)) + b
    d = calibrated - v[:, None]
    if method == 'physical':
        white = reg[:, 0] * np.exp(-0.5 * ((v + 1.1) / 1.25) ** 2)
        F = d * (1.0 - 0.85 * white[:, None])
    elif method == 'fit':
        F = np.column_stack((d, d * reg[:, :1], d * reg[:, 1:2], d * np.tanh(v[:, None] / 2)))
    else:
        F = d
    return written + _package(model_root, work, wl, name, tids, v, F, np.array(spec['weights']))


def _upstream_raw(work):
    stack = importlib.import_module('white_upstream_stack')
    wl, q, ids, tids = _load(work)
    raw = stack.collect(ids, tids)
    n = len(ids)
    matrix = stack.train_basis(raw, np.arange(n))
    return wl, ids, tids, n, raw[0], matrix


FIRST = 'local_rank_first_layer_overall'
FINAL = 'local_rank_shared_residual_response'


#white_upstream_stack.py

def upstream(stage, model_root, work):
    blend = float(stage['argv'][stage['argv'].index('--blend') + 1]) if '--blend' in stage['argv'] else 0.25
    wl, ids, tids, n, guide, matrix = _upstream_raw(work)
    #full-train branch coefficient: stored (times the .25 blend) as weights[1:] of the first-layer readout
    spec = _spec(model_root, FIRST)
    #first-layer readout weights are blend * [1, coefficient] (conditional=False branch)
    if spec.get('conditional', False) or spec.get('blend', blend) != blend or spec['weights'][0] != blend:
        raise ValueError('first-layer recipe does not carry this upstream branch')
    coefficient = np.array(spec['weights'][1:]) / blend
    te = n + np.arange(len(tids))
    branch = guide[te] + matrix[te] @ coefficient
    replace_test_fields(model_root, work, WS / 'local_rank_upstream_joint_oof.npz',
                        tids=tids.to_numpy(dtype=str), branch_test=branch)
    return [_copy(model_root, work, WS / 'local_rank_upstream_joint.json'), Path(work) / WS / 'local_rank_upstream_joint_oof.npz']


#pipeline_helpers/fixed_final_readouts.py

def readouts(stage, model_root, work):
    mode = stage['argv'][1]
    if mode == 'upstream-inner':
        #inner branches use training rows only
        return [_copy(model_root, work, rel) for rel in stage['outputs']]
    if mode == 'first-layer':
        base = importlib.import_module('white_crop_replication').BASE
        wl, ids, tids, n, guide, matrix = _upstream_raw(work)
        te = n + np.arange(len(tids))
        baseline = _test_scores(work, model_root, base, tids)
        features = np.column_stack([guide[te] - baseline, matrix[te]])
        return _package(model_root, work, wl, FIRST, tids, baseline, features, np.array(_spec(model_root, FIRST)['weights']))
    if mode == 'shared-response':
        spectrum = importlib.import_module('white_opinion_spectrum')
        shared = importlib.import_module('white_shared_residual')
        read_member = importlib.import_module('white_joint').read_member
        wl, q, ids, tids = _load(work)
        banks, regime, _ = spectrum.inputs(ids, tids)
        extra = np.column_stack([read_member(name, ids, tids)[0][:, 0] for name in shared.MEMBERS])
        n = len(ids)
        with np.load(require(Path(work) / WS / f'frozen_{FIRST}/full.npz')) as a:
            if not (np.array_equal(a['train_ids'], ids) and np.array_equal(a['test_ids'], tids)):
                raise ValueError('frozen first layer identifiers differ')
            z, v = a['oof_train'], a['test']
        matrix = shared.chart(banks, regime, extra, np.arange(n), n + np.arange(len(tids)), z, v)['response']
        return _package(model_root, work, wl, FINAL, tids, v, matrix[n:], np.array(_spec(model_root, FINAL)['weights']))
    raise ValueError(f'unknown fixed readout {mode}')


#white_freeze_first_layer.py

def freeze_first(stage, model_root, work):
    wl, q, ids, tids = _load(work)
    folder = WS / f'frozen_{FIRST}'
    written = [_copy(model_root, work, folder / f's{s}_f{f}.npz') for s in SEEDS for f in range(5)]
    replace_test_fields(model_root, work, folder / 'full.npz', test_ids=tids.to_numpy(dtype=str),
                        test=_test_scores(work, model_root, FIRST, tids))
    return written + [Path(work) / folder / 'full.npz']


ADAPTERS = {
    'white_channel_replay.py': channels, 'white_reliable_signals.py': reliable, 'white_signal_audit.py': audit,
    'white_signal_package.py': package, 'white_fold_vote.py': fold_vote, 'white_upstream_stack.py': upstream,
    'fixed_final_readouts.py': readouts, 'white_freeze_first_layer.py': freeze_first,
}
