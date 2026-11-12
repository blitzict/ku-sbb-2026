# CPU port; original experiments/weight_space/white_kernel_readout.py, SHA256 13fbbf53cd649c0b4413b9f1ed4a88b56c4e3cea9b7098779e1e0a0d3e1d9385
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Learn local score adjustments around landmarks selected from real training series."""
import contextlib as _cpu_context
import os as _cpu_os
from pathlib import Path as _CpuPath

def _cpu_workspace():
    value = _cpu_os.environ.get('KU_SBB_ROOT')
    if not value:
        raise RuntimeError('Set KU_SBB_ROOT to the fresh CPU rebuild workspace')
    root = _CpuPath(value).resolve()
    if not (root / 'CPU_REBUILD_WORKSPACE.json').is_file():
        raise RuntimeError('CPU ports require a marked rebuild workspace')
    return root
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.spatial.distance import cdist
from scipy.special import expit
from threadpoolctl import threadpool_limits
from white_checkpoint import SEEDS
from white_joint import read_member
from white_nested import audit
from white_freeze_first_layer import NAME
from partition_evidence import ROOT, OUT
import white_loop as wl

def panels(ids, tids):
    deps = []

    def table(tag, cols=None):
        chunks = []
        for part, ix in (('train', ids), ('test', tids)):
            p = OUT / f'{tag}_{part}.parquet'
            deps.append(p)
            chunks.append(pd.read_parquet(p, columns=cols).loc[ix])
        return pd.concat(chunks)

    def experts(names):
        scores = []
        for name in names:
            h, p = read_member(name, ids, tids)
            deps.extend(p)
            scores.append(h[:, 0])
        return np.column_stack(scores)
    votes = experts(('claude_joint_three_splits', 'law_independent_rank_bag_29_7', 'law_linked_plus_rank', 'claude_revorig', 'lawforecast', 'lawblock', 'lawshared', 'lawretrieval', 'net', 'lawpredictive', 'law_rank_dependence_s29'))
    shap = table('tree_channels_claude')
    shape = table('jump_law_state', ['shape0_robust', 'shape1_robust', 'level2', 'scale2'])
    smooth = table('jump_law_smoothvol', ['shape0', 'shape1', 'level2', 'scale2'])
    flow = table('volatility_law', ['pre_cluster', 'post_cluster', 'pooled_cluster', 'new_memory', 'new_law', 'new_scale'])
    features = np.column_stack([shape, smooth, flow, table('phase_law', ['mixture_evidence', 'phase_bf']), table('quantile_alignment', ['state_shoulders_after_location_scale', 'pulse_tails_after_location_scale']), table('reliable_signals', ['mode_within_mode', 'mode_occupancy', 'mode_agree'])])
    gates = np.column_stack([table('innov_v2', ['pre_white', 'pre_walk']), table('curve', ['gate']), table('periodic', ['gate']), flow.pre_cluster])
    return (votes, shap.to_numpy(), features, gates, deps)

def rank_scale(train, test):
    p = np.linspace(0.01, 0.99, 35)
    bounds = np.quantile(train, p, axis=0)
    xx = []
    for val, axis in zip(test.T, bounds.T):
        knots, pos, count = np.unique(axis, return_index=True, return_counts=True)
        levels = np.add.reduceat(p, pos) / count
        xx.append(np.interp(val, knots, levels) if len(knots) > 1 else np.full(len(val), 0.5))
    return np.column_stack(xx)

def basis(raw, tr, va, z, v):
    vote, shap, phys, gates, _ = raw
    rows = np.r_[tr, va]
    n = len(tr)
    guide = vote[rows, :3].mean(1)
    d = vote[rows] - guide[:, None]
    allz = np.r_[z, v]
    parts = (d, shap[rows], phys[rows])
    rng = np.random.default_rng(316)
    landmarks = np.concatenate([rng.choice(ix, min(len(ix), 12), replace=False) for ix in np.array_split(np.argsort(z), 20)])
    terms = []
    for raw_panel in parts:
        normalized = rank_scale(raw_panel[:n], raw_panel)
        ctx = rank_scale(gates[tr], gates[rows])
        x = np.column_stack([2.0 * normalized / np.sqrt(normalized.shape[1]), 0.57 * ctx, 0.75 * np.tanh(allz / 2.5)])
        for width in (0.55, 1.1):
            kernel = np.exp(-cdist(x, x[landmarks], metric='sqeuclidean') / (2 * width ** 2))
            gram = kernel[landmarks].copy()
            gram.flat[::len(gram) + 1] += 0.06
            phi = solve_triangular(np.linalg.cholesky(gram), kernel.T, lower=True, check_finite=False).T
            terms.append(phi)
    return np.column_stack(terms)

def edit(raw, tr, va, z, v, y):
    h = basis(raw, tr, va, z, v)
    x, u = (h[:len(tr)], h[len(tr):])
    p = expit(z)
    hess = p * (1 - p)
    A = x.T @ (x * hess[:, None])
    A.flat[::len(A) + 1] += 18.0
    w = cho_solve(cho_factor(A, check_finite=False), x.T @ (y[tr] - p), check_finite=False)
    return (v + u @ w, u, w)

def main():
    q, ids, tids, y, wt, wn, miss = wl.load()
    raw = panels(ids, tids)
    n = len(y)
    archive = np.load(OUT / (NAME + '_oof.npz'))
    parent = ROOT / f'new weights/wn_{NAME}_candidate'
    digest = wl.ef.sha(parent / 'weights.json')
    cache = OUT / ('frozen_' + NAME)
    scores, ops, pred = ({}, {}, {})
    for s in SEEDS:
        new, locked = (np.empty(n), np.empty(n))
        ref = archive[f's{s}']
        for f in range(5):
            k = f's{s}_f{f}_'
            tr, va = (q[k + 'tr'], q[k + 'va'])
            with np.load(cache / f's{s}_f{f}.npz') as a:
                assert str(a['parent_sha']) == digest and np.array_equal(a['train_ids'], ids[tr]) and np.array_equal(a['validation_ids'], ids[va])
                z, v = (a['oof_train'], a['valid'])
            assert np.max(abs(v - ref[va])) < 1e-07
            new[va] = edit(raw, tr, va, z, v, y)[0]
            locked[va] = q[k + 'zv']
        scores[s] = {k: wl.score(y, v, wt, wn, miss) for k, v in (('locked', locked), ('previous', ref), ('new', new))}
        ops[s] = audit(y, miss, ref, new)
        pred[f's{s}'] = new
        print(s, {k: round(scores[s]['new'][k], 7) for k in ('auc', 'target_vs_neg', 'remaining')}, ops[s]['candidate'], flush=True)
    tag = 'local_rank_kernel_readout'
    passed = all((a['new']['auc'] > a['previous']['auc'] and all((a['new'][k] > a['locked'][k] for k in ('remaining', 'target_vs_neg'))) for a in scores.values()))
    result = dict(accepted=bool(passed), scores=scores, audits=ops)
    (OUT / (tag + '.json')).write_text(json.dumps(result, indent=2))
    np.savez_compressed(OUT / (tag + '_oof.npz'), ids=ids.to_numpy(dtype=str), **pred)
    if passed:
        with np.load(cache / 'full.npz') as a:
            assert str(a['parent_sha']) == digest and np.array_equal(a['test_ids'], tids)
            z, v = (a['oof_train'], a['test'])
        new, h, w = edit(raw, np.arange(n), n + np.arange(len(tids)), z, v, y)
        folder = ROOT / f'new weights/wn_{tag}_candidate'
        folder.mkdir(exist_ok=True)
        csv, rec = (folder / 'submission.csv', folder / 'weights.json')
        sub = wl.ef.write_submission(expit(new), tids, csv)
        deps = [Path(__file__).resolve(), parent / 'weights.json', *raw[-1]]
        rec.write_text(json.dumps(dict(weights=w.tolist(), hashes={str(p.relative_to(ROOT)): wl.ef.sha(p) for p in dict.fromkeys(deps)}), indent=2))
        np.savez_compressed(folder / 'replay_inputs.npz', ids=tids.to_numpy(dtype=str), order=tids.get_indexer(sub.id), parent_logits=v, features=h)
        (folder / 'validation.json').write_text(json.dumps(dict(**result, recipe_sha256=wl.ef.sha(rec), submission_sha256=wl.ef.sha(csv)), indent=2))
        shutil.copyfile(parent / 'replay.py', folder / 'replay.py')
if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
