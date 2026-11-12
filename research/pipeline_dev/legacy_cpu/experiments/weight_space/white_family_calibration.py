# CPU port; original experiments/weight_space/white_family_calibration.py, SHA256 223479007fb20ee17b9ce694a252d80aa28cb6dfd34d94f243262fa56c6ca675
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Calibrate starting odds across observed process families without using historical miss IDs."""
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
from scipy.special import expit
from threadpoolctl import threadpool_limits
from white_nested import audit
from white_checkpoint import SEEDS
from partition_evidence import ROOT, OUT
import white_loop as wl
BASE = 'local_rank_without_multiview_trees'
GRID = np.linspace(-5.5, 8, 28)

def groups(ids, tids):
    ix = ids.append(tids)
    source = ROOT / 'tune_out/generator_segments.parquet'
    df = pd.read_parquet(source, columns=['id', 'part', 'p_bic', 'phi1_ar1', 't_df'])
    pre = df[df.part == 'pre'].set_index('id').loc[ix]
    paths = [source]

    def values(table, col):
        pieces = []
        for part, rows in (('train', ids), ('test', tids)):
            f = OUT / f'{table}_{part}.parquet'
            paths.append(f)
            pieces.append(pd.read_parquet(f, columns=[col]).loc[rows, col].to_numpy())
        return np.concatenate(pieces)
    periodic = values('periodic', 'pre_fraction')
    phi = pre.phi1_ar1.to_numpy()
    family = np.select([phi > 0.95, periodic > 0.35, pre.p_bic == 0, phi < -0.18, phi > 0.18], [0, 1, 2, 3, 4], default=5)
    shape = np.select([values('mixture_law', 'pre_bim') > 0.6, values('volatility_law', 'pre_cluster') > 0.6, pre.t_df < 8], [0, 1, 2], default=3)
    return (family, shape, paths)

def fit_curves(z, y, g, n, penalty):
    kernels = np.exp(-0.5 * ((z[:, None] - GRID) / 0.65) ** 2)
    curves = []
    for j in range(n):
        rows = g == j
        mass = kernels[rows]
        local_z, local_y = (z[rows, None], y[rows, None])
        u = np.zeros(len(GRID))
        for _ in range(6):
            p = expit(local_z + u)
            grad = np.sum(mass * (local_y - p), 0) - penalty * u
            curvature = np.sum(mass * p * (1 - p), 0) + penalty
            u += grad / curvature
        curves.append(u)
    return np.array(curves)

def derive(tr, va, z, v, y, fam, shape):
    n = len(tr)
    global_curve = fit_curves(z, y[tr], np.zeros(n, int), 1, 18)[0]

    def eval_group(g, num, penalty):
        cov = fit_curves(z, y[tr], g[tr], num, penalty)
        score = np.empty(len(va))
        for j in range(num):
            sel = g[va] == j
            score[sel] = np.interp(v[sel], GRID, cov[j] - global_curve)
        return (score, cov)
    f, fc = eval_group(fam, 6, 22)
    s, sc = eval_group(shape, 4, 25)
    joint, jc = eval_group(4 * fam + shape, 24, 38)
    return (np.column_stack([f, s, joint]), dict(global_curve=global_curve.tolist(), family=fc.tolist(), shape=sc.tolist(), joint=jc.tolist()))

def main():
    q, ids, tids, y, wt, wn, miss = wl.load()
    n = len(y)
    fam, shape, paths = groups(ids, tids)
    cache = OUT / ('frozen_' + BASE)
    parent = ROOT / f'new weights/wn_{BASE}_candidate'
    digest = wl.ef.sha(parent / 'weights.json')
    rules = {'pre_dynamics': np.array([0.45, 0.0, 0.0]), 'pre_laws': np.array([0.18, 0.15, 0.2])}
    scores = {k: {} for k in rules}
    arr = {k: {} for k in rules}
    auditlog = {k: {} for k in rules}
    for s in SEEDS:
        old, locked = (np.empty(n), np.empty(n))
        h = np.empty((n, 3))
        for f in range(5):
            k = f's{s}_f{f}_'
            tr, va = (q[k + 'tr'], q[k + 'va'])
            with np.load(cache / f's{s}_f{f}.npz') as a:
                assert str(a['parent_sha']) == digest and np.array_equal(ids[tr], a['train_ids']) and np.array_equal(ids[va], a['validation_ids'])
                z, v = (a['oof_train'], a['valid'])
            h[va], _ = derive(tr, va, z, v, y, fam, shape)
            old[va] = v
            locked[va] = q[k + 'zv']
        for name, w in rules.items():
            new = old + h @ w
            rec = {k: wl.score(y, val, wt, wn, miss) for k, val in (('locked', locked), ('previous', old), ('new', new))}
            scores[name][s] = rec
            arr[name][f's{s}'] = new
            auditlog[name][s] = audit(y, miss, old, new)
            print(name, s, [round(rec['new'][k], 8) for k in ('auc', 'target_vs_neg', 'remaining')], flush=True)
    for name, w in rules.items():
        metrics = scores[name]
        tag = 'local_rank_calibration_' + name
        passed = all((r['new']['auc'] > r['previous']['auc'] and all((r['new'][k] > r['locked'][k] for k in ('target_vs_neg', 'remaining'))) for r in metrics.values()))
        rec = dict(accepted=bool(passed), scores=metrics, audits=auditlog[name])
        (OUT / (tag + '.json')).write_text(json.dumps(rec, indent=2))
        np.savez_compressed(OUT / (tag + '_oof.npz'), ids=ids.to_numpy(dtype=str), **arr[name])
        if not passed:
            continue
        with np.load(cache / 'full.npz') as a:
            assert np.array_equal(tids, a['test_ids']) and str(a['parent_sha']) == digest
            z, v = (a['oof_train'], a['test'])
        h, curves = derive(np.arange(n), n + np.arange(len(tids)), z, v, y, fam, shape)
        dst = ROOT / f'new weights/wn_{tag}_candidate'
        dst.mkdir(exist_ok=True)
        recipe, csv = (dst / 'weights.json', dst / 'submission.csv')
        sub = wl.ef.write_submission(expit(v + h @ w), tids, csv)
        deps = [parent / 'weights.json', Path(__file__).resolve(), *paths]
        recipe.write_text(json.dumps(dict(weights=w.tolist(), curves=curves, grid=GRID.tolist(), hashes={str(p.relative_to(ROOT)): wl.ef.sha(p) for p in dict.fromkeys(deps)}), indent=2))
        np.savez_compressed(dst / 'replay_inputs.npz', ids=tids.to_numpy(dtype=str), order=tids.get_indexer(sub.id), parent_logits=v, features=h)
        (dst / 'validation.json').write_text(json.dumps(dict(**rec, recipe_sha256=wl.ef.sha(recipe), submission_sha256=wl.ef.sha(csv)), indent=2))
        shutil.copyfile(parent / 'replay.py', dst / 'replay.py')
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
