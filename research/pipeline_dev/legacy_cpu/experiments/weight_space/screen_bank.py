# CPU port; original experiments/weight_space/screen_bank.py, SHA256 0ac19c61b1fde346ab09ac647e7649ad2111a492fb8cac806692a35174ffb2ea
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Select residual measurements on meta-training rows and write a sparse readout."""
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
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.special import expit
from threadpoolctl import threadpool_limits
import expert_fusion as ef
import residual_trial as rt
BLOCKS = ['clean/features', *['tune_out/' + name for name in ['grid', 'grid2', 'dd', 'gh', 'lr5', 'lrz6', 'lm', 'ps', 'kn', 'sn', 'zp', 'old_extra', 'wh', 'ssa', 'sh', 'ix', 'bf', 'cl']]]
MODES = {'sparse': (12, False), 'wide': (40, False), 'conditional': (24, True)}

def tables(ids, tids):
    names = []
    sides = [[], []]
    for stem in BLOCKS:
        for j, (p, index) in enumerate([('train', ids), ('test', tids)]):
            tab = pd.read_parquet(ef.ROOT / f'{stem}_{p}.parquet')
            assert tab.index.is_unique and set(tab.index) == set(index)
            tab = tab.loc[index]
            cols = list(tab.columns)
            assert not set((c.lower() for c in cols)).intersection({'id', 'label', 'target', 'fold', 'train', 'prediction'})
            if j == 0:
                train_cols = cols
                names.extend([stem + '::' + k for k in cols])
            else:
                assert set(train_cols) == set(cols)
                tab = tab[train_cols]
            sides[j].append(tab.to_numpy(dtype=np.float32))
    assert len(names) == len(set(names))
    return (names, np.vstack([np.column_stack(s) for s in sides]))

def transform(H, center, spread):
    d = H - center
    with np.errstate(invalid='ignore', divide='ignore'):
        result = d / (np.abs(d) + spread)
    return np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

def adjoint(z, y):
    grid = np.linspace(-6, 12, 257)
    centers = (grid[:-1] + grid[1:]) / 2
    density = []
    for label in [0, 1]:
        count = np.histogram(np.clip(z[y == label], grid[0], grid[-1]), bins=grid)[0]
        rho = gaussian_filter1d(count.astype(float), 4.0) / count.sum()
        density.append(np.interp(z, centers, rho))
    g = np.where(y == 1, density[0] / (y == 1).mean(), -density[1] / (y == 0).mean())
    for b in np.array_split(np.argsort(z), 32):
        g[b] -= g[b].mean()
    return g.astype(np.float32)

def screen(H, idx, z, y):
    rng = np.random.default_rng(639)
    copies = np.array_split(rng.permutation(len(idx)), 2)
    G = np.zeros((len(idx), 2), np.float32)
    for j, ix in enumerate(copies):
        G[ix, j] = adjoint(z[ix], y[ix])
    strength, centers, spreads = ([], [], [])
    for start in range(0, H.shape[1], 96):
        h = H[idx, start:start + 96].copy()
        h[~np.isfinite(h)] = np.nan
        q = np.nanquantile(h, [0.1, 0.5, 0.9], axis=0)
        center = np.nan_to_num(q[1]).astype(np.float32)
        spread = np.maximum(np.nan_to_num(q[2] - q[0]) * 0.4, 1e-06).astype(np.float32)
        s = transform(h, center, spread)
        centers.extend(center)
        spreads.extend(spread)
        vals = []
        for basis in (s, s * s):
            sd = np.maximum(0.08, basis.std(axis=0))
            dots = basis.T @ G / sd[:, None]
            vals.append(np.where(dots[:, 0] * dots[:, 1] > 0, np.min(abs(dots), axis=1), 0))
        strength.extend(np.maximum(*vals))
    center, spread = (np.array(centers), np.array(spreads))
    order = np.argsort(-np.array(strength), kind='stable')[:200]
    probe = transform(H[idx[:min(3000, len(idx))]][:, order], center[order], spread[order])
    probe -= probe.mean(0)
    probe /= np.maximum(np.linalg.norm(probe, axis=0), 1e-06)
    corr = abs(probe.T @ probe)
    take = []
    for i in range(len(order)):
        if not take or max(corr[i, take]) < 0.85:
            take.append(i)
        if len(take) >= 40:
            break
    return (order[take], center, spread)

def features(H, ix, cols, center, spread, z, gates):
    x = transform(H[np.ix_(ix, cols)], center, spread).astype(float)
    banks = [x, x * x]
    if gates:
        banks.append(x * np.tanh(z[:, None] / 2))
    return np.column_stack(banks)

def main():
    data = ef.read_data()
    _, ids, tids, y, wt, *_ = data
    names, H = tables(ids, tids)
    print('Read', len(names), 'measurement columns', flush=True)
    folds, full_z = rt.baseline(data)
    scores, saved, picks = ({}, {}, {})
    for split, parts in folds.items():
        pred = {k: np.empty(len(y)) for k in ['anchor', *MODES]}
        picks[split] = []
        for fi, (tr, va, zr, zv) in enumerate(parts):
            sel, center, spread = screen(H, tr, zr, y[tr])
            pred['anchor'][va] = zv
            picks[split].append([names[i] for i in sel])
            for mode, (n, gates) in MODES.items():
                cols = sel[:n]
                X = features(H, tr, cols, center[cols], spread[cols], zr, gates)
                T = features(H, va, cols, center[cols], spread[cols], zv, gates)
                th = ef.fit_head(X, zr, y[tr])
                pred[mode][va] = zv + T @ th[:-1] + th[-1]
            print(split, fi, picks[split][-1][:3], flush=True)
        scores[split] = {k: ef.score(y, v, wt) for k, v in pred.items()}
        saved.update({f's{split}_{k}': v for k, v in pred.items()})
        print(split, scores[split], flush=True)
    eligible = [m for m in MODES if all((row[m][k] - row['anchor'][k] >= 0.0003 for row in scores.values() for k in ('auc', 'weighted_auc')))]
    chosen = max(eligible, key=lambda m: np.mean([row[m]['auc'] for row in scores.values()])) if eligible else None
    report = {'scores': scores, 'selected': chosen, 'fold_selections': picks}
    (ef.OUT / 'screen_bank_results.json').write_text(json.dumps(report, indent=2))
    np.savez(ef.OUT / 'screen_bank_oof.npz', ids=ids.to_numpy(dtype=str), **saved)
    if chosen:
        dest = ef.PACKAGE / 'screen_bank_candidate'
        dest.mkdir(exist_ok=True)
        tr = np.arange(len(y))
        sel, center, spread = screen(H, tr, full_z[tr], y)
        n, gates = MODES[chosen]
        cols = sel[:n]
        X = features(H, np.arange(len(H)), cols, center[cols], spread[cols], full_z, gates)
        th = ef.fit_head(X[tr], full_z[tr], y)
        path = dest / 'weights.json'
        recipe = {'mode': chosen, 'features': [names[i] for i in cols], 'center': center[cols].tolist(), 'spread': spread[cols].tolist(), 'coefficients': th.tolist(), 'gates': gates, 'scores': scores}
        path.write_text(json.dumps(recipe, indent=2))
        ef.write_submission(expit(full_z[len(y):] + X[len(y):] @ th[:-1] + th[-1]), tids, dest / 'submission.csv')
        a = json.loads(path.read_text())
        fs = np.array([names.index(k) for k in a['features']])
        F = features(H, np.arange(len(y), len(H)), fs, np.array(a['center'], np.float32), np.array(a['spread'], np.float32), full_z[len(y):], a['gates'])
        w = np.array(a['coefficients'])
        ef.write_submission(expit(full_z[len(y):] + F @ w[:-1] + w[-1]), tids, dest / 'submission_replay.csv')
        assert ef.sha(dest / 'submission.csv') == ef.sha(dest / 'submission_replay.csv')
    print('Selected', chosen, flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
