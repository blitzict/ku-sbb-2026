# CPU port; original experiments/weight_space/other_reversible_pairs.py, SHA256 53d6cc11ac0d724ebaa8bde9ec01ae2eef767b26801fcc748752f3b6508c309c
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Learn from nearby positive-negative pairs whose ordering other sources contest."""
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
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits
from other_miss_witnesses import make
from other_response_gate import run
from partition_evidence import ROOT, OUT
from white_local_rank_weights import centered

def pair_weights(x, z, y, panel):
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    neg = neg[np.argsort(z[neg])]
    left = np.searchsorted(z[neg], z[pos] - 1.8)
    right = np.searchsorted(z[neg], z[pos] + 1.8)
    active = right > left
    pos, left, right = (pos[active], left[active], right[active])
    rng = np.random.default_rng(37231)
    points = (np.arange(48) + rng.uniform(size=(len(pos), 1))) / 48
    i = np.repeat(pos, 48)
    j = neg[(left[:, None] + ((right - left)[:, None] * points).astype(int)).ravel()]
    gap = z[i] - z[j]
    dif = panel[i] - panel[j]
    opposition = np.mean(expit(-np.sign(gap)[:, None] * dif / 0.42), axis=1)
    mass = np.repeat(right - left, 48) * np.exp(-0.5 * (gap / 0.83) ** 2) * (0.3 + 1.4 * opposition)
    mass *= 0.68 * len(y) / mass.sum()

    def loss(w):
        score = z + x @ w
        d = (score[j] - score[i]) / 0.35
        v = mass * expit(d)
        grad = 0.32 * (expit(score) - y) + np.bincount(j, weights=v, minlength=len(y)) - np.bincount(i, weights=v, minlength=len(y))
        return (0.32 * np.sum(np.logaddexp(0, score) - y * score) + 0.35 * mass @ np.logaddexp(0, d) + 75 * (w @ w), x.T @ grad + 150 * w)
    res = minimize(loss, np.zeros(x.shape[1]), jac=True, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-10})
    assert res.success, res.message
    return res.x

def factory(ids, tids, y):
    panels, paths = make(ids, tids)
    E = np.column_stack([b.mean(1) for b in panels])

    def table(stem, cols):
        pieces = []
        for part, ix in (('train', ids), ('test', tids)):
            path = OUT / f'{stem}_{part}.parquet'
            paths.append(path)
            pieces.append(pd.read_parquet(path, columns=cols).loc[ix].to_numpy())
        return np.vstack(pieces)
    memory = table('strict_unit_memory', ['grid_pre_only', 'grid_post_only', 'grid_release', 'physical_pre_only', 'physical_post_only'])
    history = table('innov_v2', ['pre_white', 'pre_walk'])
    cycle = table('periodic', ['pre_fraction'])
    vol = table('volatility_law', ['pre_cluster'])
    phase = table('phase_law', ['phase_bf', 'scale_bf'])
    cov = np.column_stack([history, cycle, vol]) - 0.4

    def fit(tr, va, z, v, unused_train, unused_valid):
        ix = np.r_[tr, va]
        score = np.r_[z, v]
        n = len(tr)
        lo, med, hi = np.quantile(E[tr], [0.15, 0.5, 0.85], axis=0)
        a, b, c = np.quantile(z, [0.15, 0.5, 0.85])
        level = (E[ix] - med) * (c - a) / (hi - lo).clip(0.1) + b
        delta = np.arcsinh((level - score[:, None]).clip(-10, 10))
        risk = np.tanh((score + 0.4) / 1.4)
        auxiliary = np.arcsinh(np.column_stack([memory[ix], phase[ix]]).clip(-50, 80))
        h = np.column_stack([delta, 0.6 * delta * risk[:, None], 0.55 * (cov[ix, None, :] * delta[:, :, None]).reshape(len(ix), -1), 0.4 * auxiliary, 0.5 * (delta.max(1) - np.median(delta, axis=1))[:, None] * auxiliary])
        train, ref = centered(h[:n], z)
        valid, _ = centered(h[n:], v, ref)
        w = 0.6 * pair_weights(train, z, y[tr], level[:n])
        return (v + valid @ w, valid, w)
    return (fit, [*paths, Path(__file__).resolve()])
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run(factory, 'local_rank_contested_pairs')
