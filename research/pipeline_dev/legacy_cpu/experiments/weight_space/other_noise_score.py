# CPU port; original experiments/weight_space/other_noise_score.py, SHA256 5ed2e256e569f5ccad354108e804a7d1315e0c352fa959b344e5028d2ea9661f
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Adapt the location, scale and serial-change tests to each record's noise density.

The finite-basis score-matching identity follows Hyvarinen, JMLR 2005.
Within-record learning uses disjoint blocks and never reads a break label.
"""
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
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from partition_evidence import OUT
CENTERS = np.r_[np.linspace(-2.8, 2.8, 11), [-2.0, -0.8, 0, 0.8, 2.0]]
WIDTH = np.r_[np.full(11, 0.62), np.full(5, 0.34)]
TERMS = ('level', 'scale', 'skew', 'mode_spacing', 'tail_scale', 'lag1', 'lag2', 'lag_scale')

def score(z, train):
    d = (z[:, None] - CENTERS) / WIDTH
    bump = np.exp(-0.5 * d ** 2)
    b = np.column_stack([np.tanh(z / 2), bump])
    deriv = np.column_stack([0.5 * (1 - np.tanh(z / 2) ** 2), -d * bump / WIDTH])
    ref = 6 * z / (4 + z * z)
    B = b[train]
    G = B.T @ B / len(B)
    target = deriv[train].mean(0) - B.T @ ref[train] / len(B)
    regularizer = np.diag(np.r_[0.06, np.full(11, 0.03), np.full(5, 0.085)])
    w = np.linalg.solve(G + regularizer, target)
    return (ref, (ref + b @ w).clip(-7, 7))

def gradients(z, s, lag):
    m = np.tanh(1.65 * z)
    t = np.tanh(z / 2)
    a = np.tanh(z) ** 2
    return np.column_stack([s, z * s - 1, s * a - 2 * np.tanh(z) * (1 - a), s * m - 1.65 * (1 - m * m), z * t * t * s - t * t - z * t * (1 - t * t), s * lag[:, 0], s * lag[:, 1], (z * s - 1) * np.tanh(abs(lag[:, 0]) - 0.7)]).clip(-18, 18)

def summarize(x, b, cut):
    values = []
    for blocks in (np.arange(0, 12, 2), np.arange(1, 12, 2)):
        pre = b < 6
        mask = np.isin(b, blocks)
        a, c = (x[mask & pre], x[mask & ~pre])
        se = np.sqrt(a.var(0) / len(a) + c.var(0) / len(c) + 1e-05)
        values.append((c.mean(0) - a.mean(0)) / se)
    v = np.stack(values)
    pooled = v.sum(0) / np.sqrt(2)
    return np.r_[np.arcsinh(pooled), np.arcsinh(v.prod(0)), np.arcsinh(np.mean(np.sort(pooled ** 2)[-2:]))]

def one(r, history, cut):
    n = len(r)
    t = np.arange(n)
    b = np.where(t < cut, t * 6 // cut, 6 + (t - cut) * 6 // (n - cut))
    time = t[2:][b[2:] == b[:-2]]
    med = np.median(r)
    sd = max(1.4826 * np.median(abs(r - med)), 0.15 * r.std(), 0.0001)
    z = ((r[time] - med) / sd).clip(-14, 14)
    center = np.median(history)
    scale = max(1.4826 * np.median(abs(history - center)), 0.15 * history.std(), 0.0001)
    prior = ((history - center) / scale).clip(-5, 5)
    lag = np.column_stack([prior[time - 1], prior[time - 2]])
    b = b[time]
    g = [np.empty((len(z), len(TERMS))) for _ in range(2)]
    for parity in (0, 1):
        test = b % 2 == parity
        for v, s in zip(g, score(z, ~test)):
            v[test] = gradients(z[test], s[test], lag[test])
    fixed, adaptive = [summarize(a, b, cut) for a in g]
    return np.r_[adaptive, adaptive - fixed]

def build(part):
    arrays = []
    cols = []
    for name in ('pulse', 'state'):
        with np.load(OUT / f'{name}_streams_{part}.npz') as a:
            ids, r, n, b = [a[k] for k in ('ids', 'separate', 'n', 'b')]
        with np.load(OUT / f'pulse_streams_{part}.npz') as a:
            assert np.array_equal(ids, a['ids']) and np.array_equal(b, a['b'])
            history = a['common']
        arr = np.array([one(row[:end].astype(float), h[:end].astype(float), cut) for row, h, end, cut in zip(r, history, n, b)])
        assert np.isfinite(arr).all()
        arrays.append(arr)
        cols += [f'{name}_{k}_{t}' for k in ('adaptive', 'extra') for t in [*['signed_' + s for s in TERMS], *['repeat_' + s for s in TERMS], 'strongest']]
    pd.DataFrame(np.column_stack(arrays), index=ids, columns=cols).to_parquet(OUT / f'noise_score_{part}.parquet')
    print(part, len(ids), len(cols), flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        for part in ('train', 'test'):
            build(part)
