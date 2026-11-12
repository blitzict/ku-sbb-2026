# CPU port; original experiments/weight_space/white_variance_ramp.py, SHA256 58e66be6b01ab5dfc8bf4218dae120b31b750fa9032cb810fec19ae6113f975f
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Separate gradual changes in noise amplitude from a jump at the marked boundary."""
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
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from partition_evidence import OUT

def evidence(y, j):
    n = len(y)
    idx = np.arange(n)
    t = (idx - j) / n
    q = (idx >= j).astype(float)
    local = np.maximum(t, 0)
    left, right = (y[:j], y[j:])
    v = np.r_[left - left.mean(), right - right.mean()]
    var = max(v @ v / n, 1e-06)
    infl = 1.0
    for lag in range(1, 8):
        cov = (v[lag:j] @ v[:j - lag] + v[j + lag:] @ v[j:-lag]) / (n - 2 * lag)
        infl += 2 * (1 - lag / 8) * cov / var
    y = (y - y.mean()) / np.sqrt(var)
    infl = np.clip(infl, 0.5, 9.0)
    X = np.column_stack([t, t * t, q, local, q * (1 - np.exp(-np.maximum(idx - j, 0) / 35)), q * np.exp(-np.maximum(idx - j, 0) / 35)])
    X -= X.mean(0)
    gram = X.T @ X / infl
    rhs = X.T @ y / infl

    def bf(columns, scales):
        k = np.asarray(columns)
        s = np.asarray(scales)
        m = np.eye(len(k)) + gram[np.ix_(k, k)] * s[:, None] * s[None, :]
        b = rhs[k] * s
        return 0.5 * (b @ np.linalg.solve(m, b) - np.linalg.slogdet(m)[1])
    linear = bf([0], [0.75])
    quad = bf([0, 1], [0.65, 0.55])
    base = logsumexp([np.log(0.72), np.log(0.19) + linear, np.log(0.09) + quad])
    scores = []
    for cols, scales in (([2], [0.42]), ([0, 2], [0.7, 0.42]), ([0, 1, 2], [0.6, 0.5, 0.42]), ([0, 3], [0.65, 0.88]), ([0, 2, 3], [0.65, 0.36, 0.72]), ([0, 4], [0.65, 0.45]), ([0, 5], [0.65, 0.45])):
        scores.append(bf(cols, scales))

    def slope(x):
        t = np.linspace(-0.5, 0.5, len(x))
        return (x @ t / (t @ t), np.sqrt(infl / (t @ t)))

    def replicate(x):
        a, b = np.array_split(x, 2)
        ua, sa = slope(a)
        ub, sb = slope(b)
        estimate = ua / (1 + (sa / 0.36) ** 2)
        s2 = sb ** 2 + 1 / (sa ** (-2) + 0.36 ** (-2))
        return 0.5 * (ub ** 2 / sb ** 2 - (ub - estimate) ** 2 / s2 - np.log(s2 / sb ** 2))
    pre, post = (slope(y[:j])[0], slope(y[j:])[0])
    step = logsumexp(np.array(scores[:3]) + np.log([0.72, 0.19, 0.09])) - base
    return [base, linear, quad, step, scores[3] - base, scores[4] - base, scores[5] - base, scores[6] - base, scores[1] - scores[0], pre, post, replicate(y[:j]), replicate(y[j:])]

def extract(part):
    dst = OUT / f'variance_ramp_{part}.parquet'
    if dst.exists():
        return
    with np.load(OUT / f'hetero_path_{part}.npz') as a:
        ids, u, n, b, h = (a[k] for k in ('ids', 'u', 'n', 'b', 'common'))
    rows = []
    for row, scale, end, j in zip(u, h, n, b):
        e = row[:end].astype(float)
        e -= np.r_[np.full(j, np.median(e[:j])), np.full(end - j, np.median(e[j:]))]
        s = max(np.median(abs(e)), 0.15 * e.std(), 0.01)
        mag = abs(e) / s
        stdmag = mag / scale[:end].clip(0.01)
        features = []
        for x in (np.log(0.22 + mag), np.minimum(mag, 4.0), np.log(0.22 + stdmag)):
            features.extend(evidence(x, j))
        rows.append(features)
    cols = [f'{g}_{k}' for g in ('logamp', 'amplitude', 'adjusted_amp') for k in ('envelope', 'linear', 'quad', 'step', 'kink', 'step_kink', 'ramp', 'relax', 'slope_adjustment', 'pre_slope', 'post_slope', 'replicate_pre', 'replicate_post')]
    tab = pd.DataFrame(np.asarray(rows, np.float32).clip(-80, 130), index=pd.Index(ids, name='id'), columns=cols)
    assert np.isfinite(tab).all().all()
    tab.to_parquet(dst)
    print('variance envelope', part, tab.shape, flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        for part in ('train', 'test'):
            extract(part)
