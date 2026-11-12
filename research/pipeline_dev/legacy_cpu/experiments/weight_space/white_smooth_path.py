# CPU port; original experiments/weight_space/white_smooth_path.py, SHA256 28d7368bc7e2406f1a5d6691a8ad3b9e12a7d4901e09802b8f512e36fa8fa58d
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Compare a continuing smooth trajectory with a jump, kink or restarted trajectory."""
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

def evidence(gram, cross, yy, n, scales):
    sd = np.asarray(scales)
    c = sd * cross
    matrix = np.eye(len(sd)) + sd[:, None] * gram * sd
    sign, det = np.linalg.slogdet(matrix)
    assert sign > 0
    explained = c @ np.linalg.solve(matrix, c)
    return -0.5 * det - (0.5 * n + 3) * np.log1p(-min(explained, yy) / (yy + 4.0))

def one(row, cut, ar):
    n = len(row)
    t = np.arange(n) / (n - 1)
    where = t[cut]
    side = (np.arange(n) >= cut).astype(float)
    raw = row - np.median(row)
    white = lambda a: a[2:] - ar[0] * a[1:-1] - ar[1] * a[:-2]
    y = white(raw)
    linear = np.column_stack((np.ones(n), t, side, side * (t - where)))
    baseline = white(linear)
    resid = y - baseline @ np.linalg.lstsq(baseline, y, rcond=None)[0]
    scale = max(np.median(abs(resid - np.median(resid))) * 1.4826, 0.18 * resid.std(), 0.001)
    y = y / scale
    weighted = 1 / np.sqrt(1 + (resid / (3.0 * scale)) ** 2)
    obs = y * weighted

    def loglike(A, scales):
        X = white(A) * weighted[:, None]
        return evidence(X.T @ X, X.T @ obs, obs @ obs, len(obs), scales)
    simple = loglike(linear[:, :2], (4.0, 1.8))
    nulls = []
    alternatives = []
    for width in (0.14, 0.32, 0.7):
        knots = np.linspace(-0.15, 1.15, 10)

        def curve(time):
            k = np.exp(-0.5 * ((time[:, None] - knots) / width) ** 2)
            return k / np.linalg.norm(k, axis=1, keepdims=True)
        basis = curve(t)
        shared = np.column_stack((linear[:, :2], basis))
        sd = [4.0, 1.8] + [0.78] * len(knots)
        z0 = loglike(shared, sd)
        nulls.append(z0)
        at = curve(np.array([where]))[0]
        split = side[:, None] * (basis - at)
        alts = [loglike(np.column_stack((shared, side)), [*sd, 0.85]), loglike(np.column_stack((shared, side * (t - where))), [*sd, 1.65]), loglike(np.column_stack((shared, split)), [*sd, *[0.9] * len(knots)]), loglike(np.column_stack((shared, split, side)), [*sd, *[0.9] * len(knots), 0.85])]
        alternatives.append(alts)
    z = logsumexp(nulls) - np.log(len(nulls))
    gains = logsumexp(alternatives, axis=0) - np.log(len(nulls)) - z
    return np.r_[z - simple, gains, gains[3] - gains[2], gains[3] - gains[0]]

def extract(part):
    path = OUT / f'smooth_path_{part}.parquet'
    if path.exists():
        return
    with np.load(OUT / f'pulse_streams_{part}.npz') as a:
        ids, x, n, b = (a[k] for k in ('ids', 'common', 'n', 'b'))
    state = pd.read_parquet(OUT / f'state_{part}.parquet').loc[ids]
    ar = state[['ar1', 'ar2']].to_numpy()
    curve = pd.read_parquet(OUT / f'curve_{part}.parquet').loc[ids]
    walk = pd.read_parquet(OUT / f'innov_v2_{part}.parquet').loc[ids, 'pre_walk'].to_numpy()
    gate = np.maximum(curve.gate.to_numpy(), walk)
    A = np.zeros((len(x), 7))
    for i in np.flatnonzero(gate > 0.12):
        A[i] = gate[i] * one(x[i, :n[i]].astype(float), b[i], ar[i]).clip(-45, 90)
    cols = ['smooth', 'step', 'kink', 'new_curve', 'new_curve_and_step', 'jump_on_new_curve', 'curvature_on_step']
    tab = pd.DataFrame(A, index=pd.Index(ids, name='id'), columns=cols)
    assert np.isfinite(tab).all().all()
    tab.to_parquet(path)
    print(part, 'scored smooth trajectory', np.count_nonzero(gate > 0.12), flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        for part in ('train', 'test'):
            extract(part)
