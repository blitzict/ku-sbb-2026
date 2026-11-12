# CPU port; original experiments/weight_space/other_kernel_timing.py, SHA256 52bbccfd7fa5f2b6108fb8fc703bf3066c34ebc1373b77dbc9ea00adb0e906ce
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Does a distributional change peak at the specified time on unseen blocks?

The kernel Fisher scan is adapted from Harchaoui, Bach and Moulines (2008).
Every evaluated cut uses the same pooled transform of the observed series.
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
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from partition_evidence import OUT, ROOT
DIRECTIONS = np.array([[1, 0, 0], [0.57, 0, 0], [1.8, 0, 0], [3.1, 0, 0], [1, 1, 0], [1, -1, 0], [1, 0.35, 0], [0.35, -1, 0], [1, 0, 1], [1, 0, -1], [0.7, 0.7, 0.7], [0.7, -0.7, 0.7]])
SUMMARY = ('given', 'away_mean', 'center_advantage', 'peak_margin', 'peak_distance', 'fit_step', 'residual_shape', 'above_both_neighbors', 'nearby_mass', 'replicated')

def robust(x):
    center = np.median(x)
    return np.arcsinh((x - center) / max(1.4826 * np.median(abs(x - center)), 0.15 * x.std(), 1e-05))

def scan(v, t, cut, end):
    lag = 4
    z = robust(v)
    vec = np.column_stack([z[lag:], z[lag - 1:-1], z[:-lag]])
    tt = t[lag:]
    tau = np.unique(np.r_[np.linspace(0.17, 0.83, 15) * end, cut].astype(int))
    at = int(np.flatnonzero(tau == cut)[0])
    sides = tt[:, None] < tau
    assert np.all(sides.any(0)) and np.all((~sides).any(0))
    block = t * 16 // end
    held = block[lag:] == block[:-lag]
    phase = vec @ DIRECTIONS.T
    x = np.column_stack([np.sin(phase), np.cos(phase), np.tanh(vec[:, 0]) * np.tanh(vec[:, 1]), np.tanh(abs(vec[:, 0]) - 0.75) * np.tanh(abs(vec[:, 1]) - 0.75)])
    cov = np.cov(x[held].T)
    q, b = np.linalg.eigh(cov)
    x = (x - x[held].mean(0)) @ (b / np.sqrt(q + 0.045))
    means = []
    for parity in (0, 1):
        keep = held & (block[lag:] % 2 == parity)
        a = sides[keep].astype(float)
        left = a.sum(0)
        right = len(a) - left
        diff = (1 - a).T @ x[keep] / right[:, None] - a.T @ x[keep] / left[:, None]
        means.append(diff * np.sqrt(left * right / (left + right))[:, None])
    u, w = means
    product = np.einsum('ij,ij->i', u, w) / np.sqrt(x.shape[1])
    given = product[at]
    dist = abs(tau - cut) / end
    far = dist > 0.135
    pos = tau / end
    true = cut / end
    expected = np.minimum(pos / (1 - pos), true / (1 - true)) / np.maximum(pos / (1 - pos), true / (1 - true))
    slope = np.cov(expected, product, ddof=0)[0, 1] / max(expected.var(), 0.001)
    deviation = np.sqrt(np.mean((product - product.mean() - slope * (expected - expected.mean())) ** 2))
    left = np.flatnonzero((tau < cut) & (dist > 0.04))
    right = np.flatnonzero((tau > cut) & (dist > 0.04))
    neighbor = max(product[left[-1]], product[right[0]])
    mass = logsumexp(np.clip(product, -12, 12)[dist < 0.06]) - logsumexp(np.clip(product, -12, 12))
    rep = given / max(0.5, np.sqrt(np.sum(u[at] ** 2) * np.sum(w[at] ** 2)) / np.sqrt(x.shape[1]))
    return [given, product[far].mean(), given - product[far].mean(), given - product[far].max(), dist[product.argmax()], slope, deviation, given - neighbor, mass, rep]

def one(x, cut):
    t = np.arange(len(x))
    centered = x - np.median(x)
    scale = max(np.median(abs(centered)) * 1.4826, 0.18 * x.std(), 0.001)
    clip = centered.clip(-5 * scale, 5 * scale)
    design = np.column_stack([clip[1:-1], clip[:-2]])
    phi = np.linalg.solve(design.T @ design + 0.55 * scale ** 2 * np.eye(2), design.T @ clip[2:])
    err = centered[2:] - phi[0] * centered[1:-1] - phi[1] * centered[:-2]
    return np.r_[scan(x, t, cut, len(t)), scan(np.diff(x), t[1:], cut, len(t)), scan(err, t[2:], cut, len(t))]

def build(part):
    with np.load(ROOT / f'tune_out/seq_cache_{part}.npz', allow_pickle=True) as a:
        ids, x, n, b = (a[k] for k in ('ids', 'x', 'n', 'b'))
    ix = pd.read_parquet(ROOT / f'clean/features_{part}.parquet', columns=[]).index
    assert np.array_equal(ids, ix)
    result = np.array([one(row[:end].astype(float), cut) for row, end, cut in zip(x, n, b)])
    names = [f'{c}_{k}' for c in ('raw', 'increment', 'pooled_innov') for k in SUMMARY]
    assert np.isfinite(result).all()
    pd.DataFrame(result, index=ix, columns=names).to_parquet(OUT / f'kernel_timing_{part}.parquet')
    print(part, result.shape, flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        for part in ('train', 'test'):
            build(part)
