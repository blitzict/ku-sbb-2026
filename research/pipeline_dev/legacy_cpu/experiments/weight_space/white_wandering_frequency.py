# CPU port; original experiments/weight_space/white_wandering_frequency.py, SHA256 7ae4868d55af8a4cda7ff63419eca8e6c7fa623a7e8d18d61f101225bdec6f70
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Track an oscillation allowed to change frequency between observed windows."""
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
from functools import lru_cache
from scipy.special import expit, logsumexp
from threadpoolctl import threadpool_limits
from partition_evidence import OUT
PERIODS = np.geomspace(6.0, 48.0, 24)
v = np.log(PERIODS)
TRANSITION = np.exp(-0.5 * ((v[:, None] - v[None, :]) / 0.18) ** 2)
TRANSITION /= TRANSITION.sum(1, keepdims=True)

@lru_cache(maxsize=12)
def atoms(width):
    t = np.linspace(-1.0, 1.0, width)
    nuisance = np.column_stack([np.ones(width), t])
    project = np.eye(width) - nuisance @ np.linalg.pinv(nuisance)
    phase = 2 * np.pi / PERIODS[:, None] * np.arange(width)
    basis = np.stack([np.sin(phase), np.cos(phase)], axis=2)
    basis = np.linalg.qr(project @ basis)[0]
    return (project, basis.reshape(-1, width, 2))

def evidence(segments, width):
    project, basis = atoms(width)
    matrix = segments @ project
    rms = np.sqrt(np.mean(matrix ** 2, axis=1)).clip(0.001)
    h = np.einsum('bw,pwk->bpk', matrix / rms[:, None], basis)
    power = (h * h).sum(2)
    return logsumexp(np.stack([0.5 * s / (1 + s) * power - np.log1p(s) for s in (0.5, 2.0, 6.0)]), axis=0) - np.log(3)

def track(ll):
    weights = np.full(len(PERIODS), 1 / len(PERIODS))
    e = 0.0
    for row in ll:
        shift = row.max()
        weights *= np.exp(row - shift)
        total = weights.sum()
        weights /= total
        e += shift + np.log(total)
        weights = weights @ TRANSITION
    return e

def one(x, cut, width):
    left = x[cut % width:cut].reshape(-1, width)
    right = x[cut:cut + (len(x) - cut) // width * width].reshape(-1, width)
    e = evidence(np.vstack((left, right)), width)
    j = len(left)
    a, b = (e[:j], e[j:])
    pre, post, full = (track(a), track(b), track(e))
    fixed = logsumexp(e.sum(0)) - np.log(len(PERIODS))
    return [pre, post, pre + post - full, full - fixed, post - full, pre - full, min(pre, post), track(a[::2]) + track(b[::2]), track(a[1::2]) + track(b[1::2])]

def extract(part):
    path = OUT / f'wandering_frequency_{part}.parquet'
    if path.exists():
        return
    with np.load(OUT / f'pulse_streams_{part}.npz') as a:
        ids, x, n, b = (a[k] for k in ('ids', 'common', 'n', 'b'))
    ar = pd.read_parquet(OUT / f'state_{part}.parquet').loc[ids, ['ar1', 'ar2']].to_numpy()
    records = []
    for i, (length, cut) in enumerate(zip(n, b)):
        row = x[i, :length]
        lag = np.column_stack((row[1:-1], row[:-2]))
        residual = row[2:] - lag @ ar[i]
        j = cut - 2
        for segment in (residual[:j], residual[j:]):
            segment -= np.median(segment)
            scale = max(np.median(abs(segment)) * 1.4826, 0.17 * segment.std(), 0.001)
            segment[:] = (segment / scale).clip(-3.0, 3.0)
        stats = []
        for values in (residual, abs(residual)):
            for w in (32, 56):
                stats.extend(one(values, j, w))
        records.append(stats)
    cols = [f'{v}_{w}_{s}' for v in ('noise', 'size') for w in (32, 56) for s in ('pre', 'post', 'restart', 'wander', 'on', 'off', 'both', 'even', 'odd')]
    frame = pd.DataFrame(records, index=pd.Index(ids, name='id'), columns=cols).clip(-60, 125)
    assert np.isfinite(frame).all().all()
    frame.to_parquet(path)
    print(part, 'pre strength above five', frame[['noise_32_pre', 'noise_56_pre']].gt(5).any(axis=1).sum(), flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        for part in ('train', 'test'):
            extract(part)
