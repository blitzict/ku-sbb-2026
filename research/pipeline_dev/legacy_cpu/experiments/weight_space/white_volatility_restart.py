# CPU port; original experiments/weight_space/white_volatility_restart.py, SHA256 0e2de1e153f9a1611ca1423e73189db6be811c6dcda7ba0c16c1ba35e3f61f6d
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""At a marked boundary, test continuing, clearing, or redrawing the current volatility state."""
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
from scipy.signal import lfilter
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from partition_evidence import OUT
BETA = np.array([0.0, 0.35, 0.68, 0.86, 0.95])
ALPHA = np.array([0.0, 0.26, 0.55, 0.82])
HORIZONS = (3, 10, 24)
NAMES = [f'{kind}_{k}' for k in HORIZONS for kind in ('clear', 'redraw', 'best')] + ['memory', 'boundary_state']

def spread(x):
    return max(1.4826 * np.median(abs(x - np.median(x))), 0.2 * x.std(), 0.01)

def normalizer(e):
    return lambda v: -0.5 * np.log(v) - 4.5 * np.log1p(e ** 2 / (6 * v))

def state_evidence(before, after):
    before = before[-112:]
    after = after[:96]
    scale = spread(before)
    z = (before - np.median(before)) / scale
    calibration = after[24:]
    offset = np.median(calibration)
    post = (after[:24] - offset) / scale
    change = np.clip(spread(calibration) / scale, 0.3, 3.5)
    sq = np.minimum(z * z, 36)
    past = np.r_[1.0, sq[:-1]]
    pre = np.stack([lfilter([1 - p], [1.0, -p], past, zi=[p])[0] for p in BETA])
    state = (1 - BETA) * sq[-1] + BETA * pre[:, -1]
    clouds = np.quantile(pre[:, 7:], [0.1, 0.3, 0.5, 0.7, 0.9], axis=1).T
    v = 1 - ALPHA[None, :, None] + ALPHA[None, :, None] * pre[:, None, :]
    ll = normalizer(z)(v)[:, :, 7:].sum(2)
    prior = np.broadcast_to(np.array([0.82, 0.08, 0.065, 0.035]), ll.shape) / len(BETA)
    lp = ll + np.log(prior)
    lp -= logsumexp(lp)
    memory = 1.0 - np.exp(lp[:, 0]).sum()
    sig = np.array([1.0, np.sqrt(change), change]) ** 2
    beta = BETA[:, None, None, None, None]
    alpha = ALPHA[None, :, None, None, None]
    equilibrium = sig[None, None, :, None, None]
    initial = np.concatenate([state[:, None], np.ones((len(BETA), 1)), clouds], axis=1)
    start = np.broadcast_to(initial[:, None, None, :, None], (len(BETA), 1, len(sig), 7, 1)).copy()
    start[:, :, :, 1:] *= equilibrium
    start = equilibrium + (beta + (1 - beta) * alpha) ** 2 * (start - equilibrium)
    obs = np.r_[0.0, np.minimum(post[:-1] ** 2, 36)]
    decay = BETA[:, None] ** np.arange(len(post))
    driven = np.stack([lfilter([1 - p], [1.0, -p], obs) for p in BETA])
    h = start * decay[:, None, None, None, :] + driven[:, None, None, None, :]
    variance = ((1 - alpha) * equilibrium + alpha * h).clip(0.025)
    cumulative = normalizer(post)(variance).cumsum(axis=-1)
    base = lp[:, :, None, None] + np.log(np.array([0.5, 0.24, 0.26]))[None, None, :, None]
    result = []
    for k in HORIZONS:
        e = logsumexp(base + cumulative[:, :, :, :, k - 1], axis=(0, 1, 2))
        result.extend([e[1] - e[0], logsumexp(e[2:]) - np.log(5) - e[0], np.max(e[2:]) - e[0]])
    return np.r_[result, memory, np.exp(lp).sum(1) @ np.log(state.clip(0.02))]

def run(part):
    dst = OUT / f'volatility_restart_{part}.parquet'
    if dst.exists():
        return
    with np.load(OUT / f'pulse_streams_{part}.npz') as a:
        ids, raw, n, b = (a[k] for k in ('ids', 'separate', 'n', 'b'))
    phi = pd.read_parquet(OUT / f'state_{part}.parquet').loc[ids, ['ar1', 'ar2']].to_numpy()
    records = []
    for v, end, j, ar in zip(raw, n, b, phi):
        x = v[:end].astype(float)

        def innov(seg):
            return seg[2:] - ar[0] * seg[1:-1] - ar[1] * seg[:-2]
        left, right = (innov(x[:j]), innov(x[j:]))
        val = state_evidence(left, right)
        controls = []
        for seg in (left, right):
            mid = len(seg) // 2
            if mid >= 36 and len(seg) - mid >= 38:
                controls.append(state_evidence(seg[:mid], seg[mid + 2:]))
        ref = np.mean(controls, axis=0) if controls else val
        records.append(np.r_[val, val - ref])
    cols = [p + s for p in ('marked_', 'over_interior_') for s in NAMES]
    df = pd.DataFrame(records, index=pd.Index(ids, name='id'), columns=cols).clip(-90, 120)
    assert np.isfinite(df).all().all()
    df.astype(np.float32).to_parquet(dst)
    print(part, df.shape, flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        for part in ('train', 'test'):
            run(part)
