# CPU port; original experiments/weight_space/white_seasonal_memory.py, SHA256 bf15704417b51a7841b1d0569427da3b061605508216eecfa354eaa29fe0ad35
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Test recurrence at integer lags even when the ordinary spectrum has no single line."""
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
from scipy.special import ndtri, logsumexp
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
from partition_evidence import OUT
LAGS = np.arange(3, 65)
RHO = np.array([0.0, -0.9, -0.68, -0.44, -0.25, -0.13, -0.055, 0.055, 0.13, 0.25, 0.44, 0.68, 0.9])
MASS = np.r_[0.62, np.array([0.02, 0.035, 0.065, 0.12, 0.15, 0.11, 0.11, 0.15, 0.12, 0.065, 0.035, 0.02]) * 0.38]
MASS /= MASS.sum()
LP = np.log(MASS)
INV = 1 / (1 - RHO * RHO)
JAC = -0.5 * np.log(1 - RHO * RHO)

def norm(v):
    return ndtri((rankdata(v) - 0.5) / len(v))

def streams(raw, res):
    u = norm(res)
    amp = norm(abs(res - np.median(res)))
    return np.stack([norm(raw), u, amp, norm(u * np.r_[u[0], u[:-1]])])

def summary(x, n):
    d = np.fft.rfft(x, n=2 * x.shape[2], axis=2)
    xy = np.fft.irfft(abs(d) ** 2, n=2 * x.shape[2], axis=2)[:, :, LAGS]
    ss = np.pad(np.cumsum(x * x, 2), ((0, 0), (0, 0), (1, 0)))
    sel = lambda i: np.take_along_axis(ss, np.broadcast_to(i[:, None, :], (len(n), x.shape[1], i.shape[1])), axis=2)
    qq = sel(n[:, None]) + sel(n[:, None] - LAGS) - ss[:, :, LAGS]
    pairs = (n[:, None] - LAGS)[:, None, :, None]
    return xy[:, :, :, None] * (RHO * INV) - 0.5 * qq[:, :, :, None] * (RHO * RHO * INV) + pairs * JAC

def block(left, right, n, m):
    a = summary(left, n)
    b = summary(right, m)
    bridge = []
    rot = []
    for j in LAGS:
        idx = n[:, None] - j + np.arange(j)
        l = np.take_along_axis(left, np.broadcast_to(idx[:, None, :], (len(n), left.shape[1], j)), axis=2)
        r = right[:, :, :j]
        xy = np.fft.irfft(np.fft.rfft(l, axis=2).conj() * np.fft.rfft(r, axis=2), n=j, axis=2)
        assert np.max(abs(xy[:, :, 0] - (l * r).sum(2))) < 1e-05
        ss = (l * l + r * r).sum(2)
        value = xy[:, :, :, None] * (RHO * INV) - 0.5 * ss[:, :, None, None] * (RHO * RHO * INV) + j * JAC
        bridge.append(value[:, :, 0])
        rot.append(logsumexp(value[:, :, 1:], 2) - np.log(j - 1))
    cross = np.stack(bridge, 2)
    rotation = np.stack(rot, 2)
    h0 = a + b + cross
    out = {}
    for lo, hi in ((3, 11), (12, 32), (33, 64), (3, 64)):
        mask = (LAGS >= lo) & (LAGS <= hi)
        T = mask.sum()
        x, y, c, r, v = (q[:, :, mask, :] for q in (a, b, cross, rotation, h0))
        integrate = lambda q: logsumexp(q + LP, (2, 3)) - np.log(T)
        z = integrate(v)
        pre = logsumexp(x + LP, 3)
        after = logsumexp(y + LP, 3)
        joint = logsumexp(x + LP, 3) + logsumexp(y + c + LP, 3)
        nullweights = np.exp(v + LP - z[:, :, None, None] - np.log(T))
        rec = {'null': z, 'reset': integrate(x + y) - z, 'rotate': integrate(x + y + r) - z, 'memory': logsumexp(joint, 2) - np.log(T) - z, 'new_period': integrate(x) + integrate(y) - z, 'start': integrate(y + c) - z, 'stop': integrate(x) - z, 'strong': nullweights[:, :, :, abs(RHO) >= 0.4].sum((2, 3))}
        if lo == 3 and hi == 64:
            rec['pre'] = integrate(x)
            rec['post'] = integrate(y)
            rec['replicate_pre'] = np.take_along_axis(after, np.argmax(pre, 2)[:, :, None], 2)[:, :, 0]
            rec['replicate_post'] = np.take_along_axis(pre, np.argmax(after, 2)[:, :, None], 2)[:, :, 0]
        for key, h in rec.items():
            for i, name in enumerate(('level', 'innovation', 'amplitude', 'short_memory')):
                out[f'{name}_{lo}_{hi}_{key}'] = h[:, i]
    return out

def extract(part):
    path = OUT / f'seasonal_memory_{part}.parquet'
    if path.exists():
        return
    with np.load(OUT / f'pulse_streams_{part}.npz') as a:
        ids, x, n, b = (a[k] for k in ('ids', 'separate', 'n', 'b'))
    with np.load(OUT / f'state_streams_{part}.npz') as a:
        assert np.array_equal(ids, a['ids'])
        e = a['separate']
    rows = []
    for i in range(0, len(x), 42):
        sl = slice(i, i + 42)
        n1 = b[sl]
        n2 = n[sl] - n1
        assert min(n1.min(), n2.min()) > LAGS.max()
        left = np.zeros((len(n1), 4, n1.max()))
        right = np.zeros((len(n1), 4, n2.max()))
        for k, (a, c, j, m) in enumerate(zip(x[sl], e[sl], n1, n2)):
            left[k, :, :j] = streams(a[:j], c[:j])
            right[k, :, :m] = streams(a[j:j + m], c[j:j + m])
        rec = block(left, right, n1, n2)
        rows.append(pd.DataFrame(rec, index=ids[sl]))
    f = pd.concat(rows).clip(-65, 105).astype(np.float32)
    f.index.name = 'id'
    assert np.isfinite(f).all().all()
    f.to_parquet(path)
    print(part, 'seasonal dependence and phase reset', f.shape, flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        for part in ('train', 'test'):
            extract(part)
