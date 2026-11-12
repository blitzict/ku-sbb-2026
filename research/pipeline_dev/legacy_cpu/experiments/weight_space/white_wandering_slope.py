# CPU port; original experiments/weight_space/white_wandering_slope.py, SHA256 717cc437807e6c68811e77c8a631d5755aeb87b61e3f71e72afd73fcc69994f7
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Compare an uninterrupted random walk with a wandering slope against boundary changes."""
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
import torch
from scipy.linalg import toeplitz, solve_triangular
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from partition_evidence import OUT
RADII = (52, 72, 104, 144, 192, 256)

def models():
    rows = []
    for obs, wobs in ((0.0, 0.6), (0.32, 0.22), (0.72, 0.12), (0.95, 0.06)):
        rows.append((obs, 0.0, 1.0, 0.7 * wobs))
        for share, w in ((0.035, 0.46), (0.16, 0.37), (0.46, 0.17)):
            for duration, wd in ((4.0, 0.19), (16.0, 0.35), (64.0, 0.34), (180.0, 0.12)):
                rows.append((obs, share * (1 - obs), duration, 0.3 * wobs * w * wd))
    return np.array(rows)

def chart(n, j):
    t = np.arange(n) - j
    funcs = []
    for center in (0, -j // 2, (n - j) // 2):
        local = t - center
        funcs.extend([(local == 0).astype(float), (local >= 0).astype(float), (local >= 0) * np.exp(-np.maximum(local, 0) / 18.0)])
    return np.column_stack([np.ones(n), *funcs])

def templates(n):
    ms = models()
    dist = np.arange(n)
    inverse, lnd = ([], [])
    for obs, slope, duration, _ in ms:
        cov = slope * np.exp(-dist / duration)
        cov[0] = 1.0
        cov[1] -= 0.5 * obs
        chol = np.linalg.cholesky(toeplitz(cov))
        inv = solve_triangular(chol, np.eye(n), lower=True, check_finite=False)
        inverse.append(inv)
        lnd.append(np.log(np.diag(chol)).sum())
    dev = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device='cpu')
    return (dev(inverse), dev(lnd), ms)

def scores(y, cut, bank):
    inv, logdet, ms = bank
    B, n = y.shape
    dev = inv.device
    x = torch.as_tensor(chart(n, cut), dtype=inv.dtype, device=dev)
    obs = torch.as_tensor(y, dtype=inv.dtype, device=dev)
    h = inv @ x
    w = (inv @ obs.T).transpose(1, 2)
    total = (w * w).sum(2)
    intercept = h[:, :, 0]
    c0 = (w * intercept[:, None]).sum(2)
    g0 = 1 / 0.18 ** 2 + intercept.square().sum(1)
    rss = (total - c0.square() / g0[:, None]).clamp_min(0)
    k = 3 + n / 2
    base = -logdet[:, None] - 0.5 * (g0 * 0.18 ** 2).log()[:, None] - k * (2.0 + 0.5 * rss).log()
    v = h[:, :, 1:]
    cross = (v * intercept[:, :, None]).sum(1)
    G = (v.square().sum(1) - cross.square() / g0[:, None]).clamp_min(0)
    u = w @ v - (c0 / g0[:, None])[:, :, None] * cross[:, None, :]
    alts = []
    for s in ((0.42, 0.02, 0.065), (1.4, 0.085, 0.22), (3.6, 0.26, 0.65)):
        sd = torch.tensor(s * 3, device=dev)
        den = 1 + G * sd.square()
        proj = (u.square() * sd.square() / den[:, None, :]).clamp_min(0)
        r = (rss[:, :, None] - proj).clamp_min(0)
        alts.append(base[:, :, None] - 0.5 * den.log()[:, None, :] - k * torch.log((2.0 + 0.5 * r) / (2.0 + 0.5 * rss[:, :, None])))
    ll = (torch.logsumexp(torch.stack(alts), 0) - np.log(len(alts))).cpu().numpy()
    base = base.cpu().numpy()
    prior = np.log(ms[:, 3])[:, None]
    marginal = logsumexp(base + prior, 0)
    change = logsumexp(ll + prior[:, :, None], axis=0) - marginal[:, None]
    fixed = ms[:, 1] == 0
    fixednorm = logsumexp(base[fixed] + prior[fixed], 0)
    no_slope = logsumexp(ll[fixed] + prior[fixed, :, None], axis=0) - fixednorm[:, None]
    post = np.exp(base + prior - marginal[None, :])
    vals = dict(slope_present=post[~fixed].sum(0), slow_slope=post[ms[:, 2] >= 64].sum(0), slope_null_gain=marginal - fixednorm + np.log(ms[fixed, 3].sum()))
    for i, name in enumerate(('level_jump', 'kink', 'temporary_drift')):
        vals[name] = change[:, i]
        vals[name + '_less_fixed'] = change[:, i] - no_slope[:, i]
        vals[name + '_at_boundary'] = change[:, i] - logsumexp(change[:, [3 + i, 6 + i]], axis=1) + np.log(2)
    return pd.DataFrame(vals)

def extract(part):
    path = OUT / f'wandering_slope_{part}.parquet'
    if path.exists():
        return
    with np.load(OUT / f'pulse_streams_{part}.npz') as a:
        ids, x, n, b = (a[k] for k in ('ids', 'separate', 'n', 'b'))
    width = np.minimum(b, n - b)
    radius = np.array(RADII)[np.searchsorted(RADII, width, side='right') - 1]
    assert np.all(radius <= width)
    frames = []
    for r in np.unique(radius):
        selected = np.flatnonzero(radius == r)
        length = 2 * r - 1
        prior = templates(length)
        for start in range(0, len(selected), 100):
            ix = selected[start:start + 100]
            ys = []
            for i in ix:
                delta = np.diff(x[i, b[i] - r:b[i] + r].astype(float))
                med = np.median(delta)
                scale = max(1.4826 * np.median(abs(delta - med)), 0.16 * delta.std(), 0.001)
                ys.append(np.clip(delta / scale, -16, 16))
            f = scores(np.stack(ys), r - 1, prior)
            f.index = ids[ix]
            frames.append(f)
    f = pd.concat(frames).loc[ids].clip(-85, 160).astype(np.float32)
    f.index.name = 'id'
    assert f.index.is_unique and np.isfinite(f).all().all()
    f.to_parquet(path)
    print(part, f.shape, flush=True)
if __name__ == '__main__':
    with torch.no_grad(), threadpool_limits(limits=1):
        for part in ('train', 'test'):
            extract(part)
