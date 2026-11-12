# CPU port; original experiments/weight_space/white_latent_duration.py, SHA256 1129fdb860adb5714ee0c6d66764524dffd1ac3aa742c009e8bf35b4642c81ca
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Compare changes in the duration and spread of unobserved high-variance spells."""
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
import itertools
import numpy as np
import pandas as pd
import torch
from scipy.special import gammaln, logsumexp
from threadpoolctl import threadpool_limits
from partition_evidence import OUT
MASS = np.array([0.2, 0.6, 0.2])
SPREAD = np.array([0.0, 0.24, 0.53, 0.86])
PERSIST = np.array([0.0, 0.5, 0.8, 0.94])
GRID = np.r_[np.linspace(0, 5, 251), np.geomspace(5.1, 24, 70)]

def bank():
    scale = np.exp(SPREAD[:, None] * np.arange(-1, 2))
    scale /= np.sqrt(scale ** 2 @ MASS)[:, None]
    scales = np.array([0.8, 1.0, 1.25])[:, None, None] * scale
    z = GRID[:, None, None, None] / scales
    e = np.stack([-0.5 * (np.log(2 * np.pi) + z * z), gammaln(3.5) - gammaln(3.0) - 0.5 * np.log(4 * np.pi) - 3.5 * np.log1p(z * z / 4)], axis=1)
    e -= np.log(scales)
    offset = e.max(axis=(1, 2, 3, 4))
    e = np.exp(e - offset[:, None, None, None, None]).clip(1e-25)
    transition = PERSIST[:, None, None] * np.eye(3) + (1 - PERSIST[:, None, None]) * MASS[None, None, :]
    return (e, transition)

def likelihood(code, lengths, e, transition):
    obs = e[code]
    m = obs[..., None, None, :] * transition
    valid = torch.arange(code.shape[1], device=code.device)[None, :] < lengths[:, None]
    m = torch.where(valid[:, :, None, None, None, None, None, None], m, torch.eye(3, device=code.device))
    scale = 0.0
    while m.shape[1] > 1:
        m = m[:, ::2] @ m[:, 1::2]
        norm = m.amax(dim=(-1, -2)).clamp_min(1e-30)
        scale = scale + norm.log().sum(1)
        m = m / norm[..., None, None]
    stationary = torch.as_tensor(MASS, dtype=e.dtype, device=e.device)
    return (scale + (m[:, 0].sum(-1) @ stationary).log()).cpu().numpy()

def check(e, trans):
    raw, matrices = bank()
    codes = [5, 82, 151]
    ll = likelihood(torch.tensor([codes + [0]], device=e.device), torch.tensor([len(codes)], device=e.device), e, trans)[0]
    for law, scale, d, r in itertools.product(range(2), range(3), range(4), range(4)):
        p = MASS.copy()
        for k in codes:
            p = p @ matrices[r] * raw[k, law, scale, d]
        assert abs(np.log(p.sum()) - ll[law, scale, d, r]) < 4e-05

def evidence(a, b):
    law = np.log([0.65, 0.35])[None, :, None, None]
    spread = np.log([0.68, 0.15, 0.12, 0.05])[None, None, :, None]
    dur = np.log([0.55, 0.22, 0.16, 0.07])[None, None, None, :]
    prior = law + spread + dur

    def total(x):
        return logsumexp(x, axis=(1, 2, 3))
    a = logsumexp(a + np.log([0.2, 0.6, 0.2])[None, None, :, None, None], axis=2)
    b = logsumexp(b + np.log([0.2, 0.6, 0.2])[None, None, :, None, None], axis=2)
    a -= a.max((1, 2, 3), keepdims=True)
    b -= b.max((1, 2, 3), keepdims=True)
    null = total(a + b + prior)

    def marginal(x, p, dim):
        return logsumexp(x + p, axis=dim, keepdims=True)
    pre, post = (total(a + prior), total(b + prior))
    out = dict(duration=marginal(a, dur, 3) + marginal(b, dur, 3) + law + spread, dispersion=marginal(a, spread, 2) + marginal(b, spread, 2) + law + dur, duration_and_dispersion=marginal(a, spread + dur, (2, 3)) + marginal(b, spread + dur, (2, 3)) + law)
    out = {k: total(v) - null for k, v in out.items()}
    for prefix, x, norm in (('pre', a, pre), ('post', b, post)):
        posterior = np.exp(x + prior - norm[:, None, None, None])
        out[prefix + '_cluster'] = posterior[:, :, 1:, 1:].sum((1, 2, 3))
        out[prefix + '_long'] = posterior[:, :, 1:, -1].sum((1, 2))
        out[prefix + '_strong'] = posterior[:, :, -1, 1:].sum((1, 2))
    out['cluster_change'] = out['post_cluster'] - out['pre_cluster']
    return out

def extract(part, e, trans):
    dest = OUT / f'latent_duration_{part}.parquet'
    if dest.exists():
        return
    with np.load(OUT / f'pulse_streams_{part}.npz') as a:
        ids, x, n, cut = (a[k] for k in ('ids', 'separate', 'n', 'b'))
    ar = pd.read_parquet(OUT / f'state_{part}.parquet').loc[ids, ['ar1', 'ar2']].to_numpy()
    frames = []
    for start in range(0, len(ids), 32):
        sl = slice(start, start + 32)
        sums = []
        for side in (0, 1):
            codes = []
            for row, end, j, phi in zip(x[sl], n[sl], cut[sl], ar[sl]):
                v = row[:j] if side == 0 else row[j:end]
                v = v[2:] - phi[0] * v[1:-1] - phi[1] * v[:-2]
                v = v[-254:] if side == 0 else v[:254]
                z = abs(v - np.median(v))
                winsorized = np.minimum(z, np.quantile(z, 0.99))
                signal = z / max(float(np.sqrt(np.mean(winsorized ** 2))), 1e-05)
                codes.append(np.searchsorted((GRID[1:] + GRID[:-1]) * 0.5, signal))
            length = np.array([len(v) for v in codes])
            code = np.zeros((len(length), 1 << (int(length.max()) - 1).bit_length()), dtype=np.int32)
            for i, v in enumerate(codes):
                code[i, :len(v)] = v
            sums.append(likelihood(torch.as_tensor(code, device=e.device), torch.as_tensor(length, device=e.device), e, trans))
        frames.append(pd.DataFrame(evidence(*sums), index=ids[sl]))
        if start and start % 4096 == 0:
            print(part, start, flush=True)
    df = pd.concat(frames).clip(-80, 160).astype(np.float32)
    df.index.name = 'id'
    assert df.index.is_unique and np.isfinite(df).all().all()
    df.to_parquet(dest)
    print(part, df.shape, flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        tensors = [torch.as_tensor(v, dtype=torch.float32, device='cpu') for v in bank()]
        check(*tensors)
        for part in ('train', 'test'):
            extract(part, *tensors)
