# CPU port; original experiments/weight_space/white_conditional_noise_law.py, SHA256 78268ac5e1e481c8086a440394fa0a5a82919d2673a75a11459038f014612e0a
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Test which parts of the noise distribution change during calm and active spells."""
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
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from partition_evidence import OUT
DELTAS = np.array([-0.95, -0.48, -0.2, 0.2, 0.48, 0.95])
LP = np.log([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
NAMES = ('unadjusted', 'shared', 'separate', 'calm', 'active', 'tail_shared', 'tail_separate', 'opposed')

def histograms(x, h, b, shape):
    v = np.r_[x[:b] - np.median(x[:b]), x[b:] - np.median(x[b:])]
    if shape == 'whitened':
        v = v / h
    if shape != 'raw':
        for sl in (slice(0, b), slice(b, None)):
            u = v[sl]
            v[sl] = u / max(1.4826 * np.median(abs(u)), 0.2 * u.std(), 0.001)
    bin_id = np.searchsorted(np.quantile(v, np.arange(1, 16) / 16), v)
    state = np.searchsorted(np.quantile(h, [1 / 3, 2 / 3]), h)
    side = np.arange(len(v)) >= b
    return np.bincount(16 * (3 * side + state) + bin_id, minlength=96).reshape(2, 3, 16)

def tilts(hist):
    a, c = (hist[:, 0], hist[:, 1])

    def nodes(side):
        left, right = ([], [])
        for size in (16, 8, 4, 2):
            s = side.reshape(len(side), side.shape[1], -1, 2, size // 2).sum(-1)
            left.append(s[:, :, :, 0])
            right.append(s[:, :, :, 1])
        return (torch.cat(left, 2), torch.cat(right, 2))
    pre, pre_other = nodes(a)
    post, post_other = nodes(c)
    successes, failures = (pre + post, pre_other + post_other)
    draws = post + post_other
    lower = (draws - failures).clamp_min(0)
    upper = torch.minimum(draws, successes)
    k = lower[..., None] + torch.arange(int((upper - lower).max().item()) + 1, device=a.device)
    valid = k <= upper[..., None]
    fac = torch.arange(int(hist.sum((1, 2, 3)).max().item()) + 1, dtype=torch.float32, device=a.device).add(1).lgamma()
    mass = -sum((fac[i.clamp(0, len(fac) - 1)] for i in (k, successes[..., None] - k, draws[..., None] - k, failures[..., None] - draws[..., None] + k)))
    mass = mass.masked_fill(~valid, -torch.inf)
    norm = mass.logsumexp(-1)
    off = k - post[..., None]
    return torch.stack([norm - (mass + d * off).logsumexp(-1) for d in DELTAS], -1).cpu().numpy()

def summary(h):
    orig = tilts(h)
    pooled = tilts(h.sum(2, keepdims=True))[:, 0]
    independent = logsumexp(orig + LP, axis=-1)
    shared = logsumexp(orig.sum(1) + LP, axis=-1)
    opposed = logsumexp(orig[:, 0] + orig[:, 2, :, ::-1] + LP, axis=-1)
    w = np.repeat([0.2, 0.16, 0.11, 0.065], [1, 2, 4, 8])

    def mix_nodes(x, tail=False):
        idx = np.array([3, 6, 7, 8, 13, 14]) if tail else np.arange(15)
        return np.logaddexp(np.log1p(-w[idx]), np.log(w[idx]) + x[:, idx]).sum(1)
    varying = np.logaddexp(np.log(0.65), np.log(0.35) + independent).sum(1)
    return np.stack([mix_nodes(logsumexp(pooled + LP, axis=-1)), mix_nodes(shared), mix_nodes(varying), mix_nodes(independent[:, 0]), mix_nodes(independent[:, 2]), mix_nodes(shared, True), mix_nodes(varying, True), mix_nodes(opposed)], 1)

def extract(part):
    target = OUT / f'conditional_noise_law_{part}.parquet'
    if target.exists():
        return
    with np.load(OUT / f'hetero_path_{part}.npz') as f:
        ids, x, n, b, h = (f[k] for k in ('ids', 'u', 'n', 'b', 'common'))
    views = []
    for shape in ('raw', 'shape', 'whitened'):
        data = np.stack([histograms(v[:end].astype(float), field[:end].astype(float), j, shape) for v, field, end, j in zip(x, h, n, b)])
        assert np.array_equal(data.sum((1, 2, 3)), n)
        vals = []
        for begin in range(0, len(ids), 64):
            counts = torch.as_tensor(data[begin:begin + 64], device='cpu')
            if begin == 0:
                a = tilts(counts[:4])
                swap = tilts(counts[:4].flip((1,)))
                assert np.max(abs(a - swap[..., ::-1])) < 0.002
            vals.append(summary(counts))
        v = np.vstack(vals)
        frame = pd.DataFrame(v, index=pd.Index(ids, name='id'), columns=[shape + '_' + k for k in NAMES])
        frame[shape + '_extra'] = v[:, 2] - v[:, 0]
        frame[shape + '_by_state'] = v[:, 2] - v[:, 1]
        views.append(frame)
    out = pd.concat(views, axis=1).clip(-75, 150).astype(np.float32)
    assert out.index.is_unique and np.isfinite(out).all().all()
    out.to_parquet(target)
    print(part, out.shape, flush=True)
if __name__ == '__main__':
    with torch.no_grad(), threadpool_limits(limits=1):
        for part in ('train', 'test'):
            extract(part)
