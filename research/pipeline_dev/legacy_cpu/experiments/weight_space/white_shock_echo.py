# CPU port; original experiments/weight_space/white_shock_echo.py, SHA256 147593c39900c20cfafa83ba3bf41f8120d0e4938e22069aa6a788ed64ae5d46
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Compare a one-step shock echo with feedback from the preceding observation."""
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
PHI = np.array([-0.83, -0.56, -0.28, 0.0, 0.28, 0.58, 0.83, 0.97])
THETA = np.array([-0.78, -0.49, -0.23, 0.0, 0.23, 0.49, 0.78])
T = 256

def evidence(v, length):
    device = v.device
    t = torch.arange(T, device=device)
    phi = torch.tensor(PHI, device=device, dtype=v.dtype)[None, :, None, None]
    theta = torch.tensor(THETA, device=device, dtype=v.dtype)
    kernel = (-theta[:, None]) ** t[None, :]
    kernel = torch.fft.rfft(kernel, n=2 * T)
    valid = ((t[None, :] >= 12) & (t[None, :] < (length - 12)[:, None]))[:, None, None, :]
    count = valid.sum(-1)
    ll = []
    for inp in (v, v.gather(1, (length[:, None] - 1 - t).clamp(min=0))):
        f = torch.fft.rfft(inp, n=2 * T)
        u = torch.fft.irfft(f[:, None, :] * kernel[None, :, :], n=2 * T)[:, :, :T]
        prev = torch.nn.functional.pad(u[:, :, :-1], (1, 0))
        e = u[:, None, :, :] - phi * prev[:, None, :, :]
        e = e - (e * valid).sum(-1, keepdim=True) / count[..., None]
        var = (e.square() * valid).sum(-1) / count
        var = var.clamp_min(1e-05)
        gaussian = -0.5 * count * (1.0 + np.log(2 * np.pi) + var.log())
        robust = var.clone()
        for _ in range(3):
            robust = (6 * e.square() / (3 + e.square() / robust[..., None]) * valid).sum(-1) / count
            robust = robust.clamp_min(1e-05)
        student = (np.log(8 / (3 * np.pi * np.sqrt(3))) - 0.5 * robust.log()) * count
        student -= 3.0 * (torch.log1p(e.square() / (3 * robust[..., None])) * valid).sum(-1)
        ll.append(torch.stack([gaussian, student], 1))
    return torch.stack(ll, 2).cpu().numpy()

def features(a, b):
    pa = np.exp(-0.5 * (PHI / 0.62) ** 2)
    pt = np.exp(-0.5 * (THETA / 0.4) ** 2)
    pa[3], pt[3] = (0.0, 0.0)
    pa /= pa.sum()
    pt /= pt.sum()
    mix = 0.08 * pa[:, None] * pt[None, :]
    mix[:, 3] += 0.33 * pa
    mix[3, :] += 0.14 * pt
    mix[3, 3] += 0.45
    prior = np.log(mix)[None, None, None, :, :] + np.log([0.65, 0.35])[None, :, None, None, None] - np.log(2)
    a = a - a.max((1, 2, 3, 4), keepdims=True)
    b = b - b.max((1, 2, 3, 4), keepdims=True)
    joint = a + b + prior
    total = lambda x: logsumexp(x, axis=(1, 2, 3, 4))
    null = total(joint)

    def change(dim):
        marginal = logsumexp(prior, axis=dim, keepdims=True)
        c = prior - marginal
        return total(logsumexp(a + c, axis=dim, keepdims=True) + logsumexp(b + c, axis=dim, keepdims=True) + marginal) - null
    vals = {'feedback_change': change(3), 'echo_change': change(4), 'arrow_change': change(2), 'echo_and_arrow': change((2, 4)), 'dynamics_change': change((3, 4)), 'noise_change': change(1)}
    for goal, x in (('pre', a), ('post', b)):
        posterior = np.exp(x + prior - total(x + prior)[:, None, None, None, None])
        vals[goal + '_echo'] = 1.0 - posterior[:, :, :, :, 3].sum((1, 2, 3))
        vals[goal + '_reverse'] = posterior[:, :, 1].sum((1, 2, 3))
    vals['relative_echo'] = vals['echo_change'] - vals['feedback_change']
    return vals

def extract(part):
    path = OUT / f'shock_echo_{part}.parquet'
    if path.exists():
        return
    with np.load(OUT / f'pulse_streams_{part}.npz') as a:
        ids, x, n, cut = (a[k] for k in ('ids', 'separate', 'n', 'b'))
    blocks = []
    for ch in ('signal', 'sizes'):
        data = []
        for start in range(0, len(ids), 96):
            sl = slice(start, start + 96)
            ll = []
            for side in (0, 1):
                mat = np.zeros((len(n[sl]), T), np.float32)
                lens = []
                for i, (v, j, end) in enumerate(zip(x[sl], cut[sl], n[sl])):
                    v = v[max(0, j - T):j] if side == 0 else v[j:min(end, j + T)]
                    v = v - np.median(v)
                    v = v / max(1.4826 * np.median(abs(v)), 0.18 * v.std(), 0.001)
                    if ch == 'sizes':
                        v = np.log(0.25 + abs(v))
                    mat[i, :len(v)] = (v - np.median(v)).clip(-15, 15)
                    lens.append(len(v))
                ll.append(evidence(torch.tensor(mat, device='cpu'), torch.tensor(lens, device='cpu')))
            data.append(pd.DataFrame(features(*ll), index=ids[sl]))
        blocks.append(pd.concat(data).add_prefix(ch + '_'))
    f = pd.concat(blocks, axis=1).clip(-80, 140).astype(np.float32)
    f.index.name = 'id'
    assert f.index.is_unique and np.isfinite(f).all().all()
    f.to_parquet(path)
    print(part, f.shape, flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1), torch.no_grad():
        for part in ('train', 'test'):
            extract(part)
