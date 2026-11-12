# CPU port; original experiments/weight_space/white_motif_dictionary.py, SHA256 91c8c130a4e0ff2bc782edb0528cf414f123c952e3995738482a0bbd87cb3107
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Compare short waveform counts using HYDRA-style competing filters on actual observations."""
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
from torch.nn import functional as F
from threadpoolctl import threadpool_limits
from partition_evidence import OUT
DILATIONS = (1, 2, 4, 8)
GROUPS, KERNELS, WIDTH = (8, 8, 5)

def filters():
    rng = np.random.default_rng(31714)
    w = rng.normal(size=(GROUPS, KERNELS, WIDTH))
    w -= w.mean(axis=-1, keepdims=True)
    w /= np.sqrt((w * w).sum(axis=-1, keepdims=True))
    return torch.as_tensor(w.reshape(-1, 1, WIDTH), dtype=torch.float32, device='cpu')

def measure(x, n, b, w, dilation):
    span = (WIDTH - 1) * dilation
    conv = F.conv1d(x[:, None], w, dilation=dilation)
    conv = conv.reshape(len(n), GROUPS, KERNELS, -1)
    maxval, winner = conv.max(2)
    t = torch.arange(conv.shape[-1], device=x.device)[None, :]
    right = t >= b[:, None]
    left_len, right_len = (b - span, n - b - span)
    length = torch.where(right, right_len[:, None], left_len[:, None])
    origin = torch.where(right, b[:, None], 0)
    offset = t - origin
    valid = (offset >= 0) & (offset < length)
    bin_no = (offset * 4 // length.clamp_min(1)).clamp(0, 3) + 4 * right
    cell = winner + KERNELS * bin_no[:, None, :]
    cnt = torch.zeros(len(n), GROUPS, KERNELS * 8, device=x.device)
    quality = torch.zeros_like(cnt)
    cnt.scatter_add_(2, cell, valid[:, None, :].expand_as(cell).float())
    quality.scatter_add_(2, cell, valid[:, None, :] * maxval)
    cnt = cnt.reshape(len(n), GROUPS, 2, 4, KERNELS)
    quality = quality.reshape_as(cnt)
    denom = cnt.sum(-1, keepdims=True).clamp_min(1)
    out = []
    for h in (cnt / denom, quality / denom):
        delta = h[:, :, 1] - h[:, :, 0]
        pooled = h.mean(2)
        se = (h.var(3, unbiased=False).sum(2) / 3 + 0.3 * pooled.mean(2).clamp_min(0.025) * (1 / right_len + 1 / left_len)[:, None, None]).sqrt().clamp_min(0.008)
        even = delta[:, :, [0, 2]].mean(2) / se
        odd = delta[:, :, [1, 3]].mean(2) / se
        cross = (even * odd).clamp(-30, 55)
        mag = ((even + odd) / 2).square().clamp_max(80)
        out.extend([cross.mean((1, 2)), cross.amax(2).mean(1), mag.mean((1, 2)), mag.amax(2).mean(1), (even.sign() == odd.sign()).float().mean((1, 2))])
    return torch.stack(out, 1)

def normalized(row):
    z = row - np.median(row)
    return np.clip(z / max(1.4826 * np.median(abs(z)), 0.2 * z.std(), 0.0001), -4.0, 4.0)

def extract(part, w):
    path = OUT / f'motif_dictionary_{part}.parquet'
    if path.exists():
        return
    with np.load(OUT / f'pulse_streams_{part}.npz') as a:
        ids, x, n, b = (a[k] for k in ('ids', 'separate', 'n', 'b'))
    with np.load(OUT / f'state_streams_{part}.npz') as a:
        assert np.array_equal(ids, a['ids']) and np.array_equal(b, a['b'])
        innovations = a['separate']
    with np.load(OUT / f'mode_residual_streams_{part}.npz') as a:
        assert np.array_equal(ids, a['ids']) and np.array_equal(b, a['b'])
        modes = a['common']
    cols = [f'{name}_d{d}_{kind}_{stat}' for name in ('pulse', 'innovation', 'size', 'two_mode') for d in DILATIONS for kind in ('counts', 'response') for stat in ('repeat', 'specific_repeat', 'size', 'specific_size', 'agreement')]
    table = []
    for begin in range(0, len(ids), 96):
        sl = slice(begin, begin + 96)
        ends, cuts = (n[sl], b[sl])
        stages = []
        for name, raw in (('pulse', x[sl]), ('innovation', innovations[sl]), ('size', innovations[sl]), ('two_mode', modes[sl])):
            values = np.zeros((len(ends), int(ends.max())), np.float32)
            for i, (v, end, j) in enumerate(zip(raw, ends, cuts)):
                v = np.r_[normalized(v[:j]), normalized(v[j:end])]
                values[i, :end] = np.abs(v) if name == 'size' else v
            t = torch.as_tensor(values, device=w.device)
            lengths, c = [torch.as_tensor(v, device=w.device) for v in (ends, cuts)]
            assert int(torch.minimum(lengths - c, c).min()) > (WIDTH - 1) * max(DILATIONS)
            stages.extend((measure(t, lengths, c, w, d) for d in DILATIONS))
        table.append(torch.cat(stages, 1).cpu().numpy())
    frame = pd.DataFrame(np.vstack(table), index=pd.Index(ids, name='id'), columns=cols)
    assert frame.index.is_unique and np.isfinite(frame).all().all()
    frame.to_parquet(path)
    print(part, frame.shape, flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1), torch.no_grad():
        kernels = filters()
        for part in ('train', 'test'):
            extract(part, kernels)
