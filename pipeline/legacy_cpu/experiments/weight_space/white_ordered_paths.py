# CPU port; original experiments/weight_space/white_ordered_paths.py, SHA256 985b1798e3fb5707b5b7aff2169e40507ce8592614ce2419e64e65f9e70645e3
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Compare the ordered paths of signed and absolute innovations before and after the break.

For the signature transform see Chevyrev and Kormilitzin, A Primer on the
Signature Method in Machine Learning, doi:10.1007/978-3-031-97239-3_1.
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
import torch
from threadpoolctl import threadpool_limits
from white_shock_quantiles import run
from partition_evidence import OUT

def signature(d):
    first = d
    second = d.unsqueeze(-1) * d.unsqueeze(-2) / 2
    third = second.unsqueeze(-1) * d[..., None, None, :] / 3
    while first.shape[1] > 1:
        a, b = (first[:, ::2], first[:, 1::2])
        aa, bb = (second[:, ::2], second[:, 1::2])
        third = third[:, ::2] + third[:, 1::2] + aa.unsqueeze(-1) * b[..., None, None, :] + a[..., None, None] * bb.unsqueeze(-3)
        second = aa + bb + a.unsqueeze(-1) * b.unsqueeze(-2)
        first = a + b
    return (first[:, 0], second[:, 0], third[:, 0])

def one_batch(rows, lengths, cuts):
    chunks = []
    for row, n, cut in zip(rows, lengths, cuts):
        v = row[:n].astype(float)
        center = np.median(v)
        scale = max(np.median(abs(v - center)) * 1.4826, 0.2 * v.std(), 0.01)
        z = (v - center) / scale
        mid = np.median(abs(z))
        observations = np.column_stack([np.tanh(z), np.tanh(abs(z) - mid)])
        for side in (observations[:cut], observations[cut:]):
            for chunk in np.array_split(side, 12):
                chunks.append(np.column_stack([np.full(len(chunk), 1 / len(chunk)), chunk / np.sqrt(len(chunk))]))
    width = 1 << (max(map(len, chunks)) - 1).bit_length()
    padded = np.zeros((len(chunks), width, 3), np.float32)
    for i, a in enumerate(chunks):
        padded[i, :len(a)] = a
    d = torch.as_tensor(padded, device='cpu')
    first, second, third = signature(d)
    m = torch.cat([first, second.flatten(1), third.flatten(1)], 1).asinh().cpu().numpy()
    m = m.reshape(len(rows), 2, 12, -1)
    left, right = (m[:, 0, ::-1], m[:, 1])
    return np.concatenate([left, right, right - left], axis=2).transpose(0, 2, 1)

def charts(part):
    dst = OUT / f'ordered_path_curves_{part}.npz'
    if not dst.exists():
        tables = []
        for tag in ('pulse', 'state'):
            with np.load(OUT / f'{tag}_streams_{part}.npz') as f:
                current, raw, n, b = [f[k] for k in ('ids', 'separate', 'n', 'b')]
            if tables:
                assert np.array_equal(ids, current)
            ids = current
            batches = [one_batch(raw[j:j + 64], n[j:j + 64], b[j:j + 64]) for j in range(0, len(ids), 64)]
            tables.append(np.vstack(batches))
        h = np.concatenate(tables, 1).astype(np.float16)
        assert np.isfinite(h).all()
        np.savez_compressed(dst, ids=ids, h=h)
        print('path coordinates', part, h.shape, flush=True)
    with np.load(dst) as f:
        return (f['ids'], f['h'].astype(np.float32))

def check():
    d = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]])
    a, b, c = signature(d)
    torch.testing.assert_close(b + b.transpose(1, 2), a[:, :, None] * a[:, None, :])
    assert c[0, 0, 1, 2] == 1 and c[0, 2, 1, 0] == 0
    torch.testing.assert_close(c[0, 0, 0, 0], torch.tensor(1 / 6))
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        torch.set_num_threads(1)
        check()
        run(charts, 'ordered_path_models', ('law_path_control', 'law_ordered_paths'))
