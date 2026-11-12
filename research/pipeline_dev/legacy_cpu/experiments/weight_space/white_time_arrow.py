# CPU port; original experiments/weight_space/white_time_arrow.py, SHA256 1dd9d7ca954434b924e4d614bbdc72ff3aa881495c3a6024a49845a82c1f1a76
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Does the time direction of shock propagation change at the marked boundary?

The coarse versus fine comparison adapts the time-reversal diagnostic in
Gilles Zumbach, https://arxiv.org/abs/0708.4022, to observed innovations.
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
from threadpoolctl import threadpool_limits
from partition_evidence import OUT

def antisym(a, b, k):
    if len(a) <= k + 3:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    scale = max(np.sqrt(np.mean(a * a) * np.mean(b * b)), 0.025)
    return np.mean(a[:-k] * b[k:] - b[:-k] * a[k:]) / scale

def block(x):
    center = np.median(x)
    scale = max(1.4826 * np.median(abs(x - center)), 0.2 * x.std(), 0.0001)
    z = ((x - center) / scale).clip(-8, 8)
    mag = np.log1p(abs(z))
    tail = np.log1p(np.maximum(abs(z) - 1.3, 0.0) ** 2)
    response = [antisym(mag, tail, k) for k in (1, 3, 8)]
    for w in (2, 4, 8):
        s = np.r_[0.0, np.cumsum(z)]
        v = np.r_[0.0, np.cumsum(z * z)]
        fine = np.log(0.25 + (v[w:] - v[:-w]) / w)
        coarse = np.log(0.25 + (s[w:] - s[:-w]) ** 2 / w)
        response.append(antisym(coarse, fine, w))
    return np.array(response)

def segment(x):
    cuts = np.array_split(x, 4)
    mat = np.stack([block(v) for v in cuts])
    return np.stack([mat[::2].mean(0), mat[1::2].mean(0)])

def main(part):
    with np.load(OUT / f'pulse_streams_{part}.npz') as f:
        ids, raw, n, b = [f[k] for k in ('ids', 'separate', 'n', 'b')]
    ar = pd.read_parquet(OUT / f'state_{part}.parquet').loc[ids, ['ar1', 'ar2']].to_numpy()
    inputs = []
    names = ('tail_lag1', 'tail_lag3', 'tail_lag8', 'scale2', 'scale4', 'scale8')
    for row, end, cut, p in zip(raw, n, b, ar):

        def err(v):
            return v[2:] - v[1:-1] * p[0] - v[:-2] * p[1]
        left, right = (segment(err(row[:cut]).astype(float)), segment(err(row[cut:end]).astype(float)))
        delta = (right - left) * np.sqrt(cut * (end - cut) / end)
        inputs.append(np.stack([delta.mean(0), abs(delta.mean(0)), delta[0] * delta[1], (abs(right.mean(0)) - abs(left.mean(0))) * np.sqrt(cut * (end - cut) / end)]))
    mat = np.stack(inputs)
    tab = pd.DataFrame({f'{kind}_{name}': mat[:, i, j] for i, kind in enumerate(('delta', 'size', 'replicate', 'directionality')) for j, name in enumerate(names)}, index=ids)
    assert np.isfinite(tab).all().all()
    tab.to_parquet(OUT / f'time_arrow_{part}.parquet')
    print('time arrow', part, tab.shape, flush=True)
if __name__ == '__main__':
    check = np.sin(np.arange(73) * 0.36) + np.cos(np.arange(73) * 0.217) ** 3
    np.testing.assert_allclose(block(check), -block(check[::-1]), atol=1e-09)
    with threadpool_limits(limits=1):
        for part in ('train', 'test'):
            main(part)
