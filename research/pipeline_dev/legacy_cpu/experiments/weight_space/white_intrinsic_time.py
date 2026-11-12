# CPU port; original experiments/weight_space/white_intrinsic_time.py, SHA256 2382625d21902d2799ddc0e1fc0f1cae31dfc5792cf8ef7f4058bc833772f5fc
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Measure reversals of a fixed size even when their spacing in clock time varies."""
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
STEPS = np.array([0.75, 1.5, 2.5])

def measure(x, n, scale):
    threshold = scale[:, None] * STEPS
    shape = threshold.shape
    sign = np.zeros(shape, np.int8)
    high = np.broadcast_to(x[:, :1], shape).copy()
    low = high.copy()
    last = np.zeros(shape, int)
    total = np.zeros((*shape, 8))
    for t in range(1, x.shape[1]):
        now = x[:, t, None]
        active = t < n[:, None]
        high = np.where(active, np.maximum(high, now), high)
        low = np.where(active, np.minimum(low, now), low)
        up = active & (sign <= 0) & (now - low >= threshold)
        down = active & (sign >= 0) & (high - now >= threshold) & ~up
        event = up | down
        valid = event & (sign != 0)
        elapsed = t - last
        overshoot = np.maximum((high - low) / threshold - 1.0, 0.0).clip(0.0, 25.0)
        d = np.log1p(elapsed)
        val = np.log1p(overshoot)
        for j, field in enumerate((1.0, d, d * d, val, val * val, d * val, sign * d, sign * val)):
            total[:, :, j] += valid * field
        sign = np.where(up, 1, np.where(down, -1, sign))
        last = np.where(event, t, last)
        high = np.where(event, now, high)
        low = np.where(event, now, low)
    count = total[:, :, :1]
    avg = total[:, :, 1:] / np.maximum(count, 1.0)
    length, time2, size, size2, mixed, signed_time, signed_size = np.moveaxis(avg, 2, 0)
    rates = np.log((count[:, :, 0] + 0.5) / (n[:, None] + 1))
    features = np.stack((rates, length, size, np.maximum(time2 - length ** 2, 0.0), np.maximum(size2 - size ** 2, 0.0), mixed - length * size, signed_time, signed_size), axis=2)
    return (count[:, :, 0], features)

def extract(part):
    panels = []
    for tag in ('pulse', 'state'):
        with np.load(OUT / f'{tag}_streams_{part}.npz') as a:
            current, x, n, b = (a[k] for k in ('ids', 'separate', 'n', 'b'))
        if panels:
            assert np.array_equal(ids, current)
        ids = current
        frames = []
        for start in range(0, len(ids), 1024):
            sl = slice(start, start + 1024)
            sizes, cuts, data = (n[sl], b[sl], x[sl])
            s = []
            for row, length, cut in zip(data, sizes, cuts):
                diff = np.r_[np.diff(row[:cut]), np.diff(row[cut:length])]
                s.append(max(np.median(abs(diff)), 0.15 * diff.std(), 0.001))
            idx = np.arange(data.shape[1])[None, :]
            after = np.take_along_axis(data, (cuts[:, None] + idx).clip(max=data.shape[1] - 1), axis=1)
            c1, f1 = measure(data[:, :cuts.max()], cuts, np.asarray(s))
            c2, f2 = measure(after[:, :(sizes - cuts).max()], sizes - cuts, np.asarray(s))
            eff = np.sqrt((c1 + 0.5) * (c2 + 0.5) / (c1 + c2 + 1))
            delta = (f2 - f1) * eff[:, :, None]
            cols = {}
            for j, step in enumerate(STEPS):
                for k, name in enumerate(('rate', 'duration', 'overshoot', 'duration_spread', 'size_spread', 'duration_size', 'up_down_duration', 'up_down_size')):
                    cols[f'{tag}_{j}_{name}'] = abs(delta[:, j, k])
                cols[f'{tag}_{j}_faster'] = delta[:, j, 0]
            frames.append(pd.DataFrame(cols, index=ids[sl]))
        panels.append(pd.concat(frames))
    tab = pd.concat(panels, axis=1)
    tab.index.name = 'id'
    assert np.isfinite(tab).all().all()
    tab.astype(np.float32).to_parquet(OUT / f'intrinsic_time_{part}.parquet')
    print(part, tab.shape, flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        for part in ('train', 'test'):
            extract(part)
