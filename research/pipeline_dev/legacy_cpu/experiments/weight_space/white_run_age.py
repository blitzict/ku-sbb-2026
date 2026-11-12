# CPU port; original experiments/weight_space/white_run_age.py, SHA256 ee7c211431d29d7a715d2a8ad7291abe9601868f7f2be5c9b6e3f6da2b499ab2
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Does a run become more or less likely to end as it ages across the break?"""
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

def count_risk(states):
    """Count exits and exposure by state, age and alternating observation blocks."""
    n, channels = states.shape
    t = np.arange(n)
    block = t * 4 // n
    boundary = np.r_[True, np.diff(block) != 0]
    switch = np.vstack([np.zeros(channels, bool), states[1:] != states[:-1]])
    switch[boundary] = False
    last = np.maximum.accumulate(np.where(switch, t[:, None], -1), axis=0)
    block_start = np.maximum.accumulate(np.where(boundary, t, 0))
    use = (last >= block_start[:, None])[:-1] & ~boundary[1:, None]
    age = t[:, None] - last + 1
    group = (age > 1).astype(int) + (age > 3)
    index = (((np.arange(channels) + channels * (block[:, None] % 2)) * 2 + states) * 3 + group)[:-1]

    def counts(mask):
        return np.bincount(index[mask], minlength=2 * channels * 2 * 3).reshape(2, channels, 2, 3)
    return np.stack([counts(use & switch[1:]), counts(use)], axis=-1)

def contrast(counts):
    ended, exposure = (counts[..., 0], counts[..., 1])
    survived = exposure - ended
    odds = np.log((ended + 1.0) / (survived + 1.0))
    variance = 1.0 / (ended + 1.0) + 1.0 / (survived + 1.0)
    delta = odds[..., 1:] - odds[..., :1]
    return (delta, variance[..., 1:] + variance[..., :1])

def measures(before, after):

    def standardized(left, right):
        u, vu = contrast(left)
        v, vv = contrast(right)
        return (v - u) / np.sqrt(vu + vv)
    total = standardized(before.sum(0), after.sum(0))
    split = standardized(before, after)
    return np.column_stack([np.mean(total ** 2, axis=(1, 2)), total[:, :, 1].mean(1), (total[:, 1] - total[:, 0]).mean(1), np.mean(split[0] * split[1], axis=(1, 2))])

def study(x):
    center = np.median(x)
    magnitude = abs(x - center)
    return count_risk(np.column_stack([x > center, magnitude > np.median(magnitude), magnitude > np.quantile(magnitude, 0.78)]).astype(int))

def extract(part):
    columns = {}
    for stream in ('pulse', 'state'):
        with np.load(OUT / f'{stream}_streams_{part}.npz') as f:
            current, rows, n, cuts = (f[k] for k in ('ids', 'separate', 'n', 'b'))
        if columns:
            assert np.array_equal(current, ids)
        ids = current
        values = np.stack([measures(study(x[:b]), study(x[b:end])) for x, b, end in zip(rows, cuts, n)])
        for col, name in enumerate(('direction', 'amplitude', 'tail')):
            for j, stat in enumerate(('change', 'late', 'asymmetry', 'replicated')):
                columns[f'{stream}_{name}_{stat}'] = values[:, col, j]
    frame = pd.DataFrame(columns, index=ids)
    assert np.isfinite(frame).all().all()
    frame.to_parquet(OUT / f'run_age_{part}.parquet')
    print(part, frame.shape, flush=True)
if __name__ == '__main__':
    toy = count_risk(np.tile([0, 0, 1, 1, 1, 0, 0, 0], 4).reshape(-1, 1)).sum(0)[0]
    np.testing.assert_array_equal(toy[1], [[0, 4], [4, 8], [0, 0]])
    np.testing.assert_array_equal(toy[0], [[0, 4], [0, 4], [0, 0]])
    with threadpool_limits(limits=1):
        for part in ('train', 'test'):
            extract(part)
