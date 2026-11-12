# CPU port; original experiments/weight_space/white_context_changes.py, SHA256 d7326a54846b427b92cb725fcf98c3db2682a443d6d4bf2fd4e0070eb3800486
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Test changes in short event histories using binary context-tree probability mixtures.

Willems, Shtarkov and Tjalkens (1995), doi:10.1109/18.382012.
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
from scipy.special import betaln
from threadpoolctl import threadpool_limits
from partition_evidence import OUT
DEPTH = 6
KINDS = ('sign', 'size', 'tail', 'size_to_sign', 'sign_to_size')

def observations(x):
    center = np.median(x)
    mag = abs(x - center)
    sign = x > center
    size = mag > np.median(mag)
    tail = mag > np.quantile(mag, 0.8)
    source = np.column_stack([sign, size, tail, size, sign]).astype(int)
    target = np.column_stack([sign, size, tail, sign, size]).astype(int)
    block = np.arange(len(x)) * 4 // len(x)
    keep = block[DEPTH:] == block[:-DEPTH]
    code = np.zeros((len(x) - DEPTH, len(KINDS)), int)
    for lag in range(1, DEPTH + 1):
        code = 2 * code + source[DEPTH - lag:len(x) - lag]
    index = ((block[DEPTH:, None] % 2 * len(KINDS) + np.arange(len(KINDS))) * 2 ** DEPTH + code) * 2 + target[DEPTH:]
    return np.bincount(index[keep].ravel(), minlength=2 * len(KINDS) * 2 ** (DEPTH + 1)).reshape(2, len(KINDS), 2 ** DEPTH, 2)

def mixture(counts):

    def kt(a):
        return betaln(a[..., 0] + 0.5, a[..., 1] + 0.5) - np.log(np.pi)
    cur = kt(counts)
    for _ in range(DEPTH):
        n = counts.shape[-2]
        counts = counts.reshape(*counts.shape[:-2], n // 2, 2, 2).sum(-2)
        own = kt(counts)
        cur = np.logaddexp(own, cur[..., ::2] + cur[..., 1::2]) - np.log(2)
    return (cur[..., 0], own[..., 0])

def features(left, right):

    def calc(a, b):
        la, ma = mixture(a)
        lb, mb = mixture(b)
        ab, mab = mixture(a + b)
        return (la + lb - ab, la - ma + (lb - mb) - (ab - mab), la - ma, lb - mb)
    total, ordering, pre, post = calc(left.sum(1), right.sum(1))
    _, paired, _, _ = calc(left, right)
    return dict(change=total, ordering=ordering, replicate=np.min(paired, axis=1), pre=pre, post=post)

def main(part):
    result = {}
    for channel in ('pulse', 'state'):
        with np.load(OUT / f'{channel}_streams_{part}.npz') as a:
            current, raw, n, b = (a[k] for k in ('ids', 'separate', 'n', 'b'))
        if result:
            assert np.array_equal(current, ids)
        ids = current
        acc = []
        for start in range(0, len(ids), 128):
            sl = slice(start, start + 128)
            left = np.array([observations(x[:j]) for x, j in zip(raw[sl], b[sl])])
            right = np.array([observations(x[j:end]) for x, j, end in zip(raw[sl], b[sl], n[sl])])
            vals = features(left, right)
            acc.append(np.stack(list(vals.values()), axis=1))
        mat = np.concatenate(acc)
        for i, k in enumerate(vals):
            for j, law in enumerate(KINDS):
                result[f'{channel}_{law}_{k}'] = mat[:, i, j]
    frame = pd.DataFrame(result, index=ids)
    assert np.isfinite(frame).all().all()
    frame.to_parquet(OUT / f'context_changes_{part}.parquet')
    print('context trees', part, frame.shape, flush=True)
if __name__ == '__main__':
    empty = np.zeros((3, 2 ** DEPTH, 2), int)
    assert np.max(abs(mixture(empty)[0])) < 1e-10
    t = observations(np.sin(np.arange(120) * 1.23))
    assert (t.sum((0, 2, 3)) == 120 - 4 * DEPTH).all()
    with threadpool_limits(limits=1):
        for part in ('train', 'test'):
            main(part)
