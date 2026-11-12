# CPU port; original experiments/weight_space/white_ordinal_motifs.py, SHA256 f065f5efa8b3fd214c97fc6b99c0e358c5322cb4e2f1c5c1c9dc2a1fa71d92e9
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Look for changes in the order of four observed values at fixed time spacings."""
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
from itertools import permutations
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from partition_evidence import OUT
ORDERS = np.array(list(permutations(range(4))))
CODE = np.array([1, 4, 16, 64])
LOOKUP = np.full(256, -1)
LOOKUP[ORDERS @ CODE] = np.arange(24)
LAGS = (1, 3, 8, 16)

def block_rates(x, lag):
    order = np.argsort(np.column_stack([x[j * lag:len(x) - (3 - j) * lag] for j in range(4)]), axis=1, kind='stable')
    symbol = LOOKUP[order @ CODE]
    assert np.all(symbol >= 0)
    blocks = np.arange(len(symbol)) * 6 // len(symbol)
    counts = np.bincount(blocks * 24 + symbol, minlength=144).reshape(6, 24)
    return (counts + 0.25) / (counts.sum(1, keepdims=True) + 6)

def contrast(left, right):
    sizes = np.maximum(left.var(0, ddof=1) + right.var(0, ddof=1), 0.0001)
    diff = right.mean(0) - left.mean(0)
    z = diff * np.sqrt(6 / sizes)
    first = (right[::2] - left[::2]).mean(0)
    second = (right[1::2] - left[1::2]).mean(0)
    return (z, np.arcsinh(6 * first * second / sizes))

def extract(part):
    path = OUT / f'ordinal_motifs_{part}.npz'
    if path.exists():
        return
    arrays = []
    for tag in ('pulse', 'state'):
        with np.load(OUT / f'{tag}_streams_{part}.npz') as a:
            current, x, n, b = (a[k] for k in ('ids', 'separate', 'n', 'b'))
        if arrays:
            assert np.array_equal(current, ids)
        ids = current
        table = np.empty((len(ids), len(LAGS), 2, 24), dtype=np.float32)
        for i, (length, cut) in enumerate(zip(n, b)):
            for j, lag in enumerate(LAGS):
                table[i, j] = contrast(block_rates(x[i, :cut], lag), block_rates(x[i, cut:length], lag))
        arrays.append(table)
    x = np.stack(arrays, axis=1)
    assert np.isfinite(x).all()
    np.savez_compressed(path, ids=ids, x=x.astype(np.float16))
    ranks = np.argsort(ORDERS, axis=1)
    steps = np.diff(ranks, axis=1)
    turns = (steps[:, 1:] * steps[:, :-1] < 0).sum(1)
    out = {}
    for s, tag in enumerate(('pulse', 'state')):
        for j, lag in enumerate(LAGS):
            delta, rep = (x[:, s, j, 0], x[:, s, j, 1])
            out[f'{tag}_{lag}_replicated'] = rep.mean(1)
            out[f'{tag}_{lag}_strongest'] = np.max(abs(delta), axis=1)
            for k in (0, 1, 2):
                out[f'{tag}_{lag}_turns{k}'] = delta[:, turns == k].mean(1)
    pd.DataFrame(out, index=pd.Index(ids, name='id')).to_parquet(OUT / f'ordinal_motifs_{part}.parquet')
    print(part, x.shape, flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        for part in ('train', 'test'):
            extract(part)
