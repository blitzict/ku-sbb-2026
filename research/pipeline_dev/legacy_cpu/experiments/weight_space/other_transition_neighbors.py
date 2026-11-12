# CPU port; original experiments/weight_space/other_transition_neighbors.py, SHA256 fbcfebb8fb5384443b945c16fe176da9950886ec055fc5ffb5329ed75bce6228
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Find unbinned changes in observed transitions using neighbors in separate time blocks.

Related test: Schilling, JASA (1986), doi:10.1080/01621459.1986.10478337.
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
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits
from partition_evidence import OUT

def normalized(v):
    m = np.median(v)
    s = max(1.4826 * np.median(abs(v - m)), 0.2 * v.std(), 0.001)
    return np.arcsinh((v - m) / s)

def score(v, side, group, t):
    rows = []
    periods = []
    for parity in (0, 1):
        ref = np.flatnonzero(group % 2 != parity)
        query = np.flatnonzero(group % 2 == parity)
        tree = cKDTree(v[ref])
        _, near = tree.query(v[query], k=min(51, len(ref)), workers=1)
        candidates = ref[near]
        allowed = abs(t[candidates] - t[query, None]) > 12
        loc = np.argsort(~allowed, axis=1, kind='stable')[:, :24]
        assert np.all(np.take_along_axis(allowed, loc, axis=1))
        nb = side[np.take_along_axis(candidates, loc, axis=1)]
        counts = []
        for j in (0, 1):
            times = t[ref[side[ref] == j]]
            excluded = np.searchsorted(times, t[query] + 12, 'right') - np.searchsorted(times, t[query] - 12, 'left')
            counts.append(len(times) - excluded)
        p = counts[1] / (counts[0] + counts[1])
        sq = side[query]
        effective = np.sqrt(np.sum(sq) * np.sum(~sq) / len(sq))
        out = []
        for k in (4, 12, 24):
            ratio = (nb[:, :k].sum(1) + 2 * p) / (k + 2)
            evidence = (ratio - p) / np.sqrt(p * (1 - p)).clip(0.1)
            l, r = (evidence[~sq], evidence[sq])
            out.extend([effective * (r.mean() - l.mean()), effective * (np.mean(r > 0.25) - np.mean(l > 0.25)), effective * (np.mean(l < -0.25) - np.mean(r < -0.25))])
        rows.append(out)
    r = np.array(rows)
    return np.r_[r.mean(0), r.min(0)]

def one(row, end, cut):
    raw = row[:end].astype(float)
    t = np.arange(end)
    side = t >= cut
    block = np.where(side, 4 + (t - cut) * 4 // (end - cut), t * 4 // cut)
    good = t[4:][block[4:] == block[:-4]]
    assert len(good) > 100
    output = []
    for z in (normalized(raw), np.r_[normalized(raw[:cut]), normalized(raw[cut:])]):
        vectors = np.column_stack([z[good], z[good - 1], 0.72 * z[good - 4]])
        output.extend(score(vectors, side[good], block[good], good))
    return output

def run(part):
    frame = []
    for name in ('pulse', 'state'):
        with np.load(OUT / f'{name}_streams_{part}.npz') as a:
            ids, x, n, b = [a[k] for k in ('ids', 'separate', 'n', 'b')]
        cols = [f'{name}_{space}_{rep}_k{k}_{term}' for space in ('level', 'shape') for rep in ('mean', 'held') for k in (4, 12, 24) for term in ('separation', 'new_neighborhood', 'lost_neighborhood')]
        arr = np.array([one(v, end, cut) for v, end, cut in zip(x, n, b)], np.float32)
        assert np.isfinite(arr).all()
        frame.append(pd.DataFrame(arr, index=ids, columns=cols))
        print('transition neighborhoods', part, name, arr.shape, flush=True)
    pd.concat(frame, axis=1).to_parquet(OUT / f'transition_neighbors_{part}.parquet')
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        for part in ('train', 'test'):
            run(part)
