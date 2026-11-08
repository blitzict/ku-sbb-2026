# CPU port; original experiments/weight_space/white_quantile_alignment.py, SHA256 8bb86a39ea34037c2f1c5c5f492ef158a05d149d3ad6bd6043ef7c4fbf924d36
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Check whether aligning selected parts of two observed distributions predicts withheld points."""
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
from scipy.ndimage import gaussian_filter1d
from threadpoolctl import threadpool_limits
from partition_evidence import OUT
GRID = np.linspace(-9, 9, 161)
DX = GRID[1] - GRID[0]
P = np.array([0.07, 0.2, 0.35, 0.5, 0.65, 0.8, 0.93])

def density(training, query):
    ix = np.rint((training - GRID[0]) / DX).astype(int)
    good = (ix >= 0) & (ix < len(GRID))
    counts = np.bincount(ix[good], minlength=len(GRID)).astype(float)
    sigma = np.clip(0.68 * len(training) ** (-0.2), 0.16, 0.36)
    observed = gaussian_filter1d(counts, sigma / DX, mode='constant') / DX
    prior = 0.375 * (1 + query ** 2 / 4) ** (-2.5)
    return np.log((np.interp(query, GRID, observed, left=0.0, right=0.0) + 14 * prior) / (len(training) + 14))

def transform(x, source, dest):
    slope = np.diff(dest) / np.diff(source).clip(0.012)
    slope = slope.clip(0.3, 3.3)
    knots = np.r_[0.0, np.cumsum(slope * np.diff(source))]
    knots += dest[len(dest) // 2] - knots[len(dest) // 2]
    j = (np.searchsorted(source, x, side='right') - 1).clip(0, len(slope) - 1)
    return (knots[j] + (x - source[j]) * slope[j], np.log(slope[j]))

def scores(training, query):
    qs = np.stack([np.quantile(x, P) for x in training])
    n = np.array([len(x) for x in training])
    dest = np.average(qs, axis=0, weights=n)
    med = qs[:, 3]
    width = (qs[:, 5] - qs[:, 1]).clip(0.12)
    center = dest[3]
    spread = np.average(width, weights=n)
    shared = np.concatenate(training)
    ll0 = [density(shared, q) for q in query]
    alternatives = {}
    for name in ('location', 'width', 'both', 'shoulders', 'tails', 'all_knots'):
        z, v, jac = ([], [], [])
        for k, (t, q) in enumerate(zip(training, query)):
            shrink = n[k] / (n[k] + 24.0)
            if name in ('location', 'width', 'both'):
                origin = center + shrink * (med[k] - center) if name in ('location', 'both') else center
                mult = (spread / width[k]) ** shrink if name in ('width', 'both') else 1.0
                z.append((t - origin) * mult + center)
                v.append((q - origin) * mult + center)
                jac.append(np.log(mult))
            else:
                picks = {'shoulders': [1, 3, 5], 'tails': [0, 1, 3, 5, 6], 'all_knots': list(range(7))}[name]
                source = dest[picks] + shrink * (qs[k, picks] - dest[picks])
                z.append(transform(t, source, dest[picks])[0])
                u, j = transform(q, source, dest[picks])
                v.append(u)
                jac.append(j)
        common = np.concatenate(z)
        alternatives[name] = np.array([np.sum(density(common, q) + j - base) for q, j, base in zip(v, jac, ll0)])
    return alternatives

def one(x, cut):
    a, b = (x[:cut], x[cut:])
    loc = np.median(x)
    scale = max(np.quantile(abs(np.r_[a - np.median(a), b - np.median(b)]), 0.68), 0.18 * x.std(), 0.001)
    sides = ((a - loc) / scale, (b - loc) / scale)
    blocks = [np.arange(len(v)) * 8 // len(v) % 2 for v in sides]
    folds = []
    for train in (0, 1):
        t = [v[idx == train] for v, idx in zip(sides, blocks)]
        q = [v[idx != train] for v, idx in zip(sides, blocks)]
        folds.append(scores(t, q))
    result = {}
    for name in folds[0]:
        vals = np.stack([f[name] for f in folds])
        result[name + '_gain'] = vals.sum()
        result[name + '_replicate'] = vals.sum(1).min()
        result[name + '_both_sides'] = vals.sum(0).min()
        if name in ('shoulders', 'tails', 'all_knots'):
            d = vals - np.stack([f['both'] for f in folds])
            result[name + '_after_location_scale'] = d.sum()
    return result

def extract(part):
    path = OUT / f'quantile_alignment_{part}.parquet'
    if path.exists():
        return
    tables = []
    for channel in ('pulse', 'state'):
        with np.load(OUT / f'{channel}_streams_{part}.npz') as a:
            ids, x, n, b = (a[k] for k in ('ids', 'separate', 'n', 'b'))
        vals = [one(v[:end].astype(float), cut) for v, end, cut in zip(x, n, b)]
        tables.append(pd.DataFrame(vals, index=pd.Index(ids, name='id')).add_prefix(channel + '_'))
    assert tables[0].index.equals(tables[1].index)
    frame = pd.concat(tables, axis=1).clip(-80, 150)
    assert np.isfinite(frame).all().all()
    frame.astype(np.float32).to_parquet(path)
    print(part, frame.shape, flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        for part in ('train', 'test'):
            extract(part)
