# CPU port; original experiments/weight_space/white_value_bands.py, SHA256 fdadfcd93e6d01cc1cb4f4ace0dd84e8c2e11efdaf940e427b5142ff37955378
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Test whether a specific interval of observation values changes its probability."""
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
from scipy.special import betaln, logsumexp
from threadpoolctl import threadpool_limits
from partition_evidence import OUT
Q = np.array([0.035, 0.075, 0.13, 0.2, 0.29, 0.39, 0.5, 0.61, 0.71, 0.8, 0.87, 0.925, 0.965])
I, J = np.triu_indices(len(Q), k=1)

def cuts(values, adapt):
    v = np.sort(values)
    k = np.rint((len(v) - 1) * Q).astype(int).clip(1, len(v) - 2)
    if adapt:
        d = np.diff(v)
        mid = (k[1:] + k[:-1]) // 2
        lo, hi = (np.r_[max(1, k[0] // 2), mid + 1], np.r_[mid + 1, (k[-1] + len(v)) // 2])
        n = len(d)
        at = np.arange(n)
        local = (v[np.minimum(at + 8, n)] - v[np.maximum(at - 7, 0)]) / (np.minimum(at + 8, n) - np.maximum(at - 7, 0))
        rel = d / (local + 1e-07)
        k = np.array([a + np.argmax(rel[a:b]) for a, b in zip(lo, hi)])
    return (v[k] + v[k + 1]) * 0.5

def count(v, edges):
    k = np.searchsorted(np.sort(v), edges, side='right')
    return k[J] - k[I]

def log_ratio(a, b, n, m, p, tau):
    aa = p * tau
    bb = (1 - p) * tau
    return betaln(aa + a, bb + n - a) + betaln(aa + b, bb + m - b) - betaln(aa + a + b, bb + n + m - a - b) - betaln(aa, bb)

def one(train, test, adapt):
    edges = cuts(np.r_[train[0], train[1]], adapt)
    a, b = [count(v, edges) for v in train]
    c, d = [count(v, edges) for v in test]
    n, m = map(len, train)
    p = (a + b + 1) / (n + m + 2)
    fit = np.stack([log_ratio(a, b, n, m, p, tau) for tau in (20.0, 95.0)])
    total = np.stack([log_ratio(a + c, b + d, n + len(test[0]), m + len(test[1]), p, tau) for tau in (20.0, 95.0)])
    width = Q[J] - Q[I]

    def pool(mask):
        return logsumexp(total[:, mask]) - logsumexp(fit[:, mask])
    j, k = np.unravel_index(np.argmax(fit), fit.shape)
    held = total[j, k] - fit[j, k]
    effect = (d[k] / len(test[1]) - c[k] / len(test[0])) * np.sign(b[k] / m - a[k] / n)
    return np.array([pool(width > 0), pool(width < 0.2), pool((Q[I] >= 0.2) & (Q[J] <= 0.8)), pool((Q[I] < 0.14) | (Q[J] > 0.86)), held, effect, (Q[I[k]] + Q[J[k]]) / 2, width[k]])

def row(x, b):
    halves = (x[:b], x[b:])
    mask = [np.arange(len(v)) * 8 // len(v) % 2 for v in halves]
    out = []
    for scaled in (False, True):
        for adapt in (False, True):
            directions = []
            for f in (0, 1):
                tr, te = ([], [])
                for h, block in zip(halves, mask):
                    a, v = (h[block == f], h[block != f])
                    if scaled:
                        loc = np.median(a)
                        sd = max(1.4826 * np.median(abs(a - loc)), 0.2 * a.std(), 0.001)
                        a, v = ((a - loc) / sd, (v - loc) / sd)
                    tr.append(a)
                    te.append(v)
                directions.append(one(tr, te, adapt))
            l, r = directions
            out.extend([*np.minimum(l[:6], r[:6]), *np.mean([l[:6], r[:6]], 0), abs(l[6] - r[6]), abs(l[7] - r[7])])
    return out

def extract(part):
    path = OUT / f'value_bands_{part}.parquet'
    if path.exists():
        return
    with np.load(OUT / f'state_streams_{part}.npz') as a:
        ids, x, n, b = (a[k] for k in ('ids', 'separate', 'n', 'b'))
    cols = [f'{v}_{partition}_{s}' for v in ('raw', 'shape') for partition in ('quantile', 'valley') for s in [*[f'{fn}_{key}' for fn in ('replicated', 'mean') for key in ('any', 'narrow', 'center', 'shoulders', 'best', 'direction')], 'center_disagreement', 'width_disagreement']]
    data = np.stack([row(v[:e].astype(float), j) for v, e, j in zip(x, n, b)])
    frame = pd.DataFrame(data, index=pd.Index(ids, name='id'), columns=cols).clip(-70, 120).astype(np.float32)
    assert frame.index.is_unique and np.isfinite(frame).all().all()
    frame.to_parquet(path)
    print(part, frame.shape, flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        for part in ('train', 'test'):
            extract(part)
