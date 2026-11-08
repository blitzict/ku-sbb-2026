# CPU port; original experiments/weight_space/white_tail_edges.py, SHA256 ee227ff7de5ec8dcb4667b661f87fa897232c773894f2a9e64094ba34d15037e
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Test whether the support of a noise distribution shifts at the marked boundary."""
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
from scipy.special import logsumexp, betaln
from threadpoolctl import threadpool_limits
from partition_evidence import OUT
Q, W = np.polynomial.legendre.leggauss(38)
GAP = np.exp(-3.5 + 8.5 * Q)
LOGW = np.log(8.5 * W * GAP)
K = np.array([0.72, 1.0, 1.5, 2.3, 3.8, 6.5])
XIS = (0.0, 0.18, 0.38)
LS = -0.85 + 3.9 * Q
S = np.exp(LS)
LW = np.log(3.9 * W) - 0.5 * ((LS + 0.7) / 0.85) ** 2 - np.log(0.85 * np.sqrt(2 * np.pi))
PR = np.log([0.018, 0.052, 0.055, 0.075, 0.09, 0.08, 0.42, 0.15, 0.06])

def law(x):
    n = len(x)
    if n == 0:
        return np.zeros(len(PR))
    bound = x.max() + GAP
    near = np.log1p(-np.minimum(x[:, None] / bound, 1 - 1e-14)).sum(0)
    ll = n * (np.log(K[:, None]) - np.log(bound)) + (K[:, None] - 1) * near
    prior = -0.5 * ((np.log(bound / K[:, None]) + 0.7) / 0.85) ** 2 - np.log(bound * 0.85 * np.sqrt(2 * np.pi))
    finite = logsumexp(ll + prior + LOGW, axis=1)
    open_ = []
    for xi in XIS:
        logdensity = -x.sum() / S if xi == 0 else -(1 / xi + 1) * np.log1p(xi * x[:, None] / S).sum(0)
        open_.append(logsumexp(logdensity - n * LS + LW))
    return np.r_[finite, open_]

def chart(a, b):
    pool = np.r_[a, b]
    cut = np.quantile(pool, 0.77)

    def above(x):
        return x[x > cut] - cut
    x, y = (above(a), above(b))
    A, B, J = (law(x), law(y), law(np.r_[x, y]))
    first = law(above(a[:len(a) // 2]))
    total = lambda h: logsumexp(h + PR)

    def bounded(h):
        return logsumexp((h + PR)[:len(K)]) - logsumexp((h + PR)[len(K):])
    null = total(J)
    both = total(A) + total(B) - null
    move = total(A + B) - null
    probs = lambda h: np.exp(h + PR - total(h))
    edges = (probs(J)[:4].sum(), probs(A)[:4].sum(), probs(B)[:4].sum())
    order = np.argsort(pool, kind='stable')[::-1]
    k = np.unique(np.minimum([2, 3, 5, 8, 13, 21, 34], len(pool) // 6))
    h = (order < len(a)).cumsum()[k - 1]
    base = len(a) / len(pool)
    sd = np.sqrt(k * base * (1 - base) * (len(pool) - k) / (len(pool) - 1))
    signed = (h - k * base) / sd.clip(0.01)
    run = betaln(h + 1, k - h + 1) - h * np.log(base) - (k - h) * np.log1p(-base)
    return [*edges, move, both, both - move, logsumexp((A + B + PR)[:len(K)]) - logsumexp((J + PR)[:len(K)]), logsumexp((A + B + PR)[len(K):]) - logsumexp((J + PR)[len(K):]), bounded(A) - bounded(first), logsumexp(run) - np.log(len(k)), signed[np.argmax(abs(signed))]]

def norm(v):
    q = np.quantile(v, [0.25, 0.5, 0.75])
    return (v - q[1]) / max((q[2] - q[0]) / 1.35, 0.16 * v.std(), 0.01)

def extract(part):
    dst = OUT / f'tail_edges_{part}.parquet'
    if dst.exists():
        return
    with np.load(OUT / f'pulse_streams_{part}.npz') as a:
        ids, x, n, b = (a[k] for k in ('ids', 'separate', 'n', 'b'))
    rows = []
    for v, end, j in zip(x, n, b):
        v = v[:end].astype(float)
        a, both = (v[:j], v[j:])
        z = norm(v)
        raw = []
        for l, r in ((z[:j], z[j:]), (norm(a), norm(both))):
            for sign in (1, -1):
                raw.extend(chart(sign * l, sign * r))
        rows.append(raw)
    keys = 'pooled_bounded pre_bounded post_bounded change_endpoint change_either extra_shape bounded_move open_move pre_replication tail_run tail_direction'.split()
    cols = [f'{g}_{s}_{k}' for g in ('pooled', 'normalized') for s in ('up', 'down') for k in keys]
    tab = pd.DataFrame(np.asarray(rows, np.float32).clip(-100, 150), index=pd.Index(ids, name='id'), columns=cols)
    assert np.isfinite(tab).all().all()
    tab.to_parquet(dst)
    print('tail edges', part, tab.shape, flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        for p in ('train', 'test'):
            extract(p)
