# CPU port; original experiments/weight_space/white_phase_distribution.py, SHA256 0620b8267374fe57052ac29f061b8833b1ca503d2c2698c35f9b78858435f739
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Test whether a repeating phase predicts the distribution of genuinely held-out cycles."""
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
from scipy.special import gammaln, logsumexp
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
from partition_evidence import OUT
PER = np.r_[6, 8, 10, np.arange(12, 33), 36, 40, 48, 56, 64]
NP = 8
NQ = 6
T = np.arange(1025)

def histogram(x, e, j):
    n = len(x)
    t = np.arange(n) - j
    phase = t[None, :] % PER[:, None] * NP // PER[:, None]
    fold = t[None, :] // PER[:, None] % 2
    part = (t >= 0).astype(int)
    base = (((np.arange(len(PER))[:, None] * 2 + part) * 2 + fold) * NP + phase) * NQ

    def ranks(v):
        return np.minimum((rankdata(v) - 0.5) * NQ / len(v), NQ - 1).astype(int)

    def scans(v):
        cat = np.r_[ranks(v[:j]), ranks(v[j:])]
        return np.bincount((base + cat[None, :]).ravel(), minlength=len(PER) * 4 * NP * NQ).reshape(len(PER), 2, 2, NP, NQ)
    med = np.r_[np.full(j, np.median(e[:j])), np.full(n - j, np.median(e[j:]))]
    z = e - med
    lag = z * np.r_[z[0], z[:-1]]
    return np.stack([scans(v) for v in (x, e, abs(z), lag)]).astype(np.int16)

def evidence(h, alpha):
    G = gammaln(T + alpha) - gammaln(alpha)
    V = gammaln(T + alpha / NQ) - gammaln(alpha / NQ)
    phase = V[h].sum((-2, -1)) - G[h.sum(-1)].sum(-1)
    marginal = V[h.sum(-2)].sum(-1) - G[h.sum((-2, -1))]
    return phase - marginal

def features(h):
    a, c = (h[:, :, :, 0], h[:, :, :, 1])
    A = a.sum(3)
    C = c.sum(3)
    result = {}
    for concentration in (12.0, 64.0):
        ev = lambda v: evidence(v, concentration)
        x, y = (ev(A), ev(C))
        common = ev(A + C)
        prior = -np.log(len(PER))
        null = np.logaddexp(np.log(0.8), np.log(0.2) + logsumexp(common + prior, 2))

        def integrate(v):
            return logsumexp(v + prior, 2)
        turns = np.stack([ev(A + np.roll(C, s, axis=-2)) for s in range(1, NP)], 3)
        rotation = logsumexp(turns, 3) - np.log(NP - 1)
        pre_held = x - ev(a[:, :, :, 0])
        post_held = y - ev(c[:, :, :, 0])
        left = np.argmax(ev(a[:, :, :, 0]), axis=2, keepdims=True)
        right = np.argmax(ev(c[:, :, :, 0]), axis=2, keepdims=True)
        get = lambda v, i: np.take_along_axis(v, i, axis=2)[:, :, 0]
        pre_post = get(y, left)
        post_pre = get(x, right)
        vals = dict(common=null, rotate=integrate(rotation) - null, reshape=integrate(x + y) - null, newclock=integrate(x) + integrate(y) - null, on=integrate(y) - null, off=integrate(x) - null, pre_repeats=get(pre_held, left), post_repeats=get(post_held, right), second_half_same_clock=pre_post, first_half_same_clock=post_pre)
        for tag, v in vals.items():
            for i, who in enumerate(('level', 'innovation', 'amplitude', 'memory')):
                result[f'{who}_{tag}_{int(concentration)}'] = v[:, i]
    return result

def extract(part):
    path = OUT / f'phase_distribution_{part}.parquet'
    if path.exists():
        return
    with np.load(OUT / f'pulse_streams_{part}.npz') as a:
        ids, x, n, b = (a[k] for k in ('ids', 'separate', 'n', 'b'))
    with np.load(OUT / f'state_streams_{part}.npz') as a:
        assert np.array_equal(a['ids'], ids)
        e = a['separate']
    frames = []
    for start in range(0, len(x), 36):
        sl = slice(start, start + 36)
        hs = np.stack([histogram(r[:end], v[:end], j) for r, v, end, j in zip(x[sl], e[sl], n[sl], b[sl])])
        rec = features(hs)
        frames.append(pd.DataFrame(rec, index=ids[sl]))
    f = pd.concat(frames).clip(-65, 110).astype(np.float32)
    f.index.name = 'id'
    assert np.isfinite(f).all().all()
    f.to_parquet(path)
    print(part, 'held-out cycle distributions', f.shape, flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        for part in ('train', 'test'):
            extract(part)
