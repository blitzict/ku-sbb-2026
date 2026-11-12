# CPU port; original experiments/weight_space/white_shock_spacing.py, SHA256 e163d9e8f6027ef90f1562ba657b53d69b1ae52e682db1dd14021b8290ae5b87
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Compare the spacing of extremes at equal event rates on opposite sides of a break."""
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
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from partition_evidence import OUT
SLOPE = np.array([-1.55, -1.0, -0.63, -0.35, -0.16, 0.0, 0.16, 0.35, 0.63, 1.0, 1.55])
PRIOR = np.exp(-abs(SLOPE) / 0.53)
PRIOR /= PRIOR.sum()
AGES = np.arange(300)
RATES = (0.1, 0.24)

def grid(rate):
    t = np.log((AGES + 0.6) * rate + 0.13)
    lo = np.full(len(SLOPE), -12.0)
    hi = -lo
    for _ in range(35):
        c = (lo + hi) / 2
        logit = c[:, None] + SLOPE[:, None] * t
        survive = np.exp(np.cumsum(-np.logaddexp(0.0, logit), axis=1))
        small = 1 + survive.sum(1) < 1 / rate
        hi = np.where(small, c, hi)
        lo = np.where(small, lo, c)
    logit = c[:, None] + SLOPE[:, None] * t
    return np.stack([-np.logaddexp(0.0, logit), -np.logaddexp(0.0, -logit)], axis=1)
BANK = {r: grid(r) for r in RATES}

def count(events):
    ix = np.flatnonzero(events)
    if len(ix) < 2:
        return np.zeros((2, len(AGES)), dtype=int)
    pos = np.arange(ix[0] + 1, len(events))
    age = pos - ix[np.searchsorted(ix, pos, side='left') - 1] - 1
    return np.bincount(age.clip(0, len(AGES) - 1) + len(AGES) * events[pos], minlength=2 * len(AGES)).reshape(2, -1)

def contrasts(counts, rate):
    l = counts.reshape(4, -1) @ BANK[rate].reshape(len(SLOPE), -1).T
    l -= l[:, [len(SLOPE) // 2]]

    def evidence(x):
        return logsumexp(x + np.log(PRIOR))
    a, b, c, d = l
    pre = a + b
    post = c + d

    def slope(x):
        return np.exp(x + np.log(PRIOR) - evidence(x)) @ SLOPE
    joint = evidence(pre + post)
    before = evidence(pre)
    after = evidence(post)
    return [before, after, before + after - joint, after - joint, before - joint, slope(pre), slope(post), before - evidence(a), after - evidence(c)]

def extract(part):
    dst = OUT / f'shock_spacing_{part}.parquet'
    if dst.exists():
        return
    with np.load(OUT / f'hetero_path_{part}.npz') as a:
        ids, u, n, b, h = (a[k] for k in ('ids', 'u', 'n', 'b', 'common'))
    groups = ('magnitude', 'up', 'down', 'whitened_magnitude')
    metrics = ('pre', 'post', 'change', 'on', 'off', 'pre_shape', 'post_shape', 'pre_replication', 'post_replication')
    columns = [f'{g}_{int(100 * r)}_{m}' for g in groups for r in RATES for m in metrics]
    records = []
    for row, scale, end, j in zip(u, h, n, b):
        sides = [row[:j], row[j:end]]
        centered = [x - np.median(x) for x in sides]
        z = np.concatenate(centered)
        values = (abs(z), z, -z, abs(z) / scale[:end].clip(0.01))
        result = []
        for v in values:
            left, right = (v[:j], v[j:])
            for rate in RATES:
                ev = [side > np.quantile(side, 1 - rate) for side in (left, right)]
                chunks = [half for s in ev for half in np.array_split(s, 2)]
                result.extend(contrasts(np.stack([count(d) for d in chunks]), rate))
        records.append(result)
    tab = pd.DataFrame(np.asarray(records, dtype=np.float32), index=ids, columns=columns)
    tab.index.name = 'id'
    assert np.isfinite(tab).all().all()
    tab.to_parquet(dst)
    print('shock spacing', part, tab.shape, flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        for part in ('train', 'test'):
            extract(part)
