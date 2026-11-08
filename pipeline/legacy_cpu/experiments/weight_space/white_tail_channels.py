# CPU port; original experiments/weight_space/white_tail_channels.py, SHA256 dd2b37b92d64b1515715307a4524061f75733c05d740127703684fc124e3086f
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Separate unusually large innovations from the ordinary motion in each real series."""
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
from scipy.special import expit, logsumexp
from threadpoolctl import threadpool_limits
from partition_evidence import OUT
FRAC = np.array([0.025, 0.055, 0.11, 0.2])
RATIO = np.array([2.0, 3.4, 5.6])
SCALE = np.array([0.7, 0.86, 1.0, 1.18, 1.4])

def measure(v, j):
    a, b = (v[:j], v[j:])
    center = np.r_[np.full(j, np.median(a)), np.full(len(b), np.median(b))]

    def width(x):
        return max(np.median(abs(x - np.median(x))) * 1.4826, 0.1 * x.std(), 0.01)
    scale = np.r_[np.full(j, width(a)), np.full(len(b), width(b))]
    e = (v - center) / scale
    index = np.arange(len(e))
    q = np.where(index < j, 2 * index // j, 2 + 2 * (index - j) // len(b))
    bins = np.searchsorted(AXIS, abs(e)).clip(1, len(AXIS) - 1)
    f = ((abs(e) - AXIS[bins - 1]) / (AXIS[bins] - AXIS[bins - 1])).clip(0, 1)
    cells = q * len(AXIS) + bins
    H = (np.bincount(cells - 1, 1 - f, minlength=4 * len(AXIS)) + np.bincount(cells, f, minlength=4 * len(AXIS))).reshape(4, -1)
    likelihood = H @ TABLE

    def posterior(s):
        l = s + PRIOR
        return np.exp(l - logsumexp(l))
    pre, post, common = (posterior(x) for x in (likelihood[:2].sum(0), likelihood[2:].sum(0), likelihood.sum(0)))
    wr = RESP @ common
    atten = SHRINK @ common
    w = wr[bins - 1] * (1 - f) + wr[bins] * f
    retain = atten[bins - 1] * (1 - f) + atten[bins] * f
    gate = 1 - common[:len(SCALE)].sum()
    core = center + (v - center) * retain
    tail = center + (v - center) * (1 - gate + gate * (0.16 + 1.5 * w))

    def bf(s):
        return logsumexp(s[len(SCALE):] + PRIOR[len(SCALE):]) - logsumexp(s[:len(SCALE)] + PRIOR[:len(SCALE)])
    l = likelihood[:2].sum(0)
    r = likelihood[2:].sum(0)
    delta = lambda z: (z[j:].mean() - z[:j].mean()) * np.sqrt(j * len(b) / len(v))
    test = [gate, 1 - pre[:len(SCALE)].sum(), 1 - post[:len(SCALE)].sum(), bf(l), bf(r), logsumexp(l + PRIOR) + logsumexp(r + PRIOR) - logsumexp(l + r + PRIOR), delta(w), delta(w * np.sign(e)), delta(e * retain), delta(abs(e) * retain), bf(l) - bf(likelihood[0]), bf(r) - bf(likelihood[2])]
    return (core, tail, np.clip(test, -100, 120))
AXIS = np.r_[np.linspace(0, 6, 121), np.geomspace(6.15, 40, 50)]
models = [(0.0, 1.0, s) for s in SCALE] + [(p, r, s) for p in FRAC for r in RATIO for s in SCALE]
tables = []
resp = []
retain = []
prior = []
for p, r, s in models:
    x = AXIS / s
    if p:
        l = np.logaddexp(np.log1p(-p) - 0.5 * x * x, np.log(p / r) - 0.5 * (x / r) ** 2) - np.log(s)
        w = expit(np.log(p / ((1 - p) * r)) + 0.5 * x * x * (1 - r ** (-2)))
        mass = 0.26 / (len(FRAC) * len(RATIO))
    else:
        l = -0.5 * x * x - np.log(s)
        w = np.zeros_like(x)
        mass = 0.74
    prior.append(np.log(mass) - 0.5 * (np.log(s) / 0.22) ** 2)
    tables.append(l)
    resp.append(w)
    retain.append(1 - w * (1 - r ** (-2)))
TABLE = np.array(tables).T.copy()
RESP = np.array(resp).T.copy()
SHRINK = np.array(retain).T.copy()
PRIOR = np.array(prior)
PRIOR -= logsumexp(PRIOR)

def extract(part):
    dst = OUT / f'tail_channels_{part}.parquet'
    cache = OUT / f'tail_channels_streams_{part}.npz'
    if dst.exists() and cache.exists():
        return
    with np.load(OUT / f'pulse_streams_{part}.npz') as a:
        ids, x, n, b = (a[k] for k in ('ids', 'separate', 'n', 'b'))
    core = np.zeros_like(x)
    tail = np.zeros_like(x)
    rows = []
    for i, (v, j, end) in enumerate(zip(x, b, n)):
        core[i, :end], tail[i, :end], s = measure(v[:end].astype(float), j)
        rows.append(s)
    keys = 'gate pre_gate post_gate pre_bf post_bf new_law rate_change direction_change core_location core_magnitude replicate_pre replicate_post'.split()
    tab = pd.DataFrame(np.asarray(rows, np.float32), index=pd.Index(ids, name='id'), columns=keys)
    assert np.isfinite(tab).all().all() and np.isfinite(core).all() and np.isfinite(tail).all()
    tab.to_parquet(dst)
    np.savez_compressed(cache, ids=ids, n=n, b=b, common=core, separate=tail)
    print('core and tail', part, tab.shape, 'strong', int((tab.gate > 0.8).sum()), flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        for p in ('train', 'test'):
            extract(p)
