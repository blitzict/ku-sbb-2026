# CPU port; original experiments/weight_space/white_walk.py, SHA256 fdb2c5aacebe37df7a78d6e312401f5cadbfd5162a0d0b2a250ba8186a7eace7
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Likelihood of a permanent jump in a walk after marginalizing a brief pulse."""
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
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp
from threadpoolctl import threadpool_limits
from white_jump_priors import SELECT, MASS
from partition_evidence import ROOT, OUT
import white_loop as wl
LOGS = np.linspace(-1.25, 1.25, 57)
SIG = np.exp(LOGS)
X = np.r_[np.linspace(0, 5, 241), np.geomspace(5.1, 120, 70)]
TIME = np.arange(-19, 20)
C, W = np.meshgrid(np.arange(-8, 9, 2), [0.8, 1.25, 1.8, 2.6, 3.8, 5.5, 8.0])
C, W = (C.ravel(), W.ravel())

def bumps(t):
    return np.exp(-0.5 * ((t[None, :] - C[:, None]) / W[:, None]) ** 2)
B = bumps(TIME) - bumps(TIME - 1)
PRIOR = -0.5 * (C / 4.5) ** 2
PRIOR -= logsumexp(PRIOR)
NODES, GH = np.polynomial.hermite.hermgauss(5)
GH = np.log(GH / np.sqrt(np.pi))
JUMP = np.array([0.2, 0.45, 0.8, 1.25, 1.8, 2.8, 4.5, 8.0])

def logpdf(z, law):
    tag, k, r = law
    norm = lambda x: -0.5 * x * x - 0.5 * np.log(2 * np.pi)
    if tag == 'normal':
        return norm(z)
    if tag == 'student':
        return gammaln((k + 1) / 2) - gammaln(k / 2) - 0.5 * np.log(np.pi * (k - 2)) - 0.5 * (k + 1) * np.log1p(z * z / (k - 2))
    if tag == 'contam':
        s = 1 / np.sqrt(1 - k + k * r * r)
        return np.logaddexp(np.log1p(-k) + norm(z / s) - np.log(s), np.log(k) + norm(z / (s * r)) - np.log(s * r))
    s = np.sqrt(1 - k * k)
    return np.logaddexp(norm((z - k) / s), norm((z + k) / s)) - np.log(2 * s)

def histogram(x):
    v = abs(x)
    j = (np.searchsorted(X, v, side='right') - 1).clip(0, len(X) - 2)
    f = ((v - X[j]) / (X[j + 1] - X[j])).clip(0, 1)
    return np.bincount(j, 1 - f, minlength=len(X)) + np.bincount(j + 1, f, minlength=len(X))

def jump_profile(obs, sigma, law, raw_sd):
    z = obs / sigma
    on = TIME != 0
    h = B[:, on]
    sd = np.clip(1.4 * raw_sd / sigma, 1.5, 48.0)
    var = 1 / ((h * h).sum(1) + sd ** (-2))
    mu = var * (h @ z[on])
    amps = mu[:, None] + np.sqrt(2 * var[:, None]) * NODES
    resid = z[None, None, on] - amps[:, :, None] * h[:, None, :]
    ll = logpdf(resid, law).sum(2) - logpdf(z[on], law).sum()
    ratio = -0.5 * (amps / sd) ** 2 - np.log(sd) + 0.5 * (amps - mu[:, None]) ** 2 / var[:, None] + 0.5 * np.log(var[:, None])
    logw = np.r_[np.log(0.87), (np.log(0.13) + ll + ratio + GH + PRIOR[:, None]).ravel()]
    logw -= logsumexp(logw)
    central = z[TIME == 0][0] - np.r_[0.0, (amps * B[:, TIME == 0]).ravel()]
    null = logsumexp(logw + logpdf(central, law)) - np.log(sigma)
    free = logsumexp(logw[:, None] + np.logaddexp(logpdf(central[:, None] - JUMP, law), logpdf(central[:, None] + JUMP, law)) - np.log(2), axis=0) - np.log(sigma)
    return (null, free, 1 - np.exp(logw[0]))

def one(raw, clean, j, bank, ls):
    d = np.diff(raw)
    e = np.diff(clean)
    b = j - 1
    left, right = (e[:b - 19], e[b + 20:])
    a = (histogram(left) @ bank).reshape(len(SELECT), len(SIG))
    c = (histogram(right) @ bank).reshape(len(SELECT), len(SIG))
    a -= a.max()
    c -= c.max()
    pl = np.log(MASS)
    joint = a + c + pl[:, None] + ls
    h0 = logsumexp(joint)
    post = np.exp(joint - h0)
    laww = post.sum(1)
    scale = post @ SIG / laww.clip(1e-15)
    pa, pb = (logsumexp(h + ls, axis=1) for h in (a, c))
    frompre = np.exp(pa + pl - logsumexp(pa + pl))
    frompost = np.exp(pb + pl - logsumexp(pb + pl))
    hscale = logsumexp(pa + pb + pl) - h0
    hshape = logsumexp(logsumexp(a + pl[:, None], 0) + logsumexp(c + pl[:, None], 0) + ls) - h0
    hind = logsumexp(pa + pl) + logsumexp(pb + pl) - h0
    f = {'scale': hscale, 'shape': hshape, 'both': hind, 'drift_pre': left.mean() / max(left.std(), 1e-05) * np.sqrt(len(left)), 'drift_post': right.mean() / max(right.std(), 1e-05) * np.sqrt(len(right)), 'start': raw[0], 'span': raw.std(), 'return': (raw[j - 1] - raw[0]) / (np.sqrt(j) * max(post.sum(0) @ SIG, 0.05))}
    for i, p in enumerate(laww):
        f['law' + str(i)] = p
        f['d_law' + str(i)] = abs(frompre[i] - frompost[i])
    all_null, all_free, all_pulses = ([], [], [])
    active = np.argsort(laww)[-4:]
    for i in active:
        z, alts, pulse = jump_profile(d[b + TIME], max(scale[i], 0.1), SELECT[i], raw.std())
        all_null.append(z + np.log(max(laww[i], 1e-25)))
        all_free.append(alts + np.log(max(laww[i], 1e-25)))
        all_pulses.append(laww[i] * pulse)
    z0 = logsumexp(all_null)
    alts = logsumexp(all_free, axis=0) - z0
    f.update({f'step{i}': v for i, v in enumerate(alts)})
    f['null_predictive'] = z0
    f['pulse'] = sum(all_pulses)
    f['step'] = logsumexp(alts + np.log([0.12, 0.18, 0.2, 0.2, 0.14, 0.09, 0.05, 0.02]))
    f['total'] = logsumexp(np.array([0.0, hscale, hshape, hind, f['step']]) + np.log([0.12, 0.28, 0.18, 0.08, 0.34]))
    return f

def extract(part):
    dst = OUT / f'walk_jump_{part}.parquet'
    if dst.exists():
        return
    with np.load(ROOT / f'tune_out/seq_cache_{part}.npz', allow_pickle=True) as q:
        ids, x, n, b = (q[k] for k in ('ids', 'x', 'n', 'b'))
    with np.load(OUT / f'pulse_streams_{part}.npz') as q:
        assert np.array_equal(q['ids'], ids)
        clean = q['common']
    walk = pd.read_parquet(OUT / f'innov_v2_{part}.parquet').loc[ids, 'pre_walk'].to_numpy()
    bank = np.stack([logpdf(X[:, None] / SIG, law) - LOGS[None, :] for law in SELECT], axis=1).reshape(len(X), -1)
    ls = np.log(0.91 / (1 + np.exp((abs(LOGS) - 0.7) / 0.045)) + 0.09 * np.exp(-abs(LOGS)))
    ls -= logsumexp(ls)
    frames = []
    for i, p in enumerate(walk):
        if p > 0.025:
            r = one(x[i, :n[i]].astype(float), clean[i, :n[i]].astype(float), b[i], bank, ls)
        else:
            r = {}
        frames.append(r)
    df = pd.DataFrame(frames, index=pd.Index(ids, name='id')).fillna(0)
    assert np.isfinite(df).all().all()
    df.to_parquet(dst)
    print(part, 'physical walks', np.sum(walk > 0.025), flush=True)

def evaluate():
    q, ids, tids, y, wt, wn, miss = wl.load()
    pair = [pd.read_parquet(OUT / f'walk_jump_{p}.parquet').loc[ix] for p, ix in (('train', ids), ('test', tids))]
    a = pd.concat(pair)
    gate = np.concatenate([pd.read_parquet(OUT / f'innov_v2_{p}.parquet').loc[ix, 'pre_walk'].to_numpy() for p, ix in (('train', ids), ('test', tids))])
    H = np.arcsinh(a.clip(-25, 60).to_numpy())

    def design(ix, z):
        d = (a.total.to_numpy()[ix] - z).clip(-7, 9)
        return gate[ix, None] * np.column_stack([np.ones(len(ix)), np.tanh(z / 2), H[ix], d, d * np.tanh(z / 2)])
    wl.run('walk_jump', design, 75.0, dependencies=[Path(__file__).resolve(), *[OUT / f'walk_jump_{p}.parquet' for p in ('train', 'test')]])
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        for p in ('train', 'test'):
            extract(p)
        evaluate()
