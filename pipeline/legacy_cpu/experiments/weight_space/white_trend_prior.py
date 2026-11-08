# CPU port; original experiments/weight_space/white_trend_prior.py, SHA256 a456d9c9ed8464c7366a5b966fcfe1df4620286405c60feb671e5c32e6f4d1bf
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Marginalize bounded trend amplitudes and the clock before and after a break."""
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
from pathlib import Path
from threadpoolctl import threadpool_limits
from white_influence import influence, center, CORE
from white_jump_priors import SELECT, MASS
from white_walk import logpdf
from partition_evidence import OUT
import white_loop as wl

def sufficient(x, b, ar):
    n = len(x)
    u = np.arange(n) / (n - 1)
    s = (np.arange(n) >= b).astype(float)
    Q = np.stack([np.ones(n), u, u * u, s, s * u, s * u * u], 1)
    X = Q[2:] - ar[0] * Q[1:-1] - ar[1] * Q[:-2]
    y = x[2:] - ar[0] * x[1:-1] - ar[1] * x[:-2]
    ridge = np.diag([1e-07, 0.04, 0.04, 0.05, 0.07, 0.07])
    beta = np.linalg.solve(X.T @ X + ridge, X.T @ y)
    for _ in range(3):
        e = y - X @ beta
        m, sd = center(e)
        sd = max(sd / 0.67449, e.std() * 0.2)
        w = 1 / (1 + ((e - m) / (sd * 2.7)) ** 2)
        beta = np.linalg.solve(X.T @ (w[:, None] * X) + ridge, X.T @ (w * y))
    e = y - X @ beta
    sides = (s[2:] == 0, s[2:] == 1)
    c0, c1 = (center(e[m])[1] for m in sides)
    small = np.where(sides[0], c0, c1)
    ls = np.array([np.sum(logpdf(e / (small / v), law) - np.log(small / v)) for v, law in zip(CORE, SELECT)]) + np.log(MASS)
    pp = np.exp(ls - logsumexp(ls))
    idx = np.argsort(pp)[-3:]
    GG = np.zeros((6, 6))
    cc = np.zeros(6)
    for j in idx:
        sc = small / CORE[j]
        psi, _ = influence(e / sc, SELECT[j])
        info = np.where(sides[0], (psi[sides[0]] ** 2).mean(), (psi[sides[1]] ** 2).mean()).clip(0.25, 8)
        G = X.T @ ((info / sc ** 2)[:, None] * X)
        c = G @ beta + X.T @ (psi / sc)
        GG += pp[j] * G
        cc += pp[j] * c
    GG /= pp[idx].sum()
    cc /= pp[idx].sum()
    v = max((c0 + c1) / 1.34898, 0.0001)
    g0 = max(GG[0, 0], 1e-05)
    G = (GG[1:, 1:] - np.outer(GG[1:, 0], GG[0, 1:]) / g0) * v * v
    c = (cc[1:] - GG[1:, 0] * cc[0] / g0) * v
    return (G, c, b / (n - 1))
TREND = [(0.36, 0.0, 0.2), (1.05, 0.0, 0.25), (0.4, 1.65, 0.4), (0.55, 2.5, 0.15)]
CHANGE = [(0.2, 0.0, 0.23), (0.65, 0.0, 0.3), (0.18, 0.55, 0.21), (0.38, 1.35, 0.19), (0.72, 2.2, 0.07)]

def integrate(h, c, s, m):
    h = np.maximum(h, 0.0)
    den = 1 + h * s * s
    return -0.5 * np.log(den) + m * c - 0.5 * m * m * h + 0.5 * s * s * (c - m * h) ** 2 / den

def one(h, c, prior):
    vals = []
    for s, m, w in prior:
        a = integrate(h, c, s, m)
        if m:
            a = np.logaddexp(a, integrate(h, c, s, -m)) - np.log(2.0)
        vals.append(a + np.log(w))
    return logsumexp(vals, axis=0)

def evaluate_programs(G, c, b):
    N = len(c)
    U = np.broadcast_to(np.eye(5), (N, 5, 5))
    L, Q, S, SL, SQ = [U[:, i, :] for i in range(5)]
    a = b[:, None]
    shapes = [L, Q, 0.5 * (L + Q)]
    posts = [SL, SQ, 0.5 * (SL + SQ)]
    at = [a, a * a, 0.5 * (a + a * a)]

    def stat(v):
        return (np.einsum('ni,nij,nj->n', v, G, v).clip(0), np.einsum('ni,ni->n', v, c))

    def single(v, prior):
        return one(*stat(v), prior)

    def double(v, d, p=TREND, r=CHANGE):
        A, ca = stat(v)
        B, cb = stat(d)
        AB = np.einsum('ni,nij,nj->n', v, G, d)
        vals = []
        for s, m, w in p:
            den = A + 1 / s ** 2
            for mm in [m, -m] if m else [0.0]:
                val = integrate(A, ca, s, mm) + one(B - AB * AB / den, cb - AB * (ca + mm / s ** 2) / den, r)
                vals.append(val + np.log(w / (2 if m else 1)))
        return logsumexp(vals, axis=0)
    base = np.stack([np.zeros(N), *[single(v, TREND) for v in shapes], double(L, Q, TREND, TREND)], 1)
    lp = np.log([0.79, 0.085, 0.065, 0.045, 0.015])
    null = logsumexp(base + lp, axis=1)
    h = {'null' + str(k): v for k, v in enumerate(np.exp(base + lp - null[:, None]).T)}

    def marginal(name, rows, weights=None):
        w = np.full(len(rows), 1 / len(rows)) if weights is None else weights
        h[name] = logsumexp(np.stack(rows, axis=1) + np.log(w), axis=1) - null
    marginal('level', [single(S, CHANGE), *[double(v, S) for v in shapes]], [0.76, 0.1, 0.08, 0.06])
    for i, (sh, post, touch) in enumerate(zip(shapes, posts, at)):
        restart = SL - a * S if i == 0 else SQ - 2 * a * SL + a * a * S if i == 1 else 0.5 * (SL - a * S + SQ - 2 * a * SL + a * a * S)
        pre = sh - post
        continuous = pre + touch * S
        for name, v in (('stop', continuous), ('erase', pre), ('rewind', pre + restart), ('start', restart), ('activate', post)):
            h[f'{name}{i}'] = single(v, TREND) - null
        choices = [double(sh, post), double(sh, post - touch * S), double(sh, restart), double(continuous, post - touch * S, TREND, TREND), double(pre, post, TREND, TREND), double(pre, restart, TREND, TREND), double(continuous, restart, TREND, TREND)]
        for j, val in enumerate(choices):
            h[f'switch{i}_{j}'] = val - null
        for j, q in enumerate(posts):
            if j != i:
                h[f'reshape{i}_{j}'] = double(pre, q, TREND, TREND) - null
    h['aggregate'] = logsumexp(np.stack([v for k, v in h.items() if not k.startswith('null')], axis=1), axis=1) - np.log(len(h) - 5)
    return pd.DataFrame(h).clip(-60, 90)

def extract(part):
    path = OUT / f'trend_population_{part}.parquet'
    if path.exists():
        return
    with np.load(OUT / f'pulse_streams_{part}.npz') as a:
        ids, x, n, b = (a[k] for k in ('ids', 'common', 'n', 'b'))
    st = pd.read_parquet(OUT / f'state_{part}.parquet').loc[ids]
    ar = st[['ar1', 'ar2']].to_numpy()
    stats = [sufficient(v[:e].astype(float), j, p) for v, e, j, p in zip(x, n, b, ar)]
    g, c, t = (np.asarray(vals) for vals in zip(*stats))
    h = evaluate_programs(g, c, t)
    h.index = pd.Index(ids, name='id')
    assert np.isfinite(h).all().all()
    h.astype(np.float32).to_parquet(path)
    print(part, 'clock priors', h.shape, flush=True)

def score():
    _, ids, tids, *_ = wl.load()
    a = pd.concat([pd.read_parquet(OUT / f'trend_population_{p}.parquet').loc[ix] for p, ix in (('train', ids), ('test', tids))])
    g = np.r_[*[pd.read_parquet(OUT / f'curve_{p}.parquet').loc[ix, 'gate'].to_numpy() for p, ix in (('train', ids), ('test', tids))],]
    f = np.arcsinh(a.clip(-18, 45).to_numpy())

    def design(ix, z):
        d = (a['aggregate'].to_numpy()[ix] - z).clip(-7, 8)
        return (0.12 + 0.88 * g[ix, None]) * np.column_stack([f[ix], d, d * np.tanh(z / 2)])
    wl.run('trend_population', design, 100.0, dependencies=[Path(__file__).resolve(), *[OUT / f'trend_population_{p}.parquet' for p in ('train', 'test')]])
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        for p in ('train', 'test'):
            extract(p)
        score()
