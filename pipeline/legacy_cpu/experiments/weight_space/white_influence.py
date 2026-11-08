# CPU port; original experiments/weight_space/white_influence.py, SHA256 55b857f727175b110bde6fb87f5f2c4400b45aa305d476018635f0cbe0f1e9f3
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Likelihood scores for conditional memory, scale and volatility interventions."""
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
from scipy.special import expit, logsumexp
from threadpoolctl import threadpool_limits
from white_jump_priors import SELECT, MASS, noise_core
from white_walk import logpdf
from partition_evidence import OUT
import white_loop as wl
CORE = np.array([noise_core(l) for l in SELECT])
TYPES = ('loc', 'scale', 'ar1', 'ar2', 'asym', 'arch', 'vol2', 'slope', 'volslope')
PRIOR = np.array([0.65, 0.32, 0.26, 0.22, 0.22, 0.25, 0.25, 0.5, 0.22])

def influence(z, law):
    name, k, r = law
    if name == 'normal':
        psi = z
    elif name == 'student':
        psi = (k + 1) * z / (k - 2 + z * z)
    elif name == 'bimodal':
        v = 1 - k * k
        psi = (z - k * np.tanh(z * k / v)) / v
    else:
        v = 1 / (1 - k + k * r * r)
        p = expit(np.log(k / (1 - k) / r) + 0.5 * z * z * (1 - 1 / (r * r)) / v)
        psi = z * ((1 - p) / v + p / (v * r * r))
    return (psi.clip(-12, 12), (z * psi - 1).clip(-12, 28))

def center(x):
    a, b = np.quantile(x, [0.25, 0.75])
    return (0.5 * (a + b), max(0.5 * (b - a), 0.1 * x.std(), 1e-06))

def info(U):
    V = U - U.mean(0)
    A = V.T @ V
    r = V[:-1].T @ V[1:]
    s = V[:-2].T @ V[2:]
    return A + 0.45 * (r + r.T) + 0.15 * (s + s.T) + np.diag(np.maximum(np.diag(A) * 0.015, 0.001))

def one(raw, innov, b):
    a, c = (innov[2:b], innov[b:])
    ma, ca = center(a)
    mb, cb = center(c)
    n = len(a) + len(c)
    frac = len(a) / n
    shared = frac * ma + (1 - frac) * mb
    spread = (frac * ca * ca + (1 - frac) * cb * cb) ** 0.5
    own = np.r_[a - ma, c - mb]
    full = np.r_[a, c] - shared
    scales = spread / CORE
    ll = np.array([np.sum(logpdf(own / s, l)) - len(own) * np.log(s) for l, s in zip(SELECT, scales)]) + np.log(MASS)
    post = np.exp(ll - logsumexp(ll))
    active = np.argsort(post)[-3:]
    past = raw[1:-1]
    past2 = raw[:-2]
    sl = np.arange(n) < len(a)

    def local_standard(v):
        u = v.copy()
        for rows in (sl, ~sl):
            m, s = center(v[rows])
            u[rows] = (u[rows] - m) / max(s / 0.67449, 1e-05)
        return u.clip(-4, 4)
    x = local_standard(past)
    x2 = local_standard(past2)
    prev = np.r_[0.0, own[:-1]]
    prev2 = np.r_[0.0, 0.0, own[:-2]]
    t = np.linspace(-1, 1, n)
    records = []
    for j in active:
        scale = scales[j]
        law = SELECT[j]
        psi, chi = influence(full / scale, law)
        ps, cs = influence(own / scale, law)
        m = np.tanh(abs(prev / scale) - 0.75)
        m2 = np.tanh(abs(prev2 / scale) - 0.75)
        U = np.column_stack([psi, chi, ps * x, ps * x2, ps * (abs(x) - 0.75), cs * m, cs * m2, psi * t, chi * t])
        ua, ub = (U[:len(a)], U[len(a):])
        L, R = (info(ua), info(ub))
        G = L + R
        A, Bs = (ua.sum(0), ub.sum(0))
        rhs = R @ np.linalg.inv(G + np.diag(1 / (2 * PRIOR) ** 2))
        s = Bs - rhs @ (A + Bs)
        i = np.diag(R - rhs @ R).clip(0.01)
        items = []
        for size in (0.36, 1.0, 2.2):
            v = (size * PRIOR) ** 2
            items.extend(0.5 * (s * s / (i + 1 / v) - np.log1p(i * v)))
        Ld, Rd = (np.diag(L).clip(0.01), np.diag(R).clip(0.01))
        v = PRIOR ** 2
        BF = lambda S, I: 0.5 * (S * S / (I + 1 / v) - np.log1p(I * v))
        pre, postbf, pooled = (BF(A, Ld), BF(Bs, Rd), BF(A + Bs, Ld + Rd))
        for block in (pre, postbf, pre + postbf - pooled):
            items.extend(block[2:7])
        items.extend((s / np.sqrt(i)).clip(-20, 20))
        records.append(np.array(items))
    H = np.stack(records)
    w = post[active] / post[active].sum()
    avg = w @ H
    mix = logsumexp(H.clip(-30, 60) + np.log(w[:, None].clip(1e-30)), axis=0)
    return np.r_[avg, mix[:27], post].clip(-80, 100)

def extract(part):
    path = OUT / f'conditional_scores_{part}.parquet'
    if path.exists():
        return
    with np.load(OUT / f'pulse_streams_{part}.npz') as p:
        ids, x, n, b = (p[k] for k in ('ids', 'separate', 'n', 'b'))
    with np.load(OUT / f'state_streams_{part}.npz') as p:
        assert np.array_equal(ids, p['ids'])
        r = p['common']
    mat = np.stack([one(u[:e].astype(float), v[:e].astype(float), j) for u, v, e, j in zip(x, r, n, b)])
    assert np.isfinite(mat).all()
    pd.DataFrame(mat, index=pd.Index(ids, name='id'), columns=[f'c{j}' for j in range(mat.shape[1])]).astype(np.float32).to_parquet(path)
    print(part, 'conditional scores', mat.shape, flush=True)

def evaluate():
    _, ids, tids, *_ = wl.load()
    h = np.vstack([pd.read_parquet(OUT / f'conditional_scores_{p}.parquet').loc[ix].to_numpy() for p, ix in (('train', ids), ('test', tids))])
    f = np.arcsinh(h.clip(-18, 40))
    w = np.r_[*[pd.read_parquet(OUT / f'innov_v2_{p}.parquet').loc[ix, 'pre_white'].to_numpy() for p, ix in (('train', ids), ('test', tids))],]

    def design(ix, z):
        return np.column_stack([f[ix], f[ix, :27] * w[ix, None], f[ix, 9:18] * np.tanh(z[:, None] / 2)])
    wl.run('conditional', design, 180.0, dependencies=[Path(__file__).resolve(), *[OUT / f'conditional_scores_{p}.parquet' for p in ('train', 'test')]])
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        for p in ('train', 'test'):
            extract(p)
        evaluate()
