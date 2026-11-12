# CPU port; original experiments/weight_space/white_joint_density.py, SHA256 92c66f2be155bfd49385751a4027ab18836cedeb3584b8ec07bc47d05aa520e1
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Class-conditional latent transitions with heterogeneous measurement errors.

Errors are measured by deleting contiguous blocks of each real segment. The
Gaussian posterior updates follow the extreme-deconvolution identity. Training
groups similar measurement covariances to avoid repeated matrix inversions.
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
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from white_deconv import side
from partition_evidence import OUT
import white_loop as wl

def observe(part, stream):
    path = OUT / f'joint_latent_{stream}_{part}.npz'
    if path.exists():
        with np.load(path) as a:
            return (a['ids'], a['m'], a['V'])
    with np.load(OUT / f'{stream}_streams_{part}.npz') as a:
        ids, x, b, n = (a[k] for k in ('ids', 'separate', 'b', 'n'))
    m = []
    v = []
    for row, j, end in zip(x, b, n):
        obs = []
        errs = []
        for s in (row[:j], row[j:end]):
            s = s.astype(float)
            mu, cov = side(s)
            tr = np.ones(len(mu))
            tr[0] = 1 / np.sqrt(1 + mu[0] ** 2)
            mu[0] = np.arcsinh(mu[0])
            mu[1] += np.log(np.sqrt(np.pi))
            sc = np.exp(mu[1]).clip(0.1, 8) * tr[0]
            cov = cov * tr[:, None] * tr[None, :]
            standard = np.array([sc, 0.85, 1.0, 1.1, 0.85, 1.0, 1.0, 1.0])
            cov = 0.6 * cov + 0.23 * np.diag(np.diag(cov)) + 0.17 * np.diag(standard ** 2 / len(s)) + np.eye(len(mu)) * 1e-07
            obs.append(mu)
            errs.append(cov)
        m.append(obs)
        v.append(errs)
    m = np.array(m)
    v = np.array(v)
    assert np.isfinite(m).all() and np.isfinite(v).all()
    np.savez_compressed(path, ids=ids, m=m, V=v)
    print('measured joint', part, stream, flush=True)
    return (ids, m, v)

def spd(s, floor):
    e, u = np.linalg.eigh((s + s.T) * 0.5)
    return u * e.clip(floor) @ u.T

def data(m, v, cols):
    d = len(cols)
    M = m[:, :, cols].reshape(len(m), 2 * d)
    V = np.zeros((len(m), 2 * d, 2 * d))
    V[:, :d, :d] = v[:, 0][:, cols][:, :, cols]
    V[:, d:, d:] = v[:, 1][:, cols][:, :, cols]
    return (M, V)

def classify_noise(V, tr):
    idx = np.triu_indices(V.shape[1])
    sd = np.sqrt(np.diagonal(V, axis1=1, axis2=2))
    corr = V[:, idx[0], idx[1]] / (sd[:, idx[0]] * sd[:, idx[1]])
    feat = np.column_stack([np.log(sd.clip(1e-05)), 0.3 * corr])
    km = MiniBatchKMeans(n_clusters=28, n_init=1, max_iter=60, batch_size=768, random_state=142).fit(feat[tr])
    groups = km.predict(feat)
    med = np.stack([V[tr][groups[tr] == k].mean(0) if np.any(groups[tr] == k) else V[tr].mean(0) for k in range(28)])
    return (groups, med)

def initial(x, k):
    km = MiniBatchKMeans(n_clusters=k, batch_size=768, n_init=2, max_iter=100, random_state=107).fit(x)
    cl = km.predict(x)
    fallback = np.cov(x.T) + np.eye(x.shape[1]) * 0.03
    rates = []
    mu = []
    cov = []
    for j in range(k):
        v = x[cl == j]
        rates.append(len(v) + 1)
        mu.append(v.mean(0) if len(v) else km.cluster_centers_[j])
        cov.append(spd(np.cov(v.T) * 0.93 + fallback * 0.07, 0.0003) if len(v) > x.shape[1] + 2 else fallback)
    rates = np.array(rates)
    return (rates / rates.sum(), np.stack(mu), np.stack(cov))

def likelihood(x, g, N, p, mu, S, posterior=False):
    ll = np.empty((len(x), len(p)))
    latent = []
    errors = []
    for k in range(len(p)):
        sigma = S[k] + N
        inv = np.linalg.inv(sigma)
        logdet = np.linalg.slogdet(sigma)[1]
        d = x - mu[k]
        sol = np.einsum('nij,nj->ni', inv[g], d)
        ll[:, k] = np.log(p[k]) - 0.5 * (np.sum(d * sol, axis=1) + logdet[g])
        if posterior:
            latent.append(mu[k] + sol @ S[k])
            errors.append(S[k] - S[k] @ inv @ S[k])
    return (ll, latent, errors)

def learn(x, g, N, base):
    p0, mu0, S0 = base
    p, mu, S = (p0.copy(), mu0.copy(), S0.copy())
    for _ in range(22):
        ll, h, C = likelihood(x, g, N, p, mu, S, True)
        r = np.exp(ll - logsumexp(ll, axis=1, keepdims=True))
        nk = r.sum(0)
        p = (nk + 20 * p0) / (len(x) + 20)
        for k in range(len(p)):
            shrink = 14.0
            mu[k] = (r[:, k] @ h[k] + shrink * mu0[k]) / (nk[k] + shrink)
            centered = h[k] - mu[k]
            bins = np.bincount(g, weights=r[:, k], minlength=len(N))
            cov = centered.T @ (centered * r[:, k, None]) + np.einsum('g,gij->ij', bins, C[k])
            a = mu0[k] - mu[k]
            S[k] = spd((cov + shrink * (S0[k] + np.outer(a, a))) / (nk[k] + shrink), 0.0001)
    return (p, mu, S)

def score_one(M, V, train, target, y):
    scale = M[train].std(0).clip(0.11)
    center = np.median(M[train], axis=0)
    x = ((M - center) / scale).clip(-10, 10)
    V = V / scale[None, :, None] / scale[None, None, :]
    g, N = classify_noise(V, train)
    base = initial(x[train], 14)
    h = x.shape[1] // 2
    results = []
    for label in (0, 1):
        ix = train[y[train] == label]
        p, mu, S = learn(x[ix], g[ix], N, base)
        vals = []
        for s in (slice(None), slice(0, h), slice(h, None)):
            ll, *_ = likelihood(x[target, s], g[target], N[:, s, s], p, mu[:, s], S[:, s, s])
            vals.append(logsumexp(ll, axis=1))
        results.append(np.stack(vals, 1))
    diff = results[1] - results[0]
    return np.column_stack([diff, diff[:, 0] - diff[:, 1], diff[:, 0] - diff[:, 2], results[0][:, 0] - results[0][:, 1] - results[0][:, 2]]).clip(-35, 35)

def run(stream):
    _, ids, tids, y, *_ = wl.load()
    ti, m, v = observe('train', stream)
    ei, mm, vv = observe('test', stream)
    assert np.array_equal(ti, ids) and np.array_equal(ei, tids)
    m = np.vstack([m, mm])
    v = np.vstack([v, vv])
    n = len(y)
    te = n + np.arange(len(tids))
    group = np.r_[*[pd.read_parquet(OUT / f'innov_v2_{p}.parquet').loc[ix, 'pre_white'].to_numpy() for p, ix in (('train', ids), ('test', tids))],]
    views = ([0, 1], [1, 3, 4], [2, 3, 4], [1, 5, 6], [1, 3, 7])
    tables = [data(m, v, cols) for cols in views]
    oof = None
    tests = []
    for f, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(ids, y)):
        dst = OUT / f'joint_latent_{stream}_fold{f}.npz'
        if dst.exists():
            with np.load(dst) as a:
                assert np.array_equal(a['va'], va)
                F = a['F']
        else:
            target = np.r_[va, te]
            fields = []
            for M, V in tables:
                for mask in (group[tr] > 0.55, group[tr] <= 0.55):
                    fields.append(score_one(M, V, tr[mask], target, y))
            F = np.column_stack(fields)
            np.savez_compressed(dst, va=va, F=F)
        if oof is None:
            oof = np.empty((len(y), F.shape[1]))
        oof[va] = F[:len(va)]
        tests.append(F[len(va):])
        print('joint transition', stream, f, flush=True)
    h = np.vstack([oof, np.mean(tests, axis=0)])
    path = OUT / f'joint_latent_{stream}_scores.npz'
    np.savez_compressed(path, ids=ti, tids=ei, features=h)
    h = h.reshape(len(h), len(views), 2, -1)
    h = (group[:, None, None] * h[:, :, 0] + (1 - group[:, None, None]) * h[:, :, 1]).reshape(len(h), -1)

    def design(ix, z):
        a = np.arcsinh(h[ix])
        return np.column_stack([a, a * np.tanh(z[:, None] / 2)])
    wl.run('joint_latent_' + stream, design, 140.0, dependencies=[Path(__file__).resolve(), path])
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stream', default='state', choices=('state', 'pulse'))
    with threadpool_limits(limits=1):
        run(p.parse_args().stream)
