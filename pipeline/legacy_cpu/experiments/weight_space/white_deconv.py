# CPU port; original experiments/weight_space/white_deconv.py, SHA256 42c369c113bfb29c514656ca56f25aeb63a487d013117acd5f16061ac5137090
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Measured distributional changes with series-specific sampling errors.

Each side contributes robust L-moments, serial dependence and a delete-block
covariance. The mixture likelihood integrates that covariance for each row.
Only outer-training labels fit the latent changes and their frequencies.
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
from functools import lru_cache
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from partition_evidence import OUT
import white_loop as wl

@lru_cache(maxsize=1300)
def rank_scores(n):
    r = 2 * (np.arange(n) + 0.5) / n - 1
    return np.polynomial.legendre.legvander(r, 5)[:, [1, 2, 3, 5]].T / n

def summary(sorted_x, ordered, mask):
    n = len(sorted_x)
    lo, hi = sorted_x[[n // 100, n - 1 - n // 100]]
    s = np.clip(sorted_x, lo, hi)
    l2, l3, l4, l6 = rank_scores(n) @ s
    l2 = max(l2, 1e-06)
    med = sorted_x[n // 2]
    z = ((ordered - med) / (1.77 * l2)).clip(-3, 3)
    mag = np.tanh(abs(z) - 0.7)

    def corr(v, k):
        keep = mask[k:] & mask[:-k]
        a, b = (v[k:][keep], v[:-k][keep])
        a = a - a.mean()
        b = b - b.mean()
        return a @ b / np.sqrt(max(a @ a * (b @ b), 1e-10))
    return np.array([s.mean(), np.log(l2), 4 * l3 / l2, 6 * l4 / l2, 9 * l6 / l2, corr(z, 1), corr(z, 2), corr(mag, 1)])

def side(x):
    ix = np.argsort(x)
    n = len(x)
    groups = np.arange(n) * 8 // n
    ref = summary(x[ix], x, np.full(n, True))
    jack = []
    for g in range(8):
        mask = groups != g
        jack.append(summary(x[ix[mask[ix]]], x, mask))
    j = np.stack(jack)
    j -= j.mean(0)
    return (ref, 7 / 8 * (j.T @ j))

def measurements(part, stream):
    path = OUT / f'deconv_{stream}_measure_{part}.npz'
    if path.exists():
        with np.load(path) as q:
            return (q['ids'], q['D'], q['V'])
    with np.load(OUT / f'{stream}_streams_{part}.npz') as q:
        ids, x, n, b = (q[k] for k in ('ids', 'separate', 'n', 'b'))
    d, v = ([], [])
    for xx, j, e in zip(x, b, n):
        r = xx[:e].astype(float)
        med = np.median(r)
        r = (r - med) / max(np.median(abs(r - med)) * 1.4826, r.std() * 0.24, 1e-06)
        m1, v1 = side(r[:j])
        m2, v2 = side(r[j:])
        d.append(m2 - m1)
        V = v1 + v2
        prior = np.diag([1.3, 0.8, 0.9, 0.8, 0.5, 1.1, 1.1, 1.2]) * (1 / j + 1 / (e - j))
        v.append(0.6 * V + 0.25 * np.diag(np.diag(V)) + 0.15 * prior)
    d, v = (np.stack(d), np.stack(v))
    np.savez_compressed(path, ids=ids, D=d, V=v)
    print(stream, part, 'measured', flush=True)
    return (ids, d, v)

def start(d):
    baseline = np.full(d, 0.025 ** 2)
    modes = [baseline, baseline * 10]
    for cols, scale in (([0], 0.55), ([1], 0.3), ([2, 3, 4], 0.3), ([5, 6], 0.24), ([7], 0.16)):
        m = baseline.copy()
        m[cols] = scale ** 2
        modes.append(m)
    modes.extend([np.full(d, 0.15 ** 2), np.full(d, 0.55 ** 2)])
    return np.array([np.diag(m) for m in modes])

def logdens(x, v, S, conditional=False):
    A = v + S
    inv = np.linalg.inv(A)
    sol = (inv @ x[:, :, None])[:, :, 0]
    ll = -0.5 * ((sol * x).sum(1) + np.linalg.slogdet(A)[1])
    if conditional:
        B = sol @ S
        latent = S - S @ inv @ S + B[:, :, None] * B[:, None, :]
        return (ll, latent)
    return ll

def learn(x, v, S0):
    S = S0.copy()
    k = len(S)
    p = np.full(k, 1 / k)
    for iteration in range(13):
        ll = np.empty((len(x), k))
        lats = []
        for j, s in enumerate(S):
            ll[:, j], lat = logdens(x, v, s, True)
            lats.append(lat)
        posterior = ll + np.log(p)
        r = np.exp(posterior - logsumexp(posterior, axis=1, keepdims=True))
        nk = r.sum(0)
        p = (nk + 3) / (len(r) + 3 * k)
        for j, lat in enumerate(lats):
            S[j] = (np.einsum('i,ijk->jk', r[:, j], lat) + 35 * S0[j]) / (nk[j] + 35)
            S[j] = (S[j] + S[j].T) * 0.5 + np.eye(x.shape[1]) * 1e-08
    return (p, S)

def trial(stream):
    q, ids, tids, y, *_ = wl.load()
    ti, d, v = measurements('train', stream)
    ei, de, ve = measurements('test', stream)
    assert np.array_equal(ti, ids) and np.array_equal(ei, tids)
    d = np.clip(np.vstack([d, de]), -8, 8)
    v = np.concatenate([v, ve])
    w = np.r_[*[pd.read_parquet(OUT / f'innov_v2_{p}.parquet').loc[ix, 'pre_white'].to_numpy() for p, ix in (('train', ids), ('test', tids))],]
    n = len(y)
    te = n + np.arange(len(tids))
    S0 = start(d.shape[1])
    rows, tests, params = (None, [], {})
    for f, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(ids, y)):
        targ = np.r_[va, te]
        features = []
        for fam, flag in enumerate((w[tr] > 0.55, w[tr] <= 0.55)):
            parts = []
            for level in (0, 1):
                ix = tr[flag & (y[tr] == level)]
                p, S = learn(d[ix], v[ix], S0)
                mat = np.column_stack([logdens(d[targ], v[targ], s) for s in S]) + np.log(p)
                parts.append(mat)
                params[f'{f}_{fam}_{level}_cov'] = S
                params[f'{f}_{fam}_{level}_p'] = p
            z0, z1 = (logsumexp(m, axis=1) for m in parts)
            posterior = np.exp(parts[1] - z1[:, None])
            features.append(np.column_stack([(z1 - z0).clip(-15, 30), posterior]))
        joined = np.column_stack(features)
        if rows is None:
            rows = np.empty((n, joined.shape[1]))
        rows[va] = joined[:len(va)]
        tests.append(joined[len(va):])
        print(stream, 'deconvolution fold', f, flush=True)
    fs = np.vstack([rows, np.mean(tests, axis=0)])
    path = OUT / f'deconv_{stream}_scores.npz'
    np.savez_compressed(path, ids=ti, tids=ei, features=fs, **params)

    def design(ix, z):
        A, B = np.split(fs[ix], 2, axis=1)
        h = np.column_stack([w[ix, None] * A, (1 - w[ix, None]) * B])
        return np.column_stack([h, h[:, [0, A.shape[1]]] * np.tanh(z[:, None] / 2)])
    wl.run('deconv_' + stream, design, 100.0, dependencies=[Path(__file__).resolve(), path])
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stream', default='pulse', choices=['pulse', 'state'])
    with threadpool_limits(limits=1):
        trial(parser.parse_args().stream)
