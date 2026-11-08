# CPU port; original experiments/weight_space/white_shapes.py, SHA256 18ec7bf684f77c2a38270157955b55d0ea05c34416aa12b0d7728e843ecd6f17
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Fit empirical symmetric noise shapes and class-specific transition priors."""
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
from scipy.optimize import brentq
from scipy.special import gammaln, logsumexp, ndtr, stdtr
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
import white_loop as wl
from partition_evidence import OUT
EDGES = np.r_[np.linspace(0, 1.8, 31), 2.05, 2.35, 2.75, 3.25, 4.0, 5.0, 7.0, 11.0, np.inf]

def anchors():
    cdfs = [ndtr]
    for d in (3.0, 5.0, 8.0, 15.0):
        cdfs.append(lambda x, d=d: stdtr(d, x))
    for mu in (0.8, 1.25, 1.9, 2.7):
        cdfs.append(lambda x, mu=mu: 0.5 * (ndtr(x - mu) + ndtr(x + mu)))
    for p, scale in ((0.02, 4.0), (0.05, 3.0), (0.12, 3.0)):
        cdfs.append(lambda x, p=p, scale=scale: (1 - p) * ndtr(x) + p * ndtr(x / scale))
    cdfs.append(lambda x: np.clip((x + 1) / 2, 0, 1))
    priors = []
    for cdf in cdfs:
        q = brentq(lambda x: cdf(x) - 0.75, 0.03, 5)
        mass = np.maximum(np.diff(2 * cdf(EDGES * q / 0.67449) - 1), 4e-05)
        priors.append(mass / mass.sum())
    return np.stack(priors)

def counts(x):
    q1, q3 = np.quantile(x, [0.25, 0.75])
    z = abs(x - (q1 + q3) * 0.5) / max((q3 - q1) / 1.34898, 0.15 * x.std(), 1e-06)
    return np.histogram(z, EDGES)[0]

def data(part):
    path = OUT / f'shape_counts_{part}.npz'
    if not path.exists():
        with np.load(OUT / f'pulse_streams_{part}.npz') as q:
            ids, x, n, b = (q[k] for k in ('ids', 'separate', 'n', 'b'))
        c = np.asarray([[counts(a[:j]), counts(a[j:e])] for a, j, e in zip(x, b, n)], np.uint16)
        np.savez_compressed(path, ids=ids, hist=c)
    with np.load(path) as a:
        return (a['ids'], a['hist'].astype(float))

def templates(obs, init):
    p = init.copy()
    pi = np.full(len(p), 1 / len(p))
    for _ in range(28):
        s = 0.65 * (obs @ np.log(p).T) + np.log(pi)
        r = np.exp(s - logsumexp(s, axis=1, keepdims=True))
        pi = (r.sum(0) + 8) / (len(obs) + 8 * len(p))
        cnt = r.T @ obs + 1400 * init
        m = cnt / (cnt.sum(1, keepdims=True) * init)
        sm = 0.8 * m + 0.1 * np.column_stack([m[:, 0], m[:, :-1]]) + 0.1 * np.column_stack([m[:, 1:], m[:, -1]])
        p = (sm * init).clip(1e-06)
        p /= p.sum(1, keepdims=True)
    return (p, pi)

def dirichlet_log(H, p, conc):
    n = H.sum(1)
    prior = p * conc
    base = gammaln(conc) - gammaln(conc + n)
    cols = [base + (gammaln(H + a) - gammaln(a)).sum(1) for a in prior]
    return np.column_stack(cols)

def weights(E, ix, prior):
    p = prior.copy()
    for _ in range(24):
        r = E[ix] + np.log(p)
        r = np.exp(r - logsumexp(r, axis=1, keepdims=True))
        p = (r.sum(0) + 95 * prior) / (len(ix) + 95)
    return p

def fold(H, tr, white, y):
    stable = tr[(y[tr] == 0) & (white[tr] > 0.6)]
    white_tr = tr[white[tr] > 0.6]
    obs = np.vstack([H[white_tr, 0], H[stable, 1]])
    p, pi = templates(obs, anchors())
    prior = np.r_[0.8 * pi, 0.2 * np.outer(pi, pi).ravel()]
    features = []
    for conc in (120.0, 650.0, 3600.0):
        a, b, j = (dirichlet_log(h, p, conc) for h in (H[:, 0], H[:, 1], H.sum(1)))
        mat = (a[:, :, None] + b[:, None, :]).reshape(len(a), -1)
        E = np.column_stack([j, mat])
        E -= E.max(1, keepdims=True)
        pp = [weights(E, white_tr[y[white_tr] == label], prior) for label in (0, 1)]
        ll = [logsumexp(E + np.log(pr), axis=1) for pr in pp]
        alt = logsumexp(mat + np.log(prior[len(pi):]), axis=1)
        common = logsumexp(j + np.log(pi), axis=1)
        u, v = [np.exp(t + np.log(pi) - logsumexp(t + np.log(pi), axis=1, keepdims=True)) for t in (a, b)]
        features.append(np.column_stack([(ll[1] - ll[0]).clip(-15, 20), (alt - common).clip(-15, 30), (u * v).sum(1), abs(u - v).max(1), -(u * np.log(u.clip(1e-15))).sum(1), -(v * np.log(v.clip(1e-15))).sum(1)]))
        if conc == 650.0:
            features.append(np.column_stack([u, v, u * v]))
    return (np.column_stack(features), p, pi)

def main():
    _, ids, tids, y, *_ = wl.load()
    ti, a = data('train')
    ei, b = data('test')
    assert np.array_equal(ti, ids) and np.array_equal(ei, tids)
    H = np.vstack([a, b])
    n = len(y)
    white = np.r_[*[pd.read_parquet(OUT / f'innov_v2_{p}.parquet').loc[ix, 'pre_white'].to_numpy() for p, ix in (('train', ids), ('test', tids))],]
    train, test, params = (None, [], {})
    for f, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(ids, y)):
        h, p, pi = fold(H, tr, white, y)
        if train is None:
            train = np.empty((n, h.shape[1]))
        train[va] = h[va]
        test.append(h[n:])
        params[f'fold{f}_templates'] = p
        params[f'fold{f}_rates'] = pi
        print('shape transitions', f, flush=True)
    fs = np.vstack([train, np.mean(test, axis=0)])
    path = OUT / 'shape_transition_scores.npz'
    np.savez_compressed(path, ids=ti, tids=ei, features=fs, **params)

    def design(ix, z):
        v = fs[ix]
        g = 0.12 + 0.88 * white[ix]
        return g[:, None] * np.column_stack([v, v[:, :3] * np.tanh(z[:, None] / 2)])
    wl.run('shape_transition', design, 120.0, dependencies=[Path(__file__).resolve(), path])
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
