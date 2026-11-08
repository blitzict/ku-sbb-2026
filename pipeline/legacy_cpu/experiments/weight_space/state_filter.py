# CPU port; original experiments/weight_space/state_filter.py, SHA256 c19d9bec9747c3f778e0af88bfe6628c0d4d5624b828dcfbeeb3722b5bbdb02a
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Recover innovations under finite-order memory and persistent conditional variance."""
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
from scipy.signal import lfilter
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from partition_evidence import OUT

def prior_ar(s):
    c = s - np.mean(s)
    X = np.column_stack([c[1:-1], c[:-2]])
    return (X.T @ X, X.T @ c[2:], c[2:] @ c[2:], len(X))

def fit_ar(stats):
    G, C, S, n = stats
    eps = max(S * 1e-06, 1e-07)
    a = C[0] / max(G[0, 0], eps)
    v = np.linalg.solve(G + eps * np.eye(2), C)
    v[1] = np.clip(v[1], -0.94, 0.94)
    v[0] = np.clip(v[0] / (1 - v[1]), -0.98, 0.98) * (1 - v[1])
    M = np.array([[0, 0], [np.clip(a, -0.975, 0.975), 0], v, [1.0, 0.0]])
    rss = (S - 2 * M @ C + (M @ G * M).sum(1)).clip(eps)
    logp = -0.5 * n * np.log(rss / n) - np.array([0, 0.5, 1.0, 0]) * np.log(n) + np.log([0.54, 0.29, 0.09, 0.08])
    w = np.exp(logp - logsumexp(logp))
    return (w @ M, w, logsumexp(logp))
DECAY = [0.0, 0.4, 0.72, 0.89, 0.96]
MIX = np.array([0.25, 0.55, 0.82, 0.97])

def vol(e, scale, df):
    scale = max(scale, 1e-06)
    v = (e / scale) ** 2
    prev = np.r_[1.0, np.minimum(v[:-1], 36.0)]
    paths = [np.ones(len(e))]
    for beta in DECAY:
        sm = lfilter([1 - beta], [1.0, -beta], prev, zi=[beta])[0]
        paths.extend(1 + MIX[:, None] * (sm - 1))
    h = np.array(paths).clip(0.025, 150.0)
    tau = (df - 2) / df
    ll = (-0.5 * np.log(h) - 0.5 * (df + 1) * np.log1p(v[None, :] / (df * tau * h))).sum(axis=1)
    return (h, ll)

def state(raw, b, df):
    a, c = (raw[:b], raw[b:])
    sr, st = (prior_ar(a), prior_ar(c))
    common, pc, cm = fit_ar([x + y for x, y in zip(sr, st)])
    pre, pa, am = fit_ar(sr)
    post, pb, bm = fit_ar(st)
    center = np.mean(raw)
    prev = np.r_[raw[0], raw[:-1]] - center
    prev2 = np.r_[raw[:1], raw[:1], raw[:-2]] - center
    u = raw - common[0] * prev - common[1] * prev2
    v = raw - np.r_[np.full(b, pre[0]), np.full(len(c), post[0])] * prev - np.r_[np.full(b, pre[1]), np.full(len(c), post[1])] * prev2
    ma, mb = (np.median(u[2:b]), np.median(u[b + 2:]))
    e1 = u[2:b] - ma
    e2 = u[b + 2:] - mb
    s1, s2 = (max(e1.std(), 1e-06), max(e2.std(), 1e-06))
    pool = np.sqrt((e1 @ e1 + e2 @ e2) / (len(e1) + len(e2)))
    ea = np.r_[u[:b] - ma, u[b:] - mb]
    h0, l0 = vol(ea, pool, df)
    h1, l1 = vol(v[:b] - np.median(v[2:b]), max(v[2:b].std(), 1e-06), df)
    h2, l2 = vol(v[b:] - np.median(v[b + 2:]), max(v[b + 2:].std(), 1e-06), df)
    prior = np.log(np.r_[0.9, np.full(len(l0) - 1, 0.1 / (len(l0) - 1))])
    z0, z1, z2 = (logsumexp(ll + prior) for ll in (l0, l1, l2))
    w0, w1, w2 = (np.exp(ll + prior - z) for ll, z in ((l0, z0), (l1, z1), (l2, z2)))
    h = w0 @ h0
    hs = np.r_[w1 @ h1, w2 @ h2]
    uz = np.r_[ma + (u[:b] - ma) / np.sqrt(h[:b]), mb + (u[b:] - mb) / np.sqrt(h[b:])]
    vm, vn = (np.median(v[2:b]), np.median(v[b + 2:]))
    vz = np.r_[vm + (v[:b] - vm) / np.sqrt(hs[:b]), vn + (v[b:] - vn) / np.sqrt(hs[b:])]
    ne = b * len(c) / len(raw)
    f = dict(order2=pc[2], pre2=pa[2], post2=pb[2], ar2=common[1], ar1=common[0], walk=pc[3], pre_ar1=pre[0], post_ar1=post[0], pre_ar2=pre[1], post_ar2=post[1], ar_change=am + bm - cm, clustering=1 - w0[0], pre_cluster=1 - w1[0], post_cluster=1 - w2[0], vol_bf=z0 - l0[0], vol_split_bf=z1 - l1[0] + (z2 - l2[0]) - (z0 - l0[0]), prior_scale=s1, post_scale=s2, scale_z=abs(np.log(s1 / s2)) * np.sqrt(ne), change_level=abs(uz[:b].mean() - uz[b:].mean()) * np.sqrt(ne) / max(pool, 1e-06), change_scale=abs(np.log(max(uz[:b].std(), 1e-06) / max(uz[b:].std(), 1e-06))) * np.sqrt(ne))
    return (uz, vz, f)

def run(part):
    with np.load(OUT / f'pulse_streams_{part}.npz') as q:
        ids, x, n, b = (q[k] for k in ('ids', 'separate', 'n', 'b'))
    t = pd.read_parquet(OUT / f'trend_{part}.parquet')
    assert np.array_equal(t.index, ids)
    common, separate = (np.zeros_like(x), np.zeros_like(x))
    rows = []
    for j in range(len(x)):
        raw = x[j, :n[j]].astype(float)
        u, v, h = state(raw, b[j], np.clip(t.df.iloc[j], 4, 60))
        common[j, :n[j]], separate[j, :n[j]] = (u, v)
        rows.append(h)
    frame = pd.DataFrame(rows, index=t.index)
    assert np.isfinite(frame).all().all()
    frame.to_parquet(OUT / f'state_{part}.parquet')
    np.savez(OUT / f'state_streams_{part}.npz', ids=ids, n=n, b=b, common=common, separate=separate)
    print(part, 'active AR2 / volatility', int((frame.order2 > 0.5).sum()), int((frame.clustering > 0.5).sum()), flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run('train')
        run('test')
