# CPU port; original experiments/weight_space/mixture_law.py, SHA256 e20ce19baac320f3175083dcb23bb72921ed5c8b0a219f2c8f61d5cc52ce0c5a
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Separate changes in the center, separation, scale and occupancy of two modes."""
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
from scipy.special import expit, logit, logsumexp
from threadpoolctl import threadpool_limits
from partition_evidence import ROOT, OUT

def optimize(a, b, free, start=None):
    x = np.r_[a, b]
    n = np.array([len(a), len(b)], float)
    j = np.arange(len(x)) >= len(a)
    ix = j.astype(int)
    sx = np.array([a.sum(), b.sum()])
    sx2 = np.array([a @ a, b @ b])
    pooled = x.std().clip(1e-06)
    if start is None:
        mu = np.array([np.median(a), np.median(b)])
        d = np.array([a.std(), b.std()]) * 0.82
        sd = np.maximum(0.12 * pooled, np.array([a.std(), b.std()]) * 0.57)
        p = np.array([0.5, 0.5])
    else:
        mu, d, sd, p = [v.copy() for v in start]
    one = np.array([0, 0])
    move = np.arange(2) if 'm' in free else one
    sep = np.arange(2) if 'd' in free else one
    sep = sep + 1 + move.max()
    dim = sep.max() + 1
    for _ in range(24):
        inv = 1 / sd ** 2
        r = expit(logit(p)[ix] + 2 * (d * inv)[ix] * (x - mu[ix]))
        signed = 2 * r - 1
        sh = np.array([signed[:len(a)].sum(), signed[len(a):].sum()])
        sxh = np.array([signed[:len(a)] @ a, signed[len(a):] @ b])
        mat = np.eye(dim) * 1e-07
        rhs = np.zeros(dim)
        for side in (0, 1):
            u, v = (move[side], sep[side])
            mat[u, u] += n[side] * inv[side]
            mat[v, v] += n[side] * inv[side]
            mat[u, v] += sh[side] * inv[side]
            mat[v, u] += sh[side] * inv[side]
            rhs[u] += sx[side] * inv[side]
            rhs[v] += sxh[side] * inv[side]
        beta = np.linalg.solve(mat, rhs)
        mu, d = (beta[move], beta[sep].clip(0.01 * pooled))
        rss = sx2 - 2 * mu * sx + n * mu ** 2 - 2 * d * (sxh - mu * sh) + n * d ** 2
        sd = np.sqrt((rss / n if 's' in free else np.full(2, rss.sum() / n.sum())).clip(0.012 * pooled ** 2))
        p = (0.5 + 0.5 * (sh / n if 'p' in free else np.full(2, sh.sum() / n.sum()))).clip(0.08, 0.92)
    u = (x - mu[ix]) / sd[ix]
    k = (d / sd)[ix]
    ll = np.logaddexp(np.log(p)[ix] - 0.5 * (u - k) ** 2, np.log1p(-p)[ix] - 0.5 * (u + k) ** 2) - np.log(sd)[ix]
    return (np.array([ll[:len(a)].sum(), ll[len(a):].sum()]), (mu, d, sd, p))

def one(a, b):
    n = np.array([len(a), len(b)])
    full, param = optimize(a, b, 'mdsp')
    null, zp = optimize(a, b, '', param)
    counts = 1 / (1 / n).sum()
    penalty = 0.5 * np.log(counts)
    gaussian = -n * (0.5 + np.log([max(a.std(), 1e-07), max(b.std(), 1e-07)]))
    model_bf = full - gaussian - np.log(n)
    m, d, s, p = param
    soft = expit((np.maximum(model_bf, -20).max() - 1) / 2) * expit((max(d / s) - 1.05) * 5)
    f = {'gate': soft, 'pre_bim': model_bf[0], 'post_bim': model_bf[1], 'sd_pre': s[0], 'sd_post': s[1], 'sep_pre': d[0] / s[0], 'sep_post': d[1] / s[1], 'balance_pre': abs(p[0] - 0.5), 'balance_post': abs(p[1] - 0.5)}
    scores = []
    for key in ('m', 'd', 's', 'p'):
        l, _ = optimize(a, b, key, param)
        bf = l.sum() - null.sum() - penalty
        f[key + '_bf'] = bf
        scores.append(bf)
    scores.append(full.sum() - null.sum() - 4 * penalty)
    f['all_bf'] = scores[-1]
    f['mix_evidence'] = logsumexp(np.r_[scores, 0]) - np.log(6)
    return f

def run(part):
    with np.load(OUT / f'innov_v2_streams_{part}.npz') as z:
        ids, x, n, b = (z[k] for k in ('ids', 'common', 'n', 'b'))
    index = pd.read_parquet(ROOT / f'clean/features_{part}.parquet', columns=[]).index
    assert np.array_equal(index, ids)
    old = pd.read_parquet(ROOT / f'tune_out/lr5_{part}.parquet', columns=['lr_bim_sep_pre', 'lr_bim_sep_post']).loc[index]
    candidate = old.max(axis=1).to_numpy() > 0.63
    rows = []
    for i in range(len(ids)):
        if candidate[i]:
            row = x[i, :n[i]].astype(float)
            rows.append(one(row[1:b[i]], row[b[i]:]))
        else:
            rows.append({})
    df = pd.DataFrame(rows, index=index).fillna(0)
    assert np.isfinite(df).all().all()
    df.to_parquet(OUT / f'mixture_law_{part}.parquet')
    print(part, 'candidates', int(candidate.sum()), 'high-confidence', int((df.gate > 0.5).sum()), flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run('train')
        run('test')
