# CPU port; original experiments/weight_space/ar_likelihood.py, SHA256 620278c3f018c06160bfce6fb7a558498e6203b0924f33caa7403bc18aecae91
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Profile one-parameter intervention likelihoods under three real process families.

An unknown Student law is fitted jointly with the location, memory and scale.
For the alternatives, change one of those parameters while sharing the others.
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
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from threadpoolctl import threadpool_limits
from partition_evidence import ROOT, OUT
DEV = 'cpu'
DFS = (3.0, 5.0, 9.0, 20.0, 200.0)

def total(t):
    return t.sum(1, keepdim=True)

def profile(x, y, valid, post, df, family, kind):
    pre = valid - post
    na, nb = (total(pre), total(post))
    size = na + nb
    w = valid
    delta = 1e-08
    for _ in range(8):
        wa, wb = (w * pre, w * post)
        a, b = (total(wa).clamp_min(delta), total(wb).clamp_min(delta))
        xa, xb = (total(wa * x) / a, total(wb * x) / b)
        ya, yb = (total(wa * y) / a, total(wb * y) / b)
        if kind != 'level':
            xa = xb = (xa * a + xb * b) / (a + b)
            ya = yb = (ya * a + yb * b) / (a + b)
        centered_x = x - torch.where(post.bool(), xb, xa)
        centered_y = y - torch.where(post.bool(), yb, ya)
        xx, xy = (centered_x.square(), centered_x * centered_y)
        if kind == 'memory':
            pa = (total(wa * xy) / total(wa * xx).clamp_min(delta)).clamp(-0.99, 1.02)
            pb = (total(wb * xy) / total(wb * xx).clamp_min(delta)).clamp(-0.99, 1.02)
        elif family == 'ar':
            pa = pb = (total(w * xy) / total(w * xx).clamp_min(delta)).clamp(-0.94, 0.94)
        else:
            pa = pb = torch.full_like(a, float(family == 'walk'))
        phi = torch.where(post.bool(), pb, pa)
        r = centered_y - phi * centered_x
        if _ == 0:
            wr = valid
        else:
            wr = valid * (df + 1) / (df - 2 + r.square() / previous_scale)
        v1 = total(pre * wr * r.square()) / na
        v2 = total(post * wr * r.square()) / nb
        if kind != 'scale':
            v1 = v2 = (na * v1 + nb * v2) / size
        variance = torch.where(post.bool(), v2, v1).clamp_min(1e-07)
        previous_scale = variance
        w = valid / variance * (df + 1) / (df - 2 + r.square() / variance)
    const = torch.lgamma((df + 1) / 2) - torch.lgamma(df / 2) - 0.5 * torch.log(torch.pi * (df - 2))
    terms = const - 0.5 * torch.log(variance) - 0.5 * (df + 1) * torch.log1p(r.square() / ((df - 2) * variance))
    ll = total(valid * terms)
    return (ll, (pa, pb, v1.sqrt(), v2.sqrt(), ya - pa * xa, yb - pb * xb), r, variance)

def batch(x, n, b):
    t = torch.arange(x.shape[1] - 1, device=DEV)[None, :] + 1
    valid = (t < n[:, None]).float()
    post = ((t >= b[:, None]) & (t < n[:, None])).float()
    pre = valid - post
    n1, n2 = (total(pre), total(post))
    ne = n1 * n2 / (n1 + n2)
    center = total(x[:, 1:] * valid) / total(valid)
    x, y = (x[:, :-1] - center, x[:, 1:] - center)
    ma, mb = (total(pre * y) / n1, total(post * y) / n2)
    za, zb = (total(pre * x) / n1, total(post * x) / n2)
    dx = x - torch.where(post.bool(), zb, za)
    dy = y - torch.where(post.bool(), mb, ma)
    p = (total(valid * dx * dy) / total(valid * dx.square()).clamp_min(1e-08)).clamp(-0.97, 1.0)
    resid = dy - p * dx
    dfgrid = torch.tensor(DFS, device=DEV)[None, None, :]
    e = resid[:, :, None]
    fmask = post[:, :, None].bool()
    scores = []
    for mask in (pre, post):
        mm = mask[:, :, None]
        sz = mm.sum(1, keepdim=True)
        variance = (mm * e.square()).sum(1, keepdim=True) / sz
        for _ in range(5):
            w = (dfgrid + 1) / (dfgrid - 2 + e.square() / variance.clamp_min(1e-08))
            variance = (mm * w * e.square()).sum(1, keepdim=True) / sz
        terms = torch.lgamma((dfgrid + 1) / 2) - torch.lgamma(dfgrid / 2) - 0.5 * torch.log(torch.pi * (dfgrid - 2) * variance.clamp_min(1e-08))
        terms = terms - 0.5 * (dfgrid + 1) * torch.log1p(e.square() / ((dfgrid - 2) * variance.clamp_min(1e-08)))
        scores.append((terms * mm).sum(1))
    pooled = scores[0] + scores[1]
    df = dfgrid.reshape(-1)[pooled.argmax(1)][:, None]
    result = {'df': df, 'tail_bf': (torch.logsumexp(scores[0], 1) + torch.logsumexp(scores[1], 1) - torch.logsumexp(pooled, 1) - np.log(len(DFS)))[:, None]}
    models = {}
    for fam in ('white', 'ar', 'walk'):
        for kind in ('null', 'level', 'scale'):
            ll, params, r, var = profile(x, y, valid, post, df, fam, kind)
            models[fam, kind] = ll
            if kind == 'null':
                ib = torch.arange(len(b), device=DEV)
                v = (r / var.sqrt())[ib, b - 1].abs()[:, None]
                result[fam + '_jump'] = v
            if fam == 'ar' and kind != 'null':
                pa, pb, s1, s2, a1, a2 = params
                result['magnitude_' + kind] = abs(a2 - a1) / (s1 + s2) * 2 if kind == 'level' else abs(torch.log(s2 / s1))
    df_mem, mp, _, _ = profile(x, y, valid, post, df, 'ar', 'memory')
    lp = torch.tensor([np.log(0.55), np.log(0.36), np.log(0.09)], device=DEV, dtype=x.dtype)

    def mix(kind):
        return torch.cat([models['white', kind], models['ar', kind] - 0.5 * torch.log(n1 + n2), models['walk', kind]], 1) + lp
    prior = mix('null')
    null = torch.logsumexp(prior, 1, keepdim=True)
    prob = prior.softmax(1)
    result.update(p_white=prob[:, 0:1], p_ar=prob[:, 1:2], p_walk=prob[:, 2:3])
    penalty = 0.5 * torch.log(ne)
    for kind in ('level', 'scale'):
        result[kind + '_bf'] = torch.logsumexp(mix(kind), 1, keepdim=True) - null - penalty
        for fam in ('white', 'ar', 'walk'):
            result[fam + '_' + kind] = models[fam, kind] - models[fam, 'null'] - penalty
    result['memory_bf'] = df_mem - 0.5 * torch.log(n1 + n2) - penalty - null
    result['memory_shift'] = abs(mp[1] - mp[0])
    factors = torch.cat([result[k] for k in ('level_bf', 'scale_bf', 'memory_bf', 'tail_bf')] + [torch.zeros_like(df)], 1)
    weight = torch.tensor([0.33, 0.33, 0.1, 0.15, 0.09], device=DEV)
    result['aggregate'] = torch.logsumexp(factors + weight.log(), 1, keepdim=True)
    return {k: v.flatten().cpu().numpy() for k, v in result.items()}

def run(part):
    with np.load(ROOT / f'tune_out/seq_cache_{part}.npz', allow_pickle=True) as z:
        ids, raw, n, b = (z[k] for k in ('ids', 'x', 'n', 'b'))
    p = pd.read_parquet(OUT / f'periodic_{part}.parquet')
    assert np.array_equal(ids, p.index)
    with np.load(OUT / f'periodic_streams_{part}.npz') as z:
        assert np.array_equal(ids, z['ids'])
        residual = z['separate']
    g = p.gate.to_numpy() * expit((p.period.to_numpy() - 9) / 1.5) * expit((38 - p.period.to_numpy()) / 2)
    rows = []
    for i in range(0, len(ids), 384):
        sl = slice(i, i + 384)
        values = raw[sl] + g[sl, None].astype(np.float32) * (residual[sl] - raw[sl])
        inputs = [torch.as_tensor(a, device=DEV) for a in (values, n[sl], b[sl])]
        rows.append(pd.DataFrame(batch(*inputs), index=p.index[sl]))
    frame = pd.concat(rows)
    assert np.isfinite(frame).all().all()
    frame.to_parquet(OUT / f'ar_law_{part}.parquet')
    print(part, frame.shape, flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=2), torch.inference_mode():
        torch.set_num_threads(2)
        run('train')
        run('test')
