# CPU port; original experiments/weight_space/white_component_moves.py, SHA256 234e9b06d3f5c6e0fc787010663fbcffef53a6a9f4d125705ea1e6d204ee135d
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Integrate single-parameter changes within an observed two-mode noise law."""
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
from white_laws import AXIS, hist
from white_jump_priors import tensor
from partition_evidence import OUT
LOC = np.linspace(-3.1, 3.1, 49)
LOGS = np.linspace(-1.75, 0.92, 25)
PROB = np.array([0.015, 0.1, 0.28, 0.5, 0.72, 0.9, 0.985])
P_WEIGHT = np.array([0.035, 0.062, 0.1, 0.606, 0.1, 0.062, 0.035])
L, U = np.triu_indices(len(LOC))
MID = (LOC[L] + LOC[U]) / 2
DELTA = (LOC[U] - LOC[L]) / 2

def dictionary():
    dev = torch.device('cpu')
    x = tensor(AXIS)[:, None, None]
    m = tensor(LOC)[None, :, None]
    s = tensor(LOGS)[None, None, :]
    gauss = -0.5 * ((x - m) * (-s).exp()).square() - s
    left = torch.as_tensor(L, device=dev)
    right = torch.as_tensor(U, device=dev)
    p = tensor(PROB)
    return torch.logaddexp(gauss[:, left, :, None] + torch.log1p(-p), gauss[:, right, :, None] + p.log()).clamp_min(-20000.0).flatten(1).contiguous()

def priors(center, scale):
    mu = center[:, None] + scale[:, None] * MID
    gap = DELTA[None, :, None]
    s = np.exp(LOGS)[None, None, :]
    sigma = scale[:, None, None] * s
    pm = 0.85 * expit((2 - abs(mu)) / 0.09) / 4 + 0.15 * np.exp(-abs(mu) / 2) / 4
    ps = 0.81 * expit((np.log(sigma) + 1.15) / 0.1) * expit((0.75 - np.log(sigma)) / 0.12) + 0.19 * np.exp(-abs(np.log(sigma)))
    dg = LOC[1] - LOC[0]
    separation = np.where(gap == 0, 0.36, 0.64 * dg / 2 / (1.2 * s) * np.exp(-gap / (1.2 * s)))
    pr = pm[:, :, None] * scale[:, None, None] * ps * separation
    v = tensor(np.log(pr.clip(1e-28)))[:, :, :, None] + tensor(np.log(P_WEIGHT))
    return v - torch.logsumexp(v, (1, 2, 3), keepdim=True)

def collapse(a, b, pr, dims):
    lse = torch.logsumexp
    z = lse(a + pr, dims) + lse(b + pr, dims) - lse(pr, dims)
    return lse(z.flatten(1), 1) if z.ndim > 1 else z

def summarize(a, b, prior, groups):
    lse = torch.logsumexp
    z = lse(a + b + prior, (1, 2, 3))
    out = {}
    for tag, dims in (('dispersion', 2), ('occupancy', 3), ('dispersion_occupancy', (2, 3)), ('centers', 1), ('centers_occupancy', (1, 3)), ('centers_dispersion', (1, 2)), ('all', (1, 2, 3))):
        out[tag] = collapse(a, b, prior, dims) - z
    for name, (index, valid) in groups.items():
        pp = prior[:, index].masked_fill(~valid[None, :, :, None, None], -100000.0)
        terms = lse(pp + a[:, index], 2) + lse(pp + b[:, index], 2) - lse(pp, 2)
        out[name] = lse(terms, (1, 2, 3)) - z

    def indicators(v):
        w = (v + prior - lse(v + prior, (1, 2, 3), keepdim=True)).exp()
        g = tensor(DELTA)[:, None] / tensor(np.exp(LOGS))[None, :]
        balance = 1 - tensor(abs(PROB - 0.5)) * 2
        return [w[:, DELTA == 0].sum((1, 2, 3)), (w * (g > 1.35)[None, :, :, None]).sum((1, 2, 3)), (w * balance[None, None, None, :]).sum((1, 2, 3))]
    for pname, v in (('pre', a), ('post', b), ('pooled', a + b)):
        for k, h in zip(('unimodal', 'separated', 'balanced'), indicators(v)):
            out[pname + '_' + k] = h
    return out

def extract(part, stream):
    path = OUT / f'component_moves_{stream}_{part}.parquet'
    if path.exists():
        return
    with np.load(OUT / f'{stream}_streams_{part}.npz') as v:
        ids, x, n, b = [v[k] for k in ('ids', 'separate', 'n', 'b')]
    groups = {}
    for k, g in (('lower_fixed', L), ('upper_fixed', U), ('midpoint_fixed', L + U), ('separation_fixed', U - L)):
        indices = [np.flatnonzero(g == i) for i in np.unique(g)]
        ix = np.full((len(indices), max(map(len, indices))), -1)
        for j, js in enumerate(indices):
            ix[j, :len(js)] = js
        groups[k] = tuple((torch.as_tensor(v, device='cpu') for v in (ix.clip(0), ix >= 0)))
    bank = dictionary()
    rows = []
    for start in range(0, len(ids), 45):
        sl = slice(start, start + 45)
        hs = []
        centers = []
        scales = []
        for raw, end, j in zip(x[sl], n[sl], b[sl]):
            y = raw[:end].astype(float)
            center = np.median(y)
            s = max(np.median(abs(y - center)) * 1.4826, 0.18 * y.std(), 0.01)
            t = (y - center) / s
            hs.extend((hist(t[:j]), hist(t[j:])))
            centers.append(center)
            scales.append(s)
        A = (tensor(np.asarray(hs)) @ bank).reshape(len(centers), 2, len(L), len(LOGS), len(PROB))
        A -= A.amax((2, 3, 4), keepdim=True)
        a, d = A.unbind(1)
        prior = priors(np.array(centers), np.array(scales))
        rec = summarize(a, d, prior, groups)
        arr = torch.stack(list(rec.values()), 1).clamp(-60, 100).cpu().numpy()
        rows.append(pd.DataFrame(arr, columns=list(rec), index=ids[sl]))
    result = pd.concat(rows)
    result.index.name = 'id'
    assert result.index.is_unique and np.isfinite(result).all().all()
    result.to_parquet(path)
    print(stream, part, 'single-component changes', result.shape, flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1), torch.inference_mode():
        torch.set_num_threads(1)
        for stream in ('state', 'pulse'):
            for part in ('train', 'test'):
                extract(part, stream)
