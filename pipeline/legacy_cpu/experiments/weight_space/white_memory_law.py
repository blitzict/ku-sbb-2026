# CPU port; original experiments/weight_space/white_memory_law.py, SHA256 9137efede662f4345242d0a62750745b32cfdb7a7284bffc40810dee6e934a9c
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Integrate changes in AR partial correlations under non-Gaussian innovations."""
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
import torch
from scipy.special import expit
from threadpoolctl import threadpool_limits
from white_jump_priors import SELECT, MASS, tensor
from white_walk import logpdf
from partition_evidence import OUT
import white_loop as wl

def axis(m, steps, tail, end):
    high = np.geomspace(m * 1.14, end, tail)
    return np.r_[-high[::-1], np.linspace(-m, m, steps), high]
E = axis(4.8, 85, 9, 55.0)
L = axis(4.8, 61, 10, 75.0)
POS = np.array([0, 0.018, 0.035, 0.055, 0.08, 0.11, 0.145, 0.19, 0.245, 0.315, 0.4, 0.5, 0.62, 0.76, 0.92, 1.1, 1.32, 1.6, 1.95])
D = np.r_[-POS[:0:-1], POS]
S = np.linspace(-0.92, 0.92, 23)
IDS = [0, 1, 2, 3, 5, 6, 8, 10, 11]
P = MASS[IDS] / MASS[IDS].sum()

def bank():
    x = (E[:, None, None, None] - L[None, :, None, None] * D[None, None, :, None]) / np.exp(S)[None, None, None, :]
    dev = torch.empty((len(E) * len(L), len(IDS), len(D), len(S)), device='cpu')
    for i, j in enumerate(IDS):
        a = logpdf(x, SELECT[j]) - S
        dev[:, i] = tensor(a.clip(-15000, 30).reshape(len(E) * len(L), len(D), len(S)))
    return dev.flatten(1).contiguous()

def interp(v, grid):
    l = (np.searchsorted(grid, v, side='right') - 1).clip(0, len(grid) - 2)
    a = ((v - grid[l]) / (grid[l + 1] - grid[l])).clip(0, 1)
    return (l, a)

def hist(y, x):
    iy, fy = interp(y, E)
    ix, fx = interp(x, L)
    res = np.zeros(len(E) * len(L))
    for dy, wy in ((0, 1 - fy), (1, fy)):
        for dx, wx in ((0, 1 - fx), (1, fx)):
            res += np.bincount((iy + dy) * len(L) + ix + dx, wy * wx, minlength=len(res))
    return res

def row(v, b, phi, lag):
    a, c = (v[:b], v[b:])
    loc = [np.mean(q.clip(*np.quantile(q, [0.05, 0.95]))) for q in (a, c)]
    center = np.r_[np.full(b, loc[0]), np.full(len(c), loc[1])]
    z = v - center
    r = z[2:] - phi[0] * z[1:-1] - phi[1] * z[:-2]
    p = phi[0] / max(1 - phi[1], 0.05)
    predictor = (1 - phi[1]) * z[1:-1] if lag == 1 else z[:-2] - p * z[1:-1]
    scale = max(1.4826 * np.median(abs(r - np.median(r))), 0.18 * r.std(), 0.01)
    return ([hist(r[:b - 2] / scale, predictor[:b - 2] / scale), hist(r[b:] / scale, predictor[b:] / scale)], scale, np.clip(p if lag == 1 else phi[1], -1, 1))

def point(at):
    l, f = interp(at, D)
    v = np.full((len(at), len(D)), 1e-12)
    v[np.arange(len(at)), l] += 1 - f
    v[np.arange(len(at)), l + 1] += f
    return v

def extract(part, lag, bankmat):
    path = OUT / f'memory_law{lag}_{part}.parquet'
    if path.exists():
        return
    with np.load(OUT / f'pulse_streams_{part}.npz') as q:
        ids, x, n, b = (q[k] for k in ('ids', 'separate', 'n', 'b'))
    phi = pd.read_parquet(OUT / f'state_{part}.parquet').loc[ids, ['ar1', 'ar2']].to_numpy()
    pm = tensor(np.log(P))
    lse = torch.logsumexp
    delta = D[None, :] - D[:, None]
    wd = np.gradient(D)
    sizes = [(0.0, 0.06), (0.0, 0.16), (0.0, 0.36), (0.25, 0.09), (0.5, 0.15)]
    kernels = [(np.exp(-0.5 * ((delta - r) / s) ** 2) + np.exp(-0.5 * ((delta + r) / s) ** 2)) * wd[None, :] for r, s in sizes]
    rows = []
    names = None
    for start in range(0, len(x), 60):
        h = []
        sc = []
        ref = []
        for r, j, e, p in zip(x[start:start + 60], b[start:start + 60], n[start:start + 60], phi[start:start + 60]):
            hh, ss, rr = row(r[:e].astype(float), j, p, lag)
            h.append(hh)
            sc.append(ss)
            ref.append(rr)
        B = len(h)
        val = (tensor(np.asarray(h)).reshape(B * 2, -1) @ bankmat).reshape(B, 2, len(IDS), len(D), len(S))
        val -= val.amax((2, 3, 4), keepdim=True)
        a, c = val.unbind(1)
        ref = np.array(ref)
        sc = np.array(sc)
        bounded = expit((0.96 - abs(D[None, :] + ref[:, None])) / 0.025)
        uni = bounded * wd[None, :]
        uni /= uni.sum(1, keepdims=True)
        spike = point(-ref)
        unit = point(1 - ref)
        prior = 0.54 * spike + 0.38 * uni + 0.08 * unit if lag == 1 else 0.9 * spike + 0.1 * uni
        pdyn = tensor(np.log(prior))[:, None, :, None]
        logsd = np.log(sc[:, None]) + S
        sv = 0.9 * expit((0.71 - abs(logsd)) / 0.05) + 0.1 * np.exp(-abs(logsd))
        ps = tensor(np.log(sv / sv.sum(1, keepdims=True)))[:, None, :]
        null = lse(lse(a + c + pdyn, 2) + ps, 2)
        z0 = lse(null + pm, 1)
        aw = lse(a + pdyn, 2)
        bw = lse(c + pdyn, 2)
        free = lse(aw + bw + ps, 2)
        sep = lse(lse(a + ps[:, :, None, :], 3) + lse(c + ps[:, :, None, :], 3) + pdyn[..., 0], 2)

        def total(v):
            return lse(v + pm, 1) - z0
        f = {'independent_memory': total(free), 'fixed_memory_new_scale': total(sep), 'pre_pacf': tensor(ref)}
        left = a + pdyn
        ma, mb = (left.amax(2, keepdim=True), c.amax(2, keepdim=True))
        ea = (left - ma).exp().transpose(2, 3).contiguous().reshape(B, -1, len(D))
        eb = (c - mb).exp().transpose(2, 3)
        for j, rawk in enumerate(kernels):
            K = rawk[None, :, :] * bounded[:, None, :]
            K = K / K.sum(2, keepdims=True).clip(1e-12)
            acc = (ea @ tensor(K)).reshape(B, len(IDS), len(S), len(D))
            ss = (acc * eb).sum(3).clamp_min(1e-37).log() + ma[:, :, 0, :] + mb[:, :, 0, :]
            val = lse(ss + ps, 2)
            f[f'jump{j}'] = total(val)
            if j in (0, 2, 3):
                prelaw = (lse(aw + ps, 2) + pm).softmax(1)
                for i in range(len(IDS)):
                    f[f'law{i}_jump{j}'] = prelaw[:, i] * (val[:, i] - null[:, i]).clip(-18, 40)
        for nm, fixed in (('zero', spike), ('unit', unit)):
            if lag == 2 and nm == 'unit':
                continue
            lp = tensor(np.log(fixed))[:, None, :, None]
            pre, post = (lse(a + lp, 2), lse(c + lp, 2))
            for k, v in (('both', pre + post), ('start', pre + bw), ('end', aw + post)):
                f[nm + '_' + k] = total(lse(v + ps, 2))
        f['mixed'] = lse(torch.stack([v for k, v in f.items() if k in ('independent_memory', 'jump1', 'jump2', 'zero_start', 'zero_end')], 1), 1) - np.log(5)
        names = list(f)
        rows.append(torch.stack(list(f.values()), 1).clip(-80, 100).cpu().numpy())
    df = pd.DataFrame(np.vstack(rows), index=pd.Index(ids, name='id'), columns=names)
    assert np.isfinite(df).all().all()
    df.to_parquet(path)
    print('PACF', lag, part, df.shape, flush=True)

def evaluate():
    _, ids, tids, *_ = wl.load()
    blocks = []
    for lag in (1, 2):
        df = pd.concat([pd.read_parquet(OUT / f'memory_law{lag}_{p}.parquet').loc[ix] for p, ix in (('train', ids), ('test', tids))])
        blocks.append(np.arcsinh(df.clip(-18, 40).to_numpy()))
    f = np.column_stack(blocks)
    w = np.r_[*[pd.read_parquet(OUT / f'innov_v2_{p}.parquet').loc[ix, 'pre_white'].to_numpy() for p, ix in (('train', ids), ('test', tids))],]

    def design(ix, z):
        return np.column_stack([f[ix], f[ix] * w[ix, None], f[ix, :8] * np.tanh(z[:, None] / 2)])
    wl.run('memory_laws', design, 160.0, dependencies=[Path(__file__).resolve(), *[OUT / f'memory_law{j}_{p}.parquet' for j in (1, 2) for p in ('train', 'test')]])
if __name__ == '__main__':
    with threadpool_limits(limits=1), torch.inference_mode():
        torch.set_num_threads(1)
        bankmat = bank()
        for lag in (1, 2):
            for p in ('train', 'test'):
                extract(p, lag, bankmat)
        evaluate()
