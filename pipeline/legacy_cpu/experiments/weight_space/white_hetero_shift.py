# CPU port; original experiments/weight_space/white_hetero_shift.py, SHA256 d0a452ed33e03674f13514d1f2bc8b0a22c6ea97dc8419ec2b9cca44e0d59596
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Integrate shifts in physical units through a time-varying innovation scale."""
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
from scipy.signal import lfilter
from threadpoolctl import threadpool_limits
from white_volatility_law import DECAY, MIX, PRIOR, K, LOGS, prepare, table, interp
from white_jump_priors import SELECT, MASS, tensor, jump_kernel, integrated_shift
from white_walk import logpdf
from partition_evidence import OUT
import white_loop as wl
AX = np.r_[-np.geomspace(5.8, 65, 15)[::-1], np.linspace(-5.6, 5.6, 113), np.geomspace(5.8, 65, 15)]
HV = np.exp(np.linspace(-1.75, 1.8, 23))
MU = np.linspace(-1.6, 1.6, 41)
S = np.linspace(-1.1, 1.1, 29)
KSEL = [0, 1, 2, 4, 5, 6, 8, 10, 11]

def pos_scale(scale):
    logs = np.log(scale[:, None]) + LOGS
    return tensor(np.log(0.88 / 1.4 / (1 + np.exp(((abs(logs) - 0.7) / 0.06).clip(-30, 30))) + 0.12 * np.exp(-abs(logs)) * 0.5) + np.log(np.diff(LOGS)[0]))

def posterior_paths(raw, b, phi, walk, theta, scale):
    u = np.r_[raw[0], raw[1] - phi[0] * (raw[0] - raw[0]), raw[2:] - (np.column_stack([raw[1:-1], raw[:-2]]) - ((1 - walk) * np.median(raw[:b]) + walk * raw[0])) @ phi]
    arr = u[2:]
    j = b - 2
    e = arr - np.r_[np.full(j, np.median(arr[:j])), np.full(len(arr) - j, np.median(arr[j:]))]
    past = np.r_[1.0, ((e / scale) ** 2)[:-1].clip(0, 144)]
    ema = np.stack([lfilter([1 - beta], [1.0, -beta], past, zi=[beta])[0] for beta in DECAY])
    mix = np.r_[0.0, np.tile(MIX[1:], len(DECAY))]
    values = []
    for w in theta:
        sigma2 = w @ np.exp(2 * LOGS)
        long = sigma2.sum()
        intercept = (1 - mix) @ sigma2
        coeff = (w.sum(1) * mix)[1:].reshape(len(DECAY), -1).sum(1)
        values.append(np.sqrt(((intercept + coeff @ ema) / max(long, 0.0001)).clip(0.025, 65)))
    a, c, _ = values
    return (arr, np.r_[a[:j], c[j:]], values[2])

def profiles(part):
    dst = OUT / f'hetero_path_{part}.npz'
    if dst.exists():
        return
    with np.load(OUT / f'pulse_streams_{part}.npz') as q:
        ids, x, n, b = (q[k] for k in ('ids', 'separate', 'n', 'b'))
    st = pd.read_parquet(OUT / f'state_{part}.parquet').loc[ids]
    phi = st[['ar1', 'ar2']].to_numpy()
    walk = st.walk.to_numpy()
    bank = table()
    pl = tensor(MASS[K] / MASS[K].sum()).log()
    pdyn = tensor(PRIOR).log()
    u = np.zeros_like(x)
    h = np.ones_like(x)
    common = np.ones_like(x)
    for start in range(0, len(ids), 64):
        sl = slice(start, min(start + 64, len(ids)))
        args = list(zip(x[sl], n[sl], b[sl], phi[sl]))
        obs, scale = map(np.asarray, zip(*(prepare(z[:end].astype(float), j, ar) for z, end, j, ar in args)))
        ll = (tensor(obs).reshape(-1, bank.shape[0]) @ bank).reshape(len(args), len(DECAY), 2, len(MIX), len(K), len(LOGS)).transpose(1, 2)
        ll = torch.cat([ll[:, :, 0, 0:1], ll[:, :, :, 1:].flatten(2, 3)], 2)
        a, c = ll.unbind(1)
        prior = pdyn[None, :, None, None] + pl[None, None, :, None] + pos_scale(scale)[:, None, None, :]
        samples = []
        for v in (a, c, a + c):
            v = torch.logsumexp(v + prior, 2)
            samples.append(v.flatten(1).softmax(1).reshape(v.shape).cpu().numpy())
        for jj, ((z, end, j, ar), theta) in enumerate(zip(args, zip(*samples))):
            i = start + jj
            u[i, :end - 2], h[i, :end - 2], common[i, :end - 2] = posterior_paths(z[:end].astype(float), j, ar, walk[i], theta, scale[jj])
    assert np.isfinite(u).all() and np.isfinite(h).all() and np.isfinite(common).all()
    np.savez(dst, ids=ids, n=n - 2, b=b - 2, u=u, separate=h, common=common)
    print('volatility paths', part, flush=True)

def dictionary():
    sd = HV[:, None, None] * np.exp(S)[None, None, :]
    xx = (AX[:, None, None, None] - MU[None, None, :, None]) / sd[None, :, :, :]
    return tensor(np.stack([logpdf(xx, SELECT[k]) - S for k in KSEL], 2)).clip(-20000.0, 35).reshape(len(AX) * len(HV), -1)

def chart(u, h, b):
    m = np.median(u)
    s = max(np.median(abs(u - m)) * 1.4826, 0.18 * u.std(), 0.01)
    i, x = interp((u - m) / s, AX)
    j, y = interp(h, HV)
    cells = len(AX) * len(HV)
    ix = (np.arange(len(u)) >= b) * cells + i * len(HV) + j
    z = np.zeros(2 * cells)
    for k, f in ((0, 1 - x), (1, x)):
        for l, g in ((0, 1 - y), (1, y)):
            z += np.bincount(ix + k * len(HV) + l, weights=f * g, minlength=len(z))
    return (z.reshape(2, cells), m, s)

def extract(part):
    dst = OUT / f'hetero_jump_{part}.parquet'
    if dst.exists():
        return
    with np.load(OUT / f'hetero_path_{part}.npz') as a:
        ids, u, n, b, h, h0 = (a[k] for k in ('ids', 'u', 'n', 'b', 'separate', 'common'))
    bank = dictionary()
    mass = tensor(MASS[KSEL] / MASS[KSEL].sum()).log()
    lse = torch.logsumexp
    loc = [jump_kernel(MU, np.exp(S), d, w) for d, w in ((0.0, 0.08), (0.0, 0.22), (0.0, 0.52), (0.0, 1.15), (0.35, 0.15), (1.0, 0.34))]
    vel = [jump_kernel(S, np.ones(1), d, w) for d, w in ((0.0, 0.07), (0.0, 0.2), (0.0, 0.48), (0.32, 0.12), (0.6, 0.17))]
    rows = []
    for start in range(0, len(n), 48):
        sl = slice(start, min(len(n), start + 48))
        feats = {}
        for hfield, context in ((h, 'split'), (h0, 'common')):
            counts, m, s = map(np.asarray, zip(*(chart(z[:end].astype(float), v[:end], j) for z, v, end, j in zip(u[sl], hfield[sl], n[sl], b[sl]))))
            B = len(m)
            ll = (tensor(counts).reshape(-1, bank.shape[0]) @ bank).reshape(B, 2, len(KSEL), len(MU), len(S))
            ll -= ll.amax((2, 3, 4), keepdim=True)
            a, c = ll.unbind(1)
            absolute = m[:, None] + s[:, None] * MU
            logs = np.log(s[:, None]) + S
            mu = tensor(np.log(0.89 / 4 / (1 + np.exp(((abs(absolute) - 2) / 0.075).clip(-35, 35))) + 0.11 * np.exp(-abs(absolute) / 2) / 4) + np.log(s[:, None] * np.diff(MU)[0]))
            sd = tensor(np.log(0.88 / 1.4 / (1 + np.exp(((abs(logs) - 0.7) / 0.045).clip(-35, 35))) + 0.12 * np.exp(-abs(logs)) * 0.5) + np.log(np.diff(S)[0]))
            pm = mu[:, None, :, None]
            ps = sd[:, None, None, :]
            null = lse(a + c + pm + ps, (2, 3))
            z0 = lse(null + mass, 1)
            pre = (lse(a + pm + ps, (2, 3)) + mass).softmax(1)
            for prefix, ks in (('level', loc), ('scale', vel)):
                for j, ker in enumerate(ks):
                    if prefix == 'level':
                        sc = integrated_shift(a.transpose(-1, -2), c.transpose(-1, -2), mu[:, None, None, :], sd[:, None, :], ker)
                    else:
                        sc = integrated_shift(a, c, ps, mu[:, None, :], ker)
                    feats[f'{context}_{prefix}{j}'] = lse(sc + mass, 1) - z0
                    if j in (1, 3):
                        for k in range(len(KSEL)):
                            feats[f'{context}_{prefix}{j}_law{k}'] = pre[:, k] * (sc[:, k] - null[:, k]).clip(-25, 70)
            af, cf = [lse(v + mass[None, :, None, None], 1) for v in (a, c)]
            feats[context + '_shape'] = lse(af + cf + mu[:, :, None] + sd[:, None, :], (1, 2)) - z0
            for k in range(len(KSEL)):
                feats[f'{context}_pre{k}'] = pre[:, k]
        mat = torch.stack(list(feats.values()), 1).clip(-100, 160).cpu().numpy()
        rows.append(pd.DataFrame(mat, index=ids[sl], columns=list(feats)))
    out = pd.concat(rows)
    out.index.name = 'id'
    assert np.isfinite(out).all().all()
    out.to_parquet(dst)
    print(part, 'heteroscedastic physical shifts', out.shape, flush=True)

def run():
    _, ids, tids, *_ = wl.load()
    a = pd.concat([pd.read_parquet(OUT / f'hetero_jump_{p}.parquet').loc[ix] for p, ix in (('train', ids), ('test', tids))])
    gate = np.concatenate([pd.read_parquet(OUT / f'volatility_law_{p}.parquet').loc[ix, ['pooled_cluster', 'pre_cluster']].max(axis=1) for p, ix in (('train', ids), ('test', tids))])
    v = np.arcsinh(a.to_numpy().clip(-25, 60))

    def design(ix, z):
        return np.column_stack([v[ix], v[ix] * gate[ix, None]])
    wl.run('hetero_jump', design, 190.0, save=False)
if __name__ == '__main__':
    with threadpool_limits(limits=1), torch.inference_mode():
        torch.set_num_threads(1)
        for p in ('train', 'test'):
            profiles(p)
            extract(p)
        run()
