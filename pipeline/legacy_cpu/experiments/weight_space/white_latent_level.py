# CPU port; original experiments/weight_space/white_latent_level.py, SHA256 ecca0d0b9717c18041baee6fbed77857ad848c0d773e346b3e21c2401db0a721
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Marginalize slowly wandering hidden levels separately from measurement noise."""
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
import numpy as np
import pandas as pd
import torch
from scipy.special import gammaln
from threadpoolctl import threadpool_limits
from partition_evidence import OUT, ROOT
from white_jump_priors import tensor
MODELS = [(0.0, 1.0, 0.49)]
for phi, prob in ((-0.85, 0.035), (-0.46, 0.045), (0.42, 0.05), (0.72, 0.055), (0.9, 0.055), (0.975, 0.06), (0.996, 0.055), (1.0, 0.155)):
    for gain, w in ((0.025, 0.14), (0.085, 0.19), (0.23, 0.2), (0.51, 0.19), (0.8, 0.1), (1.0, 0.18)):
        MODELS.append((phi, gain, prob * w))

def kernels(T):
    phi, g, mass = np.array(MODELS).T
    theta = phi * (1 - g)
    t = np.arange(T)
    u = theta[:, None] ** t
    impulse = np.column_stack([np.ones(len(theta)), -(phi * g)[:, None] * u[:, :-1]])
    stationary = g * (1 - phi ** 2 * (1 - g)) / np.maximum(1 - phi ** 2, 1e-07)
    extra = np.maximum(np.where(phi == 1, 30.0, stationary) - g, 0.0)
    return (torch.fft.rfft(tensor(impulse), n=2 * T), tensor(u), tensor(np.cumsum(impulse, 1)), tensor(np.cumsum(u, 1)), tensor(extra), tensor(np.log(mass / mass.sum())))

def gaussian(sse, logdet, n):
    n = n.float()
    k = 3 + n / 2
    return torch.lgamma(k) - gammaln(3.0) + 3 * np.log(2) - n * np.log(2 * np.pi) / 2 - k * torch.log(2 + sse.clamp_min(0) / 2) - logdet / 2

def fit(x, n, b, bank, intervene):
    H, u, step, accum, extra, _ = bank
    B, T = x.shape
    t = torch.arange(T, device=x.device)[None, :]
    present = (t < n[:, None])[:, None, :]
    y = torch.fft.irfft(torch.fft.rfft(x, n=2 * T)[:, None, :] * H[None, :, :], n=2 * T)[:, :, :T]
    U = u[None, :, :] * present
    factor = extra[None, :] / (1 + extra[None, :] * (U * U).sum(2))
    base_det = torch.log1p(extra[None, :] * (U * U).sum(2))

    def prod(x, z):
        return (x * z * present).sum(2) - factor * (x * U).sum(2) * (z * U).sum(2)
    ds = [step[None, :, :], step.cumsum(1)[None, :, :] / n[:, None, None]]
    if intervene:
        j = t - b[:, None]
        active = (j >= 0)[:, None, :]
        ii = j.clamp_min(0)

        def move(v):
            return v[:, ii].permute(1, 0, 2) * active
        ds.extend([move(step), move(u), move(accum), move(step.cumsum(1)) / n[:, None, None]])
        later = (j - 6).clamp_min(0)
        delayed = step[:, later].permute(1, 0, 2) * (j >= 6)[:, None, :]
        ds.append(move(step) - delayed)
    yy = prod(y, y)
    z = prod(ds[0], y)
    g = prod(ds[0], ds[0]) + 1 / 2.5 ** 2
    sse = yy - z * z / g
    det = base_det + torch.log(g * 2.5 ** 2)
    tdy = prod(ds[1], y)
    cross = prod(ds[1], ds[0])
    tdy = tdy - cross * z / g
    tg = (prod(ds[1], ds[1]) - cross.square() / g).clamp_min(0) + 1 / 1.5 ** 2
    ss1 = (sse - tdy.square() / tg).clamp_min(0)
    dt1 = det + torch.log(tg * 1.5 ** 2)
    alts = {}
    for k, d in enumerate(ds[2:]):
        r0 = prod(d, ds[0])
        xy = prod(d, y) - r0 * z / g
        xx = (prod(d, d) - r0.square() / g).clamp_min(0)
        rt = prod(d, ds[1]) - r0 * cross / g
        for label, s, ld, xy1, xx1 in ((0, sse, det, xy, xx), (1, ss1, dt1, xy - rt * tdy / tg, (xx - rt.square() / tg).clamp_min(0))):
            items = []
            for sd in (0.05, 0.18, 0.55) if k == 2 else (0.25, 0.75, 1.65):
                I = xx1 + sd ** (-2)
                items.append(gaussian(s - xy1.square() / I, ld + torch.log(I * sd ** 2), n[:, None]))
            alts[k, label] = torch.stack(items, 2)
    return (sse.clamp_min(0), det, ss1, dt1, alts)

def blend(a, b):
    return torch.logaddexp(a + np.log(0.83), b + np.log(0.17))

def calc(x, n, b, bank):
    H, u, step, accum, extra, logp = bank
    B, T = x.shape
    s, d, st, dt, alts = fit(x, n, b, bank, True)
    base = blend(gaussian(s, d, n[:, None]), gaussian(st, dt, n[:, None]))
    lp = base + logp
    null = torch.logsumexp(lp, 1)
    post = lp.softmax(1)
    name = ('observed_step', 'kick', 'driver', 'slope', 'pulse')
    out = {}
    for k, tag in enumerate(name):
        L = blend(alts[k, 0], alts[k, 1]) + logp[None, :, None]
        odds = L.logsumexp(1) - null[:, None]
        for i in range(3):
            out[tag + str(i)] = odds[:, i]
    pos = torch.arange(T, device=x.device)[None, :] + b[:, None]
    other = x.gather(1, pos.clamp_max(T - 1)).masked_fill(pos >= n[:, None], 0)
    a, ad, _, _, _ = fit(x, b, b, bank, False)
    c, cd, _, _, _ = fit(other, n - b, b, bank, False)
    pre = gaussian(a, ad, b[:, None])
    after = gaussian(c, cd, (n - b)[:, None])
    shared = gaussian(a + c, ad + cd, n[:, None])
    out['restart_state'] = (shared + logp).logsumexp(1) - null
    out['restart_new_scale'] = (pre + after + logp).logsumexp(1) - null
    out['restart_new_process'] = (pre + logp).logsumexp(1) + (after + logp).logsumexp(1) - null
    P = np.array(MODELS)
    for tag, mask in (('no_memory', P[:, 0] == 0), ('walk', P[:, 0] == 1), ('hidden', P[:, 1] < 0.51), ('slow_hidden', (P[:, 1] < 0.51) & (P[:, 0] > 0.9)), ('negative', P[:, 0] < 0)):
        for nm, p in (('full', post), ('pre', (pre + logp).softmax(1)), ('post', (after + logp).softmax(1))):
            out[tag + '_' + nm] = p[:, mask].sum(1)
    return out

def validate(x):
    n = 8
    v = x[:n].astype(float)
    t = np.arange(n)
    for ph, g, _ in MODELS:
        q = g * (1 - ph ** 2 * (1 - g))
        p0 = 30.0 if ph == 1 else q / (1 - ph ** 2)
        cov = p0 * ph ** (t[:, None] + t[None, :]) + (1 - g) * np.eye(n)
        for j in range(1, n):
            h = np.where(t >= j, ph ** np.maximum(t - j, 0), 0.0)
            cov += q * h[:, None] * h[None, :]
        th = ph * (1 - g)
        u = th ** t
        k = np.r_[1.0, -ph * g * u[:-1]]
        r = np.convolve(v, k)[:n]
        den = 1 + (p0 - g) * (u @ u)
        assert abs(r @ r - (p0 - g) * (r @ u) ** 2 / den - v @ np.linalg.solve(cov, v)) < 1e-05
        assert abs(np.log(den) - np.linalg.slogdet(cov)[1]) < 1e-06

def extract(stream, part):
    dst = OUT / f'latent_level_{stream}_{part}.parquet'
    if dst.exists():
        return
    path = ROOT / f'tune_out/seq_cache_{part}.npz' if stream == 'raw' else OUT / f'{stream}_streams_{part}.npz'
    with np.load(path, allow_pickle=True) as q:
        ids, x, n, b = [q[k] for k in ('ids', 'x' if stream == 'raw' else 'separate', 'n', 'b')]
    validate(x[0])
    T = x.shape[1]
    bank = kernels(T)
    rows = []
    for j in range(0, len(x), 70):
        sl = slice(j, j + 70)
        nn = torch.tensor(n[sl], device='cpu')
        bb = torch.tensor(b[sl], device='cpu')
        rec = calc(tensor(x[sl]), nn, bb, bank)
        arr = torch.stack(list(rec.values()), 1).clamp(-80, 120).cpu().numpy()
        rows.append(pd.DataFrame(arr, index=ids[sl].astype(str), columns=list(rec)))
    result = pd.concat(rows)
    result.index.name = 'id'
    assert np.isfinite(result).all().all()
    result.to_parquet(dst)
    print(stream, part, 'latent-level evidence', result.shape, flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('streams', nargs='+', choices=('pulse', 'raw', 'state'))
    with torch.inference_mode(), threadpool_limits(limits=1):
        torch.set_num_threads(1)
        for stream in p.parse_args().streams:
            for part in ('train', 'test'):
                extract(stream, part)
