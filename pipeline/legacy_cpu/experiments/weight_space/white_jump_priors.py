# CPU port; original experiments/weight_space/white_jump_priors.py, SHA256 6273ff95e87d15d95a284d9309dba451842d9532248e37341906a915ee63b053
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Integrate *changes* relative to initial parameters rather than two free draws."""
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
from scipy.optimize import brentq
from scipy.special import ndtr, stdtr
from threadpoolctl import threadpool_limits
from white_laws import LAWS, MU, LOG_SD, dictionary, hist, priors
from partition_evidence import OUT
import white_loop as wl
SELECT = [('normal', 0.0, 0.0), ('student', 4.0, 0.0), ('student', 7.0, 0.0), ('student', 10.0, 0.0), ('student', 16.0, 0.0), ('contam', 0.025, 4.5), ('contam', 0.08, 2.25), ('contam', 0.006, 4.5), ('bimodal', 0.65, 0.0), ('bimodal', 0.78, 0.0), ('bimodal', 0.86, 0.0), ('bimodal', 0.92, 0.0)]
MASS = np.array([0.28, 0.045, 0.16, 0.12, 0.085, 0.06, 0.04, 0.045, 0.04, 0.04, 0.045, 0.04])
MASS /= MASS.sum()

def tensor(v):
    return torch.as_tensor(v, dtype=torch.float32, device='cpu')

def noise_core(law):
    name, k, r = law
    if name == 'normal':
        return 0.67448975
    if name == 'student':
        fun = lambda v: stdtr(k, v * np.sqrt(k / (k - 2)))
    elif name == 'contam':
        sd = 1 / np.sqrt(1 - k + k * r * r)
        fun = lambda v: (1 - k) * ndtr(v / sd) + k * ndtr(v / (sd * r))
    else:
        sd = np.sqrt(1 - k * k)
        fun = lambda v: 0.5 * (ndtr((v - k) / sd) + ndtr((v + k) / sd))
    return brentq(lambda v: fun(v) - 0.75, 0.02, 2.0)

def jump_kernel(axis, scales, rate, width):
    step = axis[1] - axis[0]
    delta = (axis[None, None, :] - axis[None, :, None]) / scales[:, None, None]
    a = np.exp(-0.5 * ((delta - rate) / width) ** 2)
    b = np.exp(-0.5 * ((delta + rate) / width) ** 2)
    return tensor((a + b) * step / (2 * np.sqrt(2 * np.pi) * width * scales[:, None, None]))

def integrated_shift(a, b, prior_axis, prior_outer, kernel):
    a = a + prior_axis
    ma = a.amax(-1, keepdim=True)
    mb = b.amax(-1, keepdim=True)
    left = (a - ma).exp().permute(2, 0, 1, 3).contiguous()
    outer, size, k, l = left.shape
    sm = (left.reshape(outer, -1, l) @ kernel).reshape(outer, size, k, l).permute(1, 2, 0, 3)
    ll = (sm * (b - mb).exp()).sum(-1).clamp_min(1e-37).log() + ma[..., 0] + mb[..., 0]
    return torch.logsumexp(ll + prior_outer, dim=-1)

def extract(part, stream):
    path = OUT / f'jump_law_{stream}_{part}.parquet'
    if path.exists():
        return
    with np.load(OUT / f'{stream}_streams_{part}.npz') as q:
        ids, x, n, b = (q[k] for k in ('ids', 'separate', 'n', 'b'))
    k = len(SELECT)
    idx = [LAWS.index(z) for z in SELECT]
    full = dictionary()
    bank = full.reshape(len(full), len(LAWS), len(MU), len(LOG_SD))[:, idx].flatten(1).contiguous()
    del full
    pmass = tensor(MASS).log()
    levels = [(0.0, 0.09), (0.0, 0.22), (0.0, 0.55), (0.0, 1.2), (0.36, 0.15), (1.1, 0.32)]
    scales = [(0.0, 0.045), (0.0, 0.12), (0.0, 0.3), (0.0, 0.6), (0.3, 0.11), (0.54, 0.16)]
    loc_k = [jump_kernel(MU, np.exp(LOG_SD), rate, w) for rate, w in levels]
    scale_k = [jump_kernel(LOG_SD, np.ones(1), rate, w) for rate, w in scales]
    core = np.array([noise_core(z) for z in SELECT])
    spacing = LOG_SD[1] - LOG_SD[0]
    shifts = np.log(core[:, None] / core[None, :]) / spacing
    rows = []
    names = None
    for start in range(0, len(x), 40):
        zz = []
        center = []
        noise = []
        for r, j, e in zip(x[start:start + 40], b[start:start + 40], n[start:start + 40]):
            raw = r[:e].astype(float)
            m = np.median(raw)
            s = max(np.median(abs(raw - m)) * 1.4826, 0.18 * raw.std(), 0.01)
            z = (raw - m) / s
            zz.append([hist(z[:j]), hist(z[j:])])
            center.append(m)
            noise.append(s)
        B = len(zz)
        ll = (tensor(np.array(zz)).reshape(2 * B, -1) @ bank).reshape(B, 2, k, len(MU), len(LOG_SD))
        ll -= ll.amax((2, 3, 4), keepdims=True)
        a, c = ll.unbind(1)
        mu, sd = priors(np.array(center), np.array(noise))
        pm = mu[:, None, :, None]
        ps = sd[:, None, None, :]
        lse = torch.logsumexp
        null = lse(a + c + pm + ps, (2, 3))
        z0 = lse(null + pmass, 1)
        pp = lse(a + pm + ps, (2, 3)) + pmass
        post = pp.softmax(1)
        feat = {f'pre{k0}': v for k0, v in enumerate(post.T)}
        nearest = []
        for kind, kernels in (('level', loc_k), ('scale', scale_k)):
            for j, ker in enumerate(kernels):
                if kind == 'level':
                    score = integrated_shift(a.transpose(-1, -2), c.transpose(-1, -2), mu[:, None, None, :], sd[:, None, :], ker)
                else:
                    score = integrated_shift(a, c, ps, mu[:, None, :], ker)
                contrast = lse(score + pmass, 1) - z0
                feat[kind + str(j)] = contrast
                if j in (0, 2, 4):
                    for t in range(k):
                        feat[f'{kind}{j}_law{t}'] = post[:, t] * (score[:, t] - null[:, t]).clip(-18, 45)
                if j == 0:
                    nearest.append(contrast)
        relaxed = lse(torch.stack([z0 * 0, nearest[0], nearest[1]], 1) + tensor([0.7, 0.18, 0.12]).log(), 1)
        for j, delta in enumerate((None, shifts)):
            arr = []
            for p in range(k):
                cp = c
                if delta is not None:
                    off = delta[p, :, None] + np.arange(len(LOG_SD))[None, :]
                    ind = np.floor(off).astype(int).clip(0, len(LOG_SD) - 2)
                    index = torch.as_tensor(ind, device='cpu')[None, :, None, :].expand(B, -1, len(MU), -1)
                    frac = tensor((off - ind).clip(0, 1))[None, :, None, :]
                    cp = (1 - frac) * c.gather(-1, index) + frac * c.gather(-1, index + 1)
                    cp = cp.masked_fill(torch.as_tensor((off < 0) | (off > len(LOG_SD) - 1), device='cpu')[None, :, None, :], -10000)
                arr.append(lse(a[:, p:p + 1] + cp + pm + ps, (2, 3)))
            joint = torch.stack(arr, 1) + pmass[None, :, None] + pmass[None, None, :]
            mix = lse(joint, (1, 2)) - z0
            feat[f'shape{j}'] = mix
            feat[f'shape{j}_robust'] = mix - relaxed
            for t in range(k):
                feat[f'shape{j}_from{t}'] = (lse(joint[:, t], 1) - z0).clip(-18, 45)
                feat[f'shape{j}_to{t}'] = (lse(joint[:, :, t], 1) - z0).clip(-18, 45)
        feat['relax'] = relaxed
        names = list(feat)
        rows.append(torch.stack(list(feat.values()), 1).clip(-100, 180).cpu().numpy())
    tab = pd.DataFrame(np.vstack(rows), columns=names, index=pd.Index(ids, name='id'))
    assert np.isfinite(tab).all().all()
    tab.to_parquet(path)
    print(stream, part, tab.shape, flush=True)

def run(stream):
    _, ids, tids, *_ = wl.load()
    a, b = [pd.read_parquet(OUT / f'jump_law_{stream}_{p}.parquet').loc[ix] for p, ix in (('train', ids), ('test', tids))]
    H = pd.concat([a, b])
    F = np.arcsinh(H.to_numpy().clip(-25, 60))
    w = np.r_[*[pd.read_parquet(OUT / f'innov_v2_{p}.parquet').loc[ix, 'pre_white'].to_numpy() for p, ix in (('train', ids), ('test', tids))],]

    def design(ix, z):
        g = 0.2 + 0.8 * w[ix, None] if stream == 'pulse' else 0.65 + 0.35 * (1 - w[ix, None])
        v = F[ix]
        return np.column_stack([g * v, g * v[:, [12, 25, -1]] * np.tanh(z[:, None] / 2)])
    wl.run('jump_' + stream, design, 160.0, dependencies=[Path(__file__).resolve(), *[OUT / f'jump_law_{stream}_{p}.parquet' for p in ('train', 'test')]])
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stream', default='pulse', choices=('pulse', 'state'))
    arg = p.parse_args()
    with threadpool_limits(limits=1), torch.inference_mode():
        torch.set_num_threads(1)
        for part in ('train', 'test'):
            extract(part, arg.stream)
        run(arg.stream)
