# CPU port; original experiments/weight_space/white_volatility_law.py, SHA256 7da0e85493dec55589281049888e96087c3f045c47cf7bd01972c6a2eddd0b11
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Joint marginal likelihood of ARCH dynamics, unconditional scale and noise law."""
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
import torch
from scipy.signal import lfilter
from threadpoolctl import threadpool_limits
from white_jump_priors import SELECT, MASS, tensor
from white_walk import logpdf
from partition_evidence import OUT
import white_loop as wl
E = np.r_[np.linspace(0, 4.6, 69), np.geomspace(4.8, 75, 25)]
H = np.r_[0.0, np.linspace(0.06, 3.5, 38), np.geomspace(3.8, 30, 10)]
LOGS = np.linspace(-1.35, 1.35, 37)
DECAY = (0.0, 0.35, 0.68, 0.86, 0.95)
MIX = np.array([0.0, 0.18, 0.4, 0.65, 0.85, 0.96])
K = [0, 1, 2, 4, 5, 6, 8, 10, 11]
PRIOR = np.r_[0.87, np.full(len(DECAY) * (len(MIX) - 1), 0.13 / (len(DECAY) * (len(MIX) - 1)))]

def table():
    h = (1 - MIX[None, :, None]) * np.exp(2 * LOGS)[None, None, :] + MIX[None, :, None] * H[:, None, None] ** 2
    h = h.clip(0.012)
    z = E[:, None, None, None] / np.sqrt(h)
    scores = np.stack([logpdf(z, SELECT[k]) - 0.5 * np.log(h) for k in K], axis=-2)
    return tensor(scores).clip(-30000.0, 40).reshape(len(E) * len(H), -1)

def interp(x, axis):
    i = (np.searchsorted(axis, x) - 1).clip(0, len(axis) - 2)
    return (i, ((x - axis[i]) / (axis[i + 1] - axis[i])).clip(0, 1))

def hist(z, h, b):
    u, x = interp(np.abs(z), E)
    v, y = interp(np.sqrt(h), H)
    L = len(H)
    cells = len(E) * L
    lead = (np.arange(len(z)) >= b) * cells + u * L + v
    out = np.zeros(2 * cells)
    for i, a in ((0, 1 - x), (1, x)):
        for j, c in ((0, 1 - y), (1, y)):
            out += np.bincount(lead + i * L + j, weights=a * c, minlength=len(out))
    return out.reshape(2, cells)

def prepare(raw, b, phi):
    prev = np.column_stack([raw[1:-1], raw[:-2]])
    e = raw[2:] - prev @ phi
    b = b - 2
    e = e - np.r_[np.full(b, np.median(e[:b])), np.full(len(e) - b, np.median(e[b:]))]
    scale = max(np.median(abs(e)) * 1.4826, 0.18 * e.std(), 0.01)
    z = e / scale
    past = np.r_[1.0, (z[:-1] ** 2).clip(0, 144)]
    counts = []
    for beta in DECAY:
        avg = lfilter([1 - beta], [1.0, -beta], past, zi=[beta])[0]
        counts.append(hist(z, avg, b))
    return (counts, scale)

def extract(part):
    path = OUT / f'volatility_law_{part}.parquet'
    if path.exists():
        return
    with np.load(OUT / f'pulse_streams_{part}.npz') as q:
        ids, x, n, b = (q[k] for k in ('ids', 'separate', 'n', 'b'))
    phi = pd.read_parquet(OUT / f'state_{part}.parquet').loc[ids, ['ar1', 'ar2']].to_numpy()
    bank = table()
    pdyn = tensor(PRIOR).log()
    pl = tensor(MASS[K] / MASS[K].sum()).log()
    S = len(LOGS)
    lse = torch.logsumexp
    kernels = []
    for u, w in ((0.0, 0.07), (0.0, 0.2), (0.0, 0.48), (0.32, 0.12), (0.6, 0.17)):
        delta = LOGS[:, None] - LOGS[None, :]
        ker = (np.exp(-0.5 * ((delta - u) / w) ** 2) + np.exp(-0.5 * ((delta + u) / w) ** 2)) * np.diff(LOGS)[0] / (2 * np.sqrt(2 * np.pi) * w)
        kernels.append(tensor(ker).clamp_min(1e-30))
    rows = []
    for start in range(0, len(ids), 70):
        sl = slice(start, min(start + 70, len(ids)))
        obs, scale = map(np.asarray, zip(*(prepare(row[:end].astype(float), j, ar) for row, end, j, ar in zip(x[sl], n[sl], b[sl], phi[sl]))))
        B = len(scale)
        ll = (tensor(obs).reshape(-1, len(E) * len(H)) @ bank).reshape(B, len(DECAY), 2, len(MIX), len(K), S).transpose(1, 2)
        ll = torch.cat([ll[:, :, 0, 0:1], ll[:, :, :, 1:].flatten(2, 3)], 2)
        ll -= ll.amax((2, 3, 4), keepdim=True)
        a, c = ll.unbind(1)
        logs = np.log(scale[:, None]) + LOGS
        ps = tensor(np.log(0.88 / 1.4 / (1 + np.exp(((abs(logs) - 0.7) / 0.06).clip(-30, 30))) + 0.12 * np.exp(-abs(logs)) * 0.5) + np.log(np.diff(LOGS)[0]))
        prior = pdyn[None, :, None, None] + pl[None, None, :, None] + ps[:, None, None, :]
        joint = a + c + prior
        z0 = lse(joint, (1, 2, 3))
        dctx = pdyn[None, :, None] + pl[None, None, :]
        sctx = pl[None, :, None] + ps[:, None, :]
        dscale = pdyn[None, :, None] + ps[:, None, :]
        anypre = lse(a + prior, (1, 2, 3))
        anypost = lse(c + prior, (1, 2, 3))

        def prob(x):
            return x.softmax(1)
        pre, post, pooled = [prob(lse(v + prior, (2, 3))) for v in (a, c, a + c)]
        f = {'pre_cluster': 1 - pre[:, 0], 'post_cluster': 1 - post[:, 0], 'pooled_cluster': 1 - pooled[:, 0], 'all': anypre + anypost - z0}
        ad, cd = [lse(v + pdyn[None, :, None, None], 1) for v in (a, c)]
        af, cf = [lse(v + pl[None, None, :, None], 2) for v in (a, c)]
        av, cv = [lse(v + ps[:, None, None, :], 3) for v in (a, c)]
        f['new_memory'] = lse(ad + cd + sctx, (1, 2)) - z0
        f['new_law'] = lse(af + cf + dscale, (1, 2)) - z0
        f['new_scale'] = lse(av + cv + dctx, (1, 2)) - z0
        f['memory_on'] = lse(a[:, 0] + cd + sctx, (1, 2)) - z0
        f['memory_off'] = lse(ad + c[:, 0] + sctx, (1, 2)) - z0
        f['memory_and_scale'] = lse(lse(ad + ps[:, None, :], 2) + lse(cd + ps[:, None, :], 2) + pl, 1) - z0
        f['memory_and_law'] = lse(lse(ad + pl[None, :, None], 1) + lse(cd + pl[None, :, None], 1) + ps, 1) - z0
        f['scale_and_law'] = lse(lse(af + ps[:, None, :], 2) + lse(cf + ps[:, None, :], 2) + pdyn, 1) - z0
        for j, ker in enumerate(kernels):
            u = a + ps[:, None, None, :]
            ma = u.amax(3, keepdim=True)
            mc = c.amax(3, keepdim=True)
            v = (u - ma).exp() @ ker * (c - mc).exp()
            s = v.sum(3).clamp_min(1e-35).log() + ma[:, :, :, 0] + mc[:, :, :, 0]
            f[f'scale_jump{j}'] = lse(s + dctx, (1, 2)) - z0
            f[f'scale_cluster{j}'] = lse((s + dctx)[:, 1:], (1, 2)) - z0
        for j in range(len(DECAY)):
            ix = slice(1 + j * (len(MIX) - 1), 1 + (j + 1) * (len(MIX) - 1))
            f[f'pre_decay{j}'] = pre[:, ix].sum(1)
            f[f'post_decay{j}'] = post[:, ix].sum(1)
        for j in range(len(K)):
            f[f'law{j}_null'] = lse(joint[:, :, j], (1, 2)) - z0
            f[f'law{j}_memory'] = lse((ad + cd + sctx)[:, j], 1) - z0
        rr = torch.stack(list(f.values()), 1).clip(-100, 160).cpu().numpy()
        rows.append(pd.DataFrame(rr, index=ids[sl], columns=list(f)))
    tab = pd.concat(rows)
    tab.index.name = 'id'
    assert np.isfinite(tab).all().all()
    tab.to_parquet(path)
    print(part, 'dynamic volatility', tab.shape, flush=True)

def run():
    _, ids, tids, *_ = wl.load()
    a = pd.concat([pd.read_parquet(OUT / f'volatility_law_{p}.parquet').loc[ix] for p, ix in (('train', ids), ('test', tids))])
    v = np.arcsinh(a.to_numpy().clip(-25, 60))
    w = np.maximum(a.pre_cluster, a.pooled_cluster).to_numpy()

    def design(ix, z):
        return np.column_stack([v[ix], v[ix] * w[ix, None]])
    wl.run('volatility_law', design, 180.0, save=False)
if __name__ == '__main__':
    with threadpool_limits(limits=1), torch.inference_mode():
        torch.set_num_threads(1)
        for p in ('train', 'test'):
            extract(p)
        run()
