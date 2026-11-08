# CPU port; original experiments/weight_space/white_block_kernel.py, SHA256 7778289bfa9a799a76e0a87e05e1845c96221cc49a9fe7c1e50c9b0582a6128b
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Learn two-sample features and their within-record uncertainty from real blocks."""
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
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits
from partition_evidence import OUT, ROOT
LINE = np.r_[-8.0, -5.0, -3.6, np.linspace(-2.8, 2.8, 29), 3.6, 5.0, 8.0]
PLANE = np.array([-5.0, -2.5, -1.5, -0.95, -0.55, -0.22, 0.0, 0.22, 0.55, 0.95, 1.5, 2.5, 5.0])
NCH = 4
PAIR = 4
NB = 6

def normalized(v):
    q = np.quantile(v, [0.25, 0.75])
    m = q.mean()
    return (v - m) / max((q[1] - q[0]) / 1.34898, 0.18 * v.std(), 1e-07)

def indexes(v, axis):
    u = (np.searchsorted(axis, v, side='right') - 1).clip(0, len(axis) - 2)
    f = ((v - axis[u]) / (axis[u + 1] - axis[u])).clip(0, 1)
    return (u, f)

def pack(a, e, p, b):
    n = len(a)
    z = normalized(a)
    w = normalized(e)
    shape = np.r_[normalized(e[:b]), normalized(e[b:])]
    ar = normalized(np.r_[0.0, np.diff(p)])
    block = np.r_[np.arange(b) * NB // b, NB + np.arange(n - b) * NB // (n - b)]
    length = np.bincount(block, minlength=2 * NB)

    def line(x):
        g, f = indexes(x, LINE)
        base = len(LINE) * block
        v = np.bincount(g + base, 1 - f, minlength=2 * NB * len(LINE)) + np.bincount(g + base + 1, f, minlength=2 * NB * len(LINE))
        return v.reshape(2, NB, -1) / length.reshape(2, NB, 1)

    def plane(y, x):
        iy, fy = indexes(y, PLANE)
        ix, fx = indexes(x, PLANE)
        L = len(PLANE)
        total = np.zeros(2 * NB * L * L)
        base = block * L * L
        for j, v in ((0, 1 - fy), (1, fy)):
            for k, r in ((0, 1 - fx), (1, fx)):
                total += np.bincount(base + (iy + j) * L + ix + k, weights=v * r, minlength=len(total))
        return total.reshape(2, NB, -1) / length.reshape(2, NB, 1)
    lag = lambda v, k: np.r_[np.repeat(v[0], k), v[:-k]]
    a1 = np.stack([line(v) for v in (z, w, shape, ar)])
    a2 = np.stack([plane(w, lag(w, 1)), plane(w, lag(w, 2)), plane(z, lag(z, 1)), plane(w, lag(z, 1))])
    return (a1.astype(np.float16), a2.astype(np.float16))

def charts(part):
    f = OUT / f'block_kernel_{part}.npz'
    if not f.exists():
        arrays = []
        for tag, key in (('pulse', 'separate'), ('state', 'separate'), ('pulse', 'common')):
            with np.load(OUT / f'{tag}_streams_{part}.npz') as q:
                ids, x, n, b = (q[k] for k in ('ids', key, 'n', 'b'))
                arrays.append(x)
        L = np.empty((len(n), NCH, 2, NB, len(LINE)), np.float16)
        P = np.empty((len(n), PAIR, 2, NB, len(PLANE) ** 2), np.float16)
        for i, (end, j) in enumerate(zip(n, b)):
            L[i], P[i] = pack(*(x[i, :end].astype(float) for x in arrays), j)
        np.savez(f, ids=ids, l=L, p=P, n=np.stack([b, n - b], 1))
        print('block histograms', part, len(n), flush=True)
    with np.load(f) as q:
        return [q[k] for k in ('ids', 'l', 'p', 'n')]

def initial():
    x = LINE
    f = np.stack([x.clip(-3, 3), np.tanh(2 * x), np.log1p(x * x), np.cos(2 * x), np.cos(3.5 * x), np.sin(2 * x), np.tanh(3 * (abs(x) - 1.65)), np.cos(0.8 * x), np.cos(5 * x)], 1)
    y, x = np.meshgrid(PLANE, PLANE, indexing='ij')
    p = np.stack([np.tanh(y) * np.tanh(x), np.tanh(abs(y) - 0.7) * np.tanh(abs(x) - 0.7), np.tanh(y) * np.tanh(abs(x) - 0.7), np.tanh(abs(y) - 0.7) * np.tanh(x), np.cos(y - x), np.cos(y + x), np.sin(y) * np.cos(2 * x), np.tanh(2 * (abs(y) - 1.5)) * np.tanh(2 * (abs(x) - 1.5)), np.cos(3 * y) * np.cos(3 * x)], -1).reshape(-1, 9)
    return (f, p)

class LearnedKernel(nn.Module):

    def __init__(self, dim):
        super().__init__()
        f, p = initial()
        self.register_buffer('f0', torch.tensor(f, dtype=torch.float32)[None].repeat(NCH, 1, 1))
        self.register_buffer('p0', torch.tensor(p, dtype=torch.float32)[None].repeat(PAIR, 1, 1))
        self.line = nn.Parameter(torch.zeros_like(self.f0))
        self.plane = nn.Parameter(torch.zeros_like(self.p0))
        self.points = nn.Sequential(nn.Linear((NCH + PAIR) * 9 * 7, 150), nn.LayerNorm(150), nn.GELU(), nn.Dropout(0.2))
        self.phys = nn.Sequential(nn.Linear(dim, 90), nn.LayerNorm(90), nn.GELU(), nn.Dropout(0.18))
        self.read = nn.Sequential(nn.Linear(240, 120), nn.LayerNorm(120), nn.GELU(), nn.Dropout(0.22), nn.Linear(120, 1))

    def summarize(self, h, k, n):
        means = torch.einsum('bgstd,gdk->bgstk', h, k).float()
        pointvar = torch.einsum('bgsd,gdk->bgsk', h.mean(3), k.square()).float()
        m = means.mean(3)
        pv = (pointvar - m.square()).clamp_min(0.003) / n[:, None, :, None]
        bv = means.var(3, unbiased=True) / NB
        var = 0.5 * pv + 0.5 * bv
        se = (var[:, :, 0] + var[:, :, 1] + 0.0001).sqrt()
        delta = m[:, :, 1] - m[:, :, 0]
        sz = (delta / se).clamp(-20, 20)
        near = means[:, :, 1, 0] - means[:, :, 0, -1]
        return torch.stack([m[:, :, 0], m[:, :, 1], torch.asinh(3 * delta), torch.asinh(sz), torch.log1p(sz.square()), torch.log(se).clamp(-6, 2), torch.asinh(near / (se * np.sqrt(NB)))], -1).flatten(1)

    def forward(self, l, p, n, a):
        k = self.f0 + self.line
        u = self.plane.permute(0, 2, 1).reshape(PAIR * 9, 1, len(PLANE), len(PLANE))
        u = F.avg_pool2d(F.pad(u, (1, 1, 1, 1), mode='replicate'), 3, stride=1)
        u = u.reshape(PAIR, 9, -1).permute(0, 2, 1)
        stats = torch.cat([self.summarize(l, k, n), self.summarize(p, self.p0 + u, n)], 1)
        return self.read(torch.cat([self.points(stats), self.phys(a)], 1))[:, 0]

def phys(part, ids):
    columns = [f'level{j}' for j in range(6)] + [f'scale{j}' for j in range(6)] + ['shape0', 'shape1', 'relax']
    blocks = []
    for tag in ('pulse', 'state', 'nullstate', 'prestate', 'smoothvol'):
        a = pd.read_parquet(OUT / f'jump_law_{tag}_{part}.parquet').loc[ids]
        blocks.append(np.arcsinh(a[columns + ([f'pre{j}' for j in range(12)] if tag == 'state' else [])].clip(-20, 65)))
    for name, cols in (('state', ['pre2', 'ar1', 'ar2', 'walk', 'clustering', 'pre_cluster', 'prior_scale', 'post_scale']), ('curve', ['gate', 'obs_step', 'mixture']), ('periodic', ['gate', 'phase_gain', 'freq_gain']), ('walk_jump', ['step', 'scale', 'shape']), ('trend_population', ['level', 'aggregate']), ('ar_law', ['level_bf', 'memory_bf', 'tail_bf']), ('pulse', ['gate', 'step'])):
        blocks.append(np.arcsinh(pd.read_parquet(OUT / f'{name}_{part}.parquet').loc[ids, cols].clip(-25, 65)))
    return np.column_stack(blocks).astype(np.float32)

@torch.no_grad()
def predict(net, data, idx):
    net.eval()
    out = []
    for start in range(0, len(idx), 256):
        j = torch.as_tensor(idx[start:start + 256], device='cpu')
        with _cpu_context.nullcontext():
            out.append(net(*(s[j] for s in data)).float().sigmoid())
    return torch.cat(out).cpu().numpy()

def run():
    a, b = (charts('train'), charts('test'))
    ids, tids = (a[0], b[0])
    n = len(ids)
    te = n + np.arange(len(tids))
    values = [torch.tensor(np.concatenate([a[k], b[k]]), device='cpu', dtype=torch.bfloat16 if k < 3 else torch.float32) for k in (1, 2, 3)]
    del a, b
    ad = np.vstack([phys('train', ids), phys('test', tids)])
    y = pd.read_csv(ROOT / 'data/train_labels.csv', dtype={'id': str}).set_index('id').loc[ids, 'label'].to_numpy()
    yy = torch.tensor(y, device='cpu', dtype=torch.float32)
    pv = np.empty(n)
    test = []
    dst = OUT / 'block_kernel_models'
    dst.mkdir(exist_ok=True)
    for f, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(ids, y)):
        core, inner = train_test_split(tr, test_size=0.11, stratify=y[tr], random_state=535 + f)
        path = dst / f'fold{f}.npz'
        ck = dst / f'fold{f}.pt'
        if path.exists():
            with np.load(path) as res:
                assert np.array_equal(res['va'], va)
                pv[va] = res['pv']
                test.append(res['pt'])
            continue
        prior = OUT / 'law_outer0'
        law = np.vstack([pd.read_parquet(prior / f'fold{f}_{p}.parquet').loc[ix].to_numpy() for p, ix in (('train', ids), ('test', tids))])
        aux = np.column_stack([ad, np.arcsinh(law)])
        mu = aux[core].mean(0)
        sd = aux[core].std(0).clip(0.1)
        data = [*values, torch.tensor(((aux - mu) / sd).clip(-7, 7), dtype=torch.float32, device='cpu')]
        torch.manual_seed(12 + f)
        rng = np.random.default_rng(12 + f)
        net = LearnedKernel(aux.shape[1]).to('cpu')
        opt = torch.optim.AdamW(net.parameters(), lr=0.0007, weight_decay=0.012)
        best, bep = (-1, -1)
        for ep in range(32):
            net.train()
            order = rng.permutation(core)
            for start in range(0, len(core), 160):
                j = torch.as_tensor(order[start:start + 160], device='cpu')
                opt.zero_grad(set_to_none=True)
                with _cpu_context.nullcontext():
                    z = net(*(v[j] for v in data)).float()
                    loss = F.binary_cross_entropy_with_logits(z, yy[j]) + 0.003 * (net.line.square().mean() + net.plane.square().mean())
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                opt.step()
            auc = roc_auc_score(y[inner], predict(net, data, inner))
            if auc > best + 5e-05:
                best, bep = (auc, ep)
                torch.save(net.state_dict(), ck)
            if ep - bep >= 5:
                break
            if ep in (8, 17):
                for g in opt.param_groups:
                    g['lr'] *= 0.48
        net.load_state_dict(torch.load(ck, map_location='cpu', weights_only=True))
        v, t = (predict(net, data, va), predict(net, data, te))
        pv[va] = v
        test.append(t)
        np.savez_compressed(path, va=va, pv=v, pt=t, mu=mu, sd=sd)
        print('learned block kernel', f, bep, round(roc_auc_score(y[va], v), 6), flush=True)
    np.savez_compressed(OUT / 'lawblock.npz', ids=ids, tids=tids, oof=pv, test=np.mean(test, axis=0))
    print('block kernel out of fold', roc_auc_score(y, pv), flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        torch.set_num_threads(1)
        run()
