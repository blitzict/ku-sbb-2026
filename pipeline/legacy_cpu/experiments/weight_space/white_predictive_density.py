# CPU port; original experiments/weight_space/white_predictive_density.py, SHA256 b386659959146006ac13aaa2ec2a72f32ff3101ad41da547d6992bef5ee95d81
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Learn the distribution of held-out real observations and test interventions on it."""
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
from torch import nn
from torch.nn import functional as F
from scipy.special import ndtr, stdtr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits
from white_block_kernel import phys
from partition_evidence import OUT, ROOT
EDGES = np.r_[-np.inf, -8.0, -5.0, -3.6, np.linspace(-2.8, 2.8, 31), 3.6, 5.0, 8.0, np.inf]
BOUND = np.r_[-15.0, EDGES[1:-1], 15.0].astype(np.float32)
K = len(EDGES) - 1
STREAMS = ('state', 'pulse', 'smoothvol')

def measure(support, query):
    q = np.quantile(support, [0.25, 0.75])
    m = q.mean()
    sd = max((q[1] - q[0]) / 1.349, 0.19 * support.std(), 0.01)

    def hist(x):
        return np.histogram((x - m) / sd, bins=EDGES)[0]
    return np.stack([hist(support), hist(query)], 0)

def observations(part):
    path = OUT / f'predictive_density_{part}.npz'
    if not path.exists():
        views = []
        for source in STREAMS:
            with np.load(OUT / f'{source}_streams_{part}.npz') as q:
                ids, x, n, b = (q[k] for k in ('ids', 'separate', 'n', 'b'))
            channels = []
            for v, e, j in zip(x, n, b):
                left, right = (v[:j], v[j:e])
                split = []
                for side in (left, right):
                    a, c, d = np.split(side, [len(side) // 4, 3 * len(side) // 4])
                    outside = np.r_[a, d]
                    split.extend([measure(outside, c), measure(c, outside)])
                channels.append([measure(left, right), measure(right, left), *split])
            views.append(np.stack(channels))
        data = np.stack(views, 1).astype(np.int16)
        np.savez_compressed(path, ids=ids, h=data)
        print('actual context and query counts', part, data.shape, flush=True)
    with np.load(path) as d:
        return (d['ids'], d['h'])

def templates():

    def normalize(cdf):
        from scipy.optimize import brentq
        lo, hi = (brentq(lambda x: cdf(x) - p, -20, 20) for p in (0.25, 0.75))
        v = cdf(EDGES * (hi - lo) / 1.349 + 0.5 * (hi + lo))
        p = np.diff(v).clip(1.5e-05)
        return p / p.sum()
    cdfs = [ndtr]
    cdfs += [lambda x, k=k: stdtr(k, x) for k in (3.0, 6.0, 13.0)]
    cdfs += [lambda x, s=s, p=p: (1 - p) * ndtr(x) + p * ndtr(x / s) for s, p in ((4.5, 0.026), (2.5, 0.11))]
    cdfs += [lambda x, m=m, p=p: (1 - p) * ndtr(x + m) + p * ndtr(x - m) for m, p in ((0.9, 0.5), (1.5, 0.5), (2.4, 0.5), (1.65, 0.29), (1.65, 0.71))]
    cdfs += [lambda x, d=d: 0.935 * ndtr(x) + 0.065 * ndtr((x - d) / 1.4) for d in (-3.0, 3.0)]
    return np.stack([normalize(f) for f in cdfs]).astype(np.float32)

def shift_matrices():
    scenarios = [(0.0, 0.0)]
    for d in (0.075, 0.16, 0.31, 0.57, 1.0, 1.7, 2.65):
        scenarios.extend([(d, 0.0), (-d, 0.0)])
    for s in (0.055, 0.12, 0.24, 0.43, 0.68, 0.92):
        scenarios.extend([(0.0, s), (0.0, -s)])
    for d in (-0.36, 0.36):
        for s in (-0.23, 0.23):
            scenarios.append((d, s))
    rows = []
    for d, s in scenarios:
        edge = (BOUND - d) / np.exp(s)
        cdf = ((edge[None, :] - BOUND[:-1, None]) / np.diff(BOUND)[:, None]).clip(0, 1)
        cdf[:, 0] = 0
        cdf[:, -1] = 1
        mat = np.diff(cdf, axis=1)
        assert np.allclose(mat.sum(1), 1)
        rows.append(mat)
    return np.asarray(rows, np.float32)

class EmpiricalDensity(nn.Module):

    def __init__(self, naux):
        super().__init__()
        p = templates()
        self.register_buffer('p0', torch.tensor(p))
        self.register_buffer('T', torch.tensor(shift_matrices()))
        self.offset = nn.Parameter(torch.zeros(p.shape))
        self.ctx = nn.Sequential(nn.Linear(K + 1, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, len(p) + 3))
        self.aux = nn.Sequential(nn.Linear(naux, 96), nn.LayerNorm(96), nn.SiLU(), nn.Dropout(0.14))
        self.small = nn.Sequential(nn.Linear(31 + len(p) + 11, 75), nn.LayerNorm(75), nn.GELU(), nn.Dropout(0.12), nn.Linear(75, 32))
        self.head = nn.Sequential(nn.Linear(6 * 32 + 96, 128), nn.LayerNorm(128), nn.SiLU(), nn.Dropout(0.21), nn.Linear(128, 1))

    def base(self, h):
        n = h.sum(-1, keepdim=True)
        prop = h / n
        prior = F.softmax(self.p0.log() + 0.45 * torch.tanh(self.offset), -1)
        ll = h @ prior.log().T
        adj = self.ctx(torch.cat([prop.sqrt() * 4, n.log() / 6], -1))
        mix = F.softmax(ll * 0.32 + adj[..., :len(prior)], -1)
        target = mix @ prior
        strength = n * (0.28 + 3.4 * adj[..., -1:].sigmoid())
        smooth = (h + strength * target) / (n + strength)
        return ((0.996 * smooth + 0.004 * prior.mean(0)).clamp_min(2e-07), mix, adj, prior)

    def forward(self, h, aux):
        p, mix, adj, prior = self.base(h[:, :, :2, 0])
        H = h[:, :, :2, 1]
        n = H.sum(-1, keepdim=True)
        m = h[:, :, :2, 0].sum(-1, keepdim=True)
        pp = torch.einsum('bsvk,akd->bsvad', p, self.T).float().clamp_min(2e-07)
        concentration = m * (0.4 + 3.5 * adj[..., -2:-1].sigmoid())
        alpha = (pp * concentration[..., None]).clamp_min(0.0001)
        Hq = H[:, :, :, None, :]
        ll = (torch.lgamma(alpha + Hq) - torch.lgamma(alpha)).sum(-1)
        raw = (Hq * pp.log()).sum(-1)
        scales = (ll - ll[..., :1]).clamp(-45, 130)
        gap = (raw - raw[..., :1]).clamp(-45, 160)
        aa = (prior[None, None, None] * concentration[..., None]).clamp_min(0.0001)
        alts = (torch.lgamma(aa + Hq) - torch.lgamma(aa)).sum(-1)
        shape = (alts - ll[..., :1]).clamp(-45, 130)
        summaries = []
        for start, end in ((1, 15), (15, 27), (27, 31)):
            a = scales[..., start:end]
            b = gap[..., start:end]
            summaries.extend([a.amax(-1), torch.logsumexp(a, -1) - np.log(end - start), b.amax(-1)])
        summaries.extend([shape.amax(-1), -(mix * mix.clamp_min(1e-07).log()).sum(-1)])
        design = torch.cat([torch.asinh(scales), torch.asinh(shape), torch.asinh(torch.stack(summaries, -1))], -1)
        h0 = self.small(design).flatten(1)
        z = self.head(torch.cat([h0, self.aux(aux)], 1))[:, 0]
        if self.training:
            p, *_ = self.base(h[:, :, 2:, 0])
            query = h[:, :, 2:, 1]
            nll = -(query * p.log()).sum(-1) / query.sum(-1)
            return (z, nll.mean())
        return (z, None)

def run():
    ids, h = observations('train')
    tids, th = observations('test')
    n = len(ids)
    te = n + np.arange(len(tids))
    hist = torch.tensor(np.concatenate([h, th]), device='cpu', dtype=torch.float32)
    del h, th
    aux = np.vstack([phys('train', ids), phys('test', tids)])
    y = pd.read_csv(ROOT / 'data/train_labels.csv', dtype={'id': str}).set_index('id').loc[ids, 'label'].to_numpy()
    yy = torch.tensor(y, device='cpu', dtype=torch.float32)
    pv = np.empty(n)
    pt = []
    folder = OUT / 'predictive_density_models'
    folder.mkdir(exist_ok=True)
    for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(ids, y)):
        core, inner = train_test_split(tr, test_size=0.12, stratify=y[tr], random_state=518 + fold)
        src = OUT / 'law_outer0'
        r = np.load(src / f'fold{fold}_rows.npz')
        assert np.array_equal(r['validation_ids'], ids[va])
        arr = np.vstack([pd.read_parquet(src / f'fold{fold}_{p}.parquet').loc[ix].to_numpy() for p, ix in (('train', ids), ('test', tids))])
        x = np.column_stack([aux, np.arcsinh(arr)]).astype(np.float32)
        mu = x[core].mean(0)
        sd = x[core].std(0).clip(0.09)
        x = torch.tensor(((x - mu) / sd).clip(-7, 7), device='cpu')
        torch.manual_seed(843 + fold)
        rng = np.random.default_rng(843 + fold)
        model = EmpiricalDensity(x.shape[1]).to('cpu')
        opt = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=0.02)

        def predict(idx):
            model.eval()
            with torch.no_grad():
                return torch.cat([model(hist[j], x[j])[0].sigmoid() for j in np.array_split(idx, max(1, len(idx) // 220))]).cpu().numpy()
        best, bep = (-1, 0)
        ck = folder / f'fold{fold}.pt'
        for ep in range(40):
            model.train()
            for j in np.array_split(rng.permutation(core), max(1, len(core) // 176)):
                opt.zero_grad(set_to_none=True)
                z, loss_density = model(hist[j], x[j])
                loss = F.binary_cross_entropy_with_logits(z, yy[j]) + 0.65 * loss_density + 0.04 * model.offset.square().mean()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
            val = roc_auc_score(y[inner], predict(inner))
            if val > best + 4e-05:
                best, bep = (val, ep)
                torch.save(model.state_dict(), ck)
            if ep in (10, 21):
                for g in opt.param_groups:
                    g['lr'] *= 0.46
            if ep - bep >= 6:
                break
        model.load_state_dict(torch.load(ck, map_location='cpu', weights_only=True))
        pv[va] = predict(va)
        pt.append(predict(te))
        np.savez_compressed(folder / f'fold{fold}.npz', mu=mu, sd=sd, validation_ids=ids[va])
        print('contextual density', fold, bep, roc_auc_score(y[va], pv[va]), flush=True)
    np.savez_compressed(OUT / 'lawpredictive.npz', ids=ids, tids=tids, oof=pv, test=np.mean(pt, 0))
    print('contextual density OOF', roc_auc_score(y, pv), flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        torch.set_num_threads(1)
        run()
