# CPU port; original experiments/weight_space/white_fourier_kernel.py, SHA256 020d6fff1db4463c2adf41df37089206ebb87bf72ac80e61d38d09dd9488b17e
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Before/after characteristic functions of real innovations at several time scales."""
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
from scipy.ndimage import uniform_filter1d
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from white_block_kernel import phys
from partition_evidence import OUT, ROOT
FREQ = np.array([0.32, 0.58, 0.95, 1.45, 2.1, 2.9, 3.95, 5.7], np.float32)
NB = 8

def norm(x):
    lo, hi = np.quantile(x, [0.25, 0.75], axis=0, keepdims=True)
    return ((x - (lo + hi) / 2) / np.maximum.reduce([(hi - lo) / 1.349, 0.18 * x.std(0, keepdims=True), np.full_like(lo, 1e-05)])).clip(-35, 35)

def side(z, e):

    def lag(x, n):
        return np.r_[np.repeat(x[0], n), x[:-n]]
    r = norm(e)
    a = norm(z)
    v = [z, e, r, a, np.diff(z, prepend=z[0]), e - lag(e, 1), e + 0.8 * lag(e, 1), e + lag(e, 1) - 2 * lag(e, 2), np.abs(r) * np.abs(lag(r, 1)), np.tanh(r) * np.tanh(lag(r, 1)), r * lag(r, 2)]
    for w in (3, 8, 21):
        low = uniform_filter1d(z, w, mode='reflect')
        high = z - low
        v.extend([low, high, uniform_filter1d(np.abs(r), w, mode='reflect'), low - lag(low, w)])
    return np.column_stack(v)

def plot(z, e, b):
    v = norm(np.vstack([side(z[:b], e[:b]), side(z[b:], e[b:])])).astype(np.float32)
    r = v[:, :, None] * FREQ
    u = np.concatenate([np.cos(r), np.sin(r)], axis=2)
    edges = np.r_[np.arange(NB) * b // NB, b + np.arange(NB) * (len(v) - b) // NB, len(v)]
    lengths = np.diff(edges)
    h = (np.add.reduceat(u, edges[:-1], axis=0) / lengths[:, None, None]).reshape(2, NB, v.shape[1], -1)
    a, c = h.mean(1)
    d = c - a
    centered = h - h.mean(1, keepdims=True)
    byblock = np.einsum('tbgd,tbge->gde', centered, centered) / (NB * (NB - 1))
    point = []
    for sl in (u[:b], u[b:]):
        x = sl - sl.mean(0)
        point.append(np.einsum('tgd,tge->gde', x, x, optimize=True) / (len(x) * (len(x) - 1)))
    cov = 0.6 * byblock + 0.4 * sum(point)
    n = cov.shape[1]
    diag = np.diagonal(cov, axis1=1, axis2=2).clip(4e-07)
    cov = 0.72 * cov + 0.28 * diag[:, :, None] * np.eye(n)[None] + 3e-06 * np.eye(n)
    L = np.linalg.cholesky(cov)
    white = np.linalg.solve(L, d[:, :, None])[:, :, 0]
    se = np.sqrt(diag)
    near = h[1, :2].mean(0) - h[0, -2:].mean(0)
    far = h[1, -2:].mean(0) - h[0, :2].mean(0)
    arrays = [a, c, np.arcsinh(d / se), np.arcsinh(white), np.log(se).clip(-7, 2), np.arcsinh(near / (2 * se)), np.arcsinh(far / (2 * se))]
    return np.concatenate(arrays, 1).astype(np.float32)

def load(part):
    path = OUT / f'fourier_law_{part}.npz'
    if not path.exists():
        with np.load(OUT / f'pulse_streams_{part}.npz') as s:
            ids, z, n, b = (s[k] for k in ('ids', 'separate', 'n', 'b'))
        with np.load(OUT / f'state_streams_{part}.npz') as s:
            assert np.array_equal(s['ids'], ids)
            e = s['separate']
        vals = []
        for i in range(len(ids)):
            vals.append(plot(z[i, :n[i]], e[i, :n[i]], b[i]))
            if (i + 1) % 6000 == 0:
                print(part, i + 1, 'Fourier rows', flush=True)
        v = np.stack(vals).astype(np.float16)
        assert np.isfinite(v).all()
        np.savez(path, ids=ids, v=v)
    with np.load(path) as s:
        return (s['ids'], s['v'])

class ContrastNet(nn.Module):

    def __init__(self, groups, n, d):
        super().__init__()
        self.encode = nn.Sequential(nn.Linear(n, 80), nn.LayerNorm(80), nn.SiLU(), nn.Dropout(0.14), nn.Linear(80, 48))
        self.kind = nn.Parameter(torch.randn(groups, 48) * 0.15)
        self.gate = nn.Sequential(nn.Linear(d, 64), nn.LayerNorm(64), nn.SiLU(), nn.Dropout(0.15), nn.Linear(64, 48))
        self.read = nn.Sequential(nn.Linear(groups * 12 + 96, 128), nn.LayerNorm(128), nn.SiLU(), nn.Dropout(0.21), nn.Linear(128, 1))
        self.local = nn.Linear(48, 12)

    def forward(self, x, c):
        v = self.encode(x) + self.kind
        h = self.gate(c)
        a = F.softmax((v * h[:, None]).sum(2) / 6.93, 1)
        return self.read(torch.cat([self.local(v).flatten(1), (a[:, :, None] * v).sum(1), h], 1))[:, 0]

def run():
    ids, a = load('train')
    tids, b = load('test')
    n = len(ids)
    te = n + np.arange(len(tids))
    p = np.vstack([phys('train', ids), phys('test', tids)])
    y = pd.read_csv(ROOT / 'data/train_labels.csv', dtype={'id': str}).set_index('id').loc[ids, 'label'].to_numpy()
    labels = torch.tensor(y, device='cpu', dtype=torch.float32)
    v = np.concatenate([a, b])
    del a, b
    pv = np.empty(n)
    pt = []
    dest = OUT / 'fourier_law_models'
    dest.mkdir(exist_ok=True)
    for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(ids, y)):
        core, inner = train_test_split(tr, test_size=0.12, stratify=y[tr], random_state=761 + fold)
        f = np.vstack([pd.read_parquet(OUT / f'law_outer0/fold{fold}_{part}.parquet').loc[ix].to_numpy() for part, ix in (('train', ids), ('test', tids))])
        aux = np.column_stack([p, np.arcsinh(f)]).astype(np.float32)
        mu = aux[core].mean(0)
        sd = aux[core].std(0).clip(0.09)
        raw = v[core].astype(np.float32)
        center = raw.mean(0)
        dev = raw.std(0).clip(0.06)
        del raw
        x = torch.tensor(((v - center) / dev).clip(-10, 10), device='cpu', dtype=torch.float32)
        c = torch.tensor(((aux - mu) / sd).clip(-8, 8), device='cpu', dtype=torch.float32)
        torch.manual_seed(338 + fold)
        rng = np.random.default_rng(338 + fold)
        model = ContrastNet(x.shape[1], x.shape[2], c.shape[1]).to('cpu')
        opt = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=0.02)

        def predict(idx):
            model.eval()
            with torch.no_grad(), _cpu_context.nullcontext():
                return torch.cat([model(x[j], c[j]).float().sigmoid() for j in np.array_split(idx, max(1, len(idx) // 400))]).cpu().numpy()
        best, bep = (-1, -1)
        path = dest / f'fold{fold}.pt'
        for ep in range(50):
            model.train()
            for j in np.array_split(rng.permutation(core), max(1, len(core) // 208)):
                opt.zero_grad(set_to_none=True)
                with _cpu_context.nullcontext():
                    loss = F.binary_cross_entropy_with_logits(model(x[j], c[j]).float(), labels[j])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
            score = roc_auc_score(y[inner], predict(inner))
            if score > best + 5e-05:
                best, bep = (score, ep)
                torch.save(model.state_dict(), path)
            if ep in (11, 22):
                for g in opt.param_groups:
                    g['lr'] *= 0.46
            if ep - bep >= 7:
                break
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        pv[va] = predict(va)
        pt.append(predict(te))
        np.savez(dest / f'fold{fold}.npz', center=center, dev=dev, mu=mu, sd=sd, validation_ids=ids[va])
        print('fourier', fold, bep, roc_auc_score(y[va], pv[va]), flush=True)
    np.savez_compressed(OUT / 'lawfourier.npz', ids=ids, tids=tids, oof=pv, test=np.mean(pt, 0))
    print('Fourier OOF', roc_auc_score(y, pv), flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        torch.set_num_threads(1)
        run()
