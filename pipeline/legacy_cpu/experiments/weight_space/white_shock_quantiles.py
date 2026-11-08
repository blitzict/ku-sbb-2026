# CPU port; original experiments/weight_space/white_shock_quantiles.py, SHA256 2ccba4f558e2b1a8dd05cc4efb74b94787ef5c86462ba68f5a5d70039885e47b
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Learn where the innovation quantile curve moves after a large preceding shock."""
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits
from white_block_kernel import phys
from partition_evidence import ROOT, OUT
GRID = np.linspace(0.025, 0.975, 25)

def segment(x):
    center = np.median(x)
    big = abs(x - center) > np.quantile(abs(x - center), 0.67)
    group = np.arange(len(x)) * 6 // len(x)
    source, block = (big[:-1], group[1:])
    inside = group[:-1] == block
    result = []
    for r in (0, 1):
        subset = inside & (block % 2 == r)
        ref = x[1:][subset]
        base = np.quantile(ref, GRID)
        local = []
        for cond in (subset, subset & source, subset & ~source):
            sample = x[1:][cond]
            w = len(sample) / (len(sample) + 12.0)
            q = w * np.quantile(sample, GRID) + (1 - w) * base if len(sample) else base
            local.append(q)
        result.append(np.stack(local))
    return np.array(result)

def curve(x, cut):
    a, b = (segment(x[:cut]), segment(x[cut:]))
    pooled = 0.5 * (a + b)
    width = np.maximum(pooled[:, :, 18] - pooled[:, :, 6], 0.1 * x.std() + 1e-06)
    raw = (b - a) / width[:, :, None]

    def shape(side):
        scale = np.maximum(side[:, :, 18] - side[:, :, 6], 0.15 * width)
        return (side - side[:, :, 12, None]) / scale[:, :, None]
    residual = shape(b) - shape(a)
    n = len(x)
    size = np.sqrt(cut * (n - cut) / (2 * n))
    output = []
    for term in (raw, residual):
        z = np.arcsinh(size * term).clip(-5, 5)
        output.extend((z.mean(0), (z[0] * z[1]).clip(-10, 18), abs(z[0] - z[1])))
    output.append(np.arcsinh(size * (raw[:, 1] - raw[:, 2])).mean(0)[None, :])
    output.append(np.arcsinh((shape(a) + shape(b)).mean(0)))
    return np.vstack(output)

def curves(part):
    dst = OUT / f'shock_quantile_curves_{part}.npz'
    if not dst.exists():
        terms = []
        for ch in ('pulse', 'state', 'smoothvol'):
            with np.load(OUT / f'{ch}_streams_{part}.npz') as f:
                ix, x, n, b = (f[k] for k in ('ids', 'separate', 'n', 'b'))
            if terms:
                assert np.array_equal(ix, ids)
            ids = ix
            terms.append(np.stack([curve(row[:end], j) for row, end, j in zip(x, n, b)]).astype(np.float16))
        h = np.concatenate(terms, 1)
        assert np.isfinite(h).all()
        np.savez_compressed(dst, ids=ids, h=h)
    with np.load(dst) as f:
        return (f['ids'], f['h'].astype(np.float32))

class Detector(nn.Module):

    def __init__(self, d, channels):
        super().__init__()
        self.physical = nn.Sequential(nn.Linear(d, 100), nn.LayerNorm(100), nn.SiLU(), nn.Dropout(0.22), nn.Linear(100, 56), nn.SiLU())
        self.baseline = nn.Linear(56, 1)
        self.cnn = nn.Sequential(nn.Conv1d(channels, 48, 3, padding=1), nn.GroupNorm(6, 48), nn.SiLU(), nn.Conv1d(48, 32, 5, padding=2), nn.SiLU(), nn.AdaptiveAvgPool1d(5), nn.Flatten())
        self.correction = nn.Sequential(nn.Linear(160 + 56, 72), nn.LayerNorm(72), nn.SiLU(), nn.Dropout(0.27), nn.Linear(72, 1))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, x, p):
        h = self.physical(p)
        z = self.baseline(h)[:, 0]
        delta = self.correction(torch.cat([self.cnn(x), h.detach()], 1))[:, 0]
        return torch.stack([z, z.detach() + delta], 1)

def run(builder=curves, folder='shock_quantile_models', tags=('law_shock_quantile_control', 'law_shock_quantiles')):
    ids, a = builder('train')
    tids, b = builder('test')
    y = pd.read_csv(ROOT / 'data/train_labels.csv', dtype={'id': str}).set_index('id').loc[ids, 'label'].to_numpy()
    n = len(ids)
    te = n + np.arange(len(tids))
    raw = np.vstack([a, b])
    del a, b
    aux = np.vstack([phys('train', ids), phys('test', tids)])
    target = torch.tensor(y.copy(), device='cpu', dtype=torch.float32)
    dst = OUT / folder
    dst.mkdir(exist_ok=True)
    oof = np.zeros((n, 2))
    tests = []
    for f, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=29).split(ids, y)):
        record = dst / f'fold{f}.npz'
        if record.exists():
            with np.load(record) as d:
                assert np.array_equal(ids[va], d['ids'])
                oof[va] = d['pv']
                tests.append(d['pt'])
            continue
        core, inner = train_test_split(tr, test_size=0.13, stratify=y[tr], random_state=721 + f)
        mu = raw[core].mean((0, 2))[None, :, None]
        sd = raw[core].std((0, 2))[None, :, None].clip(0.15)
        path = OUT / f'law_outer29/fold{f}_rows.npz'
        with np.load(path) as d:
            assert np.array_equal(ids[tr], d['train_ids']) and np.array_equal(ids[va], d['validation_ids'])
        laws = np.vstack([pd.read_parquet(path.parent / f'fold{f}_{p}.parquet').loc[ix].to_numpy() for p, ix in (('train', ids), ('test', tids))])
        context = np.column_stack([aux, np.arcsinh(laws)])
        m = context[core].mean(0)
        s = context[core].std(0).clip(0.08)
        x = torch.tensor(((raw - mu) / sd).clip(-7, 7), device='cpu', dtype=torch.float32)
        p = torch.tensor(((context - m) / s).clip(-7, 7), device='cpu', dtype=torch.float32)
        torch.manual_seed(801 + f)
        rng = np.random.default_rng(801 + f)
        net = Detector(context.shape[1], raw.shape[1]).to('cpu')
        opt = torch.optim.AdamW(net.parameters(), lr=0.0008, weight_decay=0.026)

        def predict(ix):
            net.eval()
            with torch.no_grad(), _cpu_context.nullcontext():
                return torch.cat([net(x[j], p[j]).float().sigmoid() for j in np.array_split(ix, max(1, len(ix) // 255))]).cpu().numpy()
        best = -1.0
        best_epoch = -1
        ck = dst / f'fold{f}.pt'
        for epoch in range(32):
            net.train()
            for j in np.array_split(rng.permutation(core), max(1, len(core) // 180)):
                opt.zero_grad(set_to_none=True)
                with _cpu_context.nullcontext():
                    zz = net(x[j], p[j]).float()
                    loss = F.binary_cross_entropy_with_logits(zz, target[j, None].expand_as(zz))
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                opt.step()
            val = predict(inner)
            quality = np.mean([roc_auc_score(y[inner], t) for t in val.T])
            if quality > best + 0.0001:
                best = quality
                best_epoch = epoch
                torch.save(net.state_dict(), ck)
            if epoch in (11, 20):
                for group in opt.param_groups:
                    group['lr'] *= 0.5
            if epoch - best_epoch >= 6:
                break
        net.load_state_dict(torch.load(ck, map_location='cpu', weights_only=True))
        oof[va] = predict(va)
        tests.append(predict(te))
        np.savez_compressed(record, ids=ids[va], core=ids[core], pv=oof[va], pt=tests[-1], mu=mu, sd=sd, m=m, s=s)
        print(folder, f, best_epoch, [round(roc_auc_score(y[va], t), 6) for t in oof[va].T], flush=True)
    for k, tag in enumerate(tags):
        np.savez_compressed(OUT / (tag + '.npz'), ids=ids, tids=tids, oof=oof[:, k], test=np.mean(tests, 0)[:, k])
        print(tag, roc_auc_score(y, oof[:, k]), flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        torch.set_num_threads(1)
        run()
