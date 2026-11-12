# CPU port; original experiments/weight_space/white_stable_factors.py, SHA256 c7395ca26b35815cf6df67f29a9b7474e47a02d6a42306a425a2514a6a0d4e0f
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Learn repeatable segment properties from disjoint blocks of the actual series."""
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
from white_patch_forecast import make
from white_block_kernel import phys
from partition_evidence import ROOT, OUT

class Factors(nn.Module):

    def __init__(self, d, aux):
        super().__init__()
        self.patch = nn.Sequential(nn.Linear(d, 96), nn.LayerNorm(96), nn.SiLU(), nn.Linear(96, 64))
        self.side = nn.Sequential(nn.Linear(d + 64, 96), nn.LayerNorm(96), nn.SiLU(), nn.Linear(96, 32))
        self.physics = nn.Sequential(nn.Linear(aux, 128), nn.LayerNorm(128), nn.SiLU(), nn.Dropout(0.16), nn.Linear(128, 72), nn.LayerNorm(72), nn.SiLU(), nn.Dropout(0.12))
        self.base = nn.Linear(72, 1)
        self.read = nn.Sequential(nn.Linear(32 * 7 + 72, 128), nn.LayerNorm(128), nn.SiLU(), nn.Dropout(0.2), nn.Linear(128, 64), nn.SiLU(), nn.Dropout(0.15), nn.Linear(64, 1))
        nn.init.zeros_(self.read[-1].weight)
        nn.init.zeros_(self.read[-1].bias)

    def forward(self, x, aux):
        b = len(x)
        h = torch.cat([x, self.patch(x)], -1)
        groups = self.side(h.reshape(b, 2, 3, 4, -1).mean(2))
        avg = groups.mean(2)
        var = groups.var(2, unbiased=True)
        se = (0.25 * var.sum(1) + 0.025).sqrt()
        delta = groups[:, 1] - groups[:, 0]
        center = delta.mean(1)
        concordance = (delta.sum(1).square() - delta.square().sum(1)) / (12 * se.square())
        physics = self.physics(aux)
        base = self.base(physics)[:, 0]
        summary = torch.cat([avg[:, 0], avg[:, 1], center, torch.asinh(center / se), torch.asinh(concordance), (var[:, 1] + 0.025).log() - (var[:, 0] + 0.025).log(), se.log(), physics.detach()], 1)
        return (base, base.detach() + self.read(summary)[:, 0], groups)

def consistency(groups, targets):
    center = groups.mean(2)
    noise = (groups - center[:, :, None]).square().mean()
    no_break = 1 - targets
    stay = ((center[:, 0] - center[:, 1]).square().mean(1) * no_break).sum() / no_break.sum().clamp_min(1)
    z = center.reshape(-1, center.shape[-1])
    z = z - z.mean(0)
    cov = z.T @ z / (len(z) - 1)
    sd = cov.diagonal().clamp_min(0.001).sqrt()
    off = cov - torch.diag_embed(cov.diagonal())
    return noise + 0.4 * stay + 1.4 * (0.8 - sd).clamp_min(0).mean() + 0.04 * off.square().sum() / len(sd)

def run():
    ids, patch = make('train')
    tids, test = make('test')
    assert patch.shape[1:3] == (2, 12)
    patch = np.concatenate([patch, test]).astype(np.float32)
    n = len(ids)
    yt = pd.read_csv(ROOT / 'data/train_labels.csv', dtype={'id': str}).set_index('id').loc[ids, 'label'].to_numpy()
    y = torch.tensor(yt, dtype=torch.float32, device='cpu')
    static = np.vstack([phys('train', ids), phys('test', tids)])
    folder = OUT / 'stable_factor_models'
    folder.mkdir(exist_ok=True)
    oof = np.empty((n, 2))
    tests = []
    for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(ids, yt)):
        checkpoint = folder / f'fold{fold}.pt'
        record = folder / f'fold{fold}.npz'
        if record.exists():
            with np.load(record) as s:
                assert np.array_equal(s['ids'], ids[va]) and np.array_equal(s['tids'], tids)
                oof[va] = s['pv']
                tests.append(s['pt'])
            continue
        core, inner = train_test_split(tr, test_size=0.11, stratify=yt[tr], random_state=1061 + fold)
        rng = np.random.default_rng(723 + fold)
        torch.manual_seed(723 + fold)
        prior = OUT / 'law_outer0'
        with np.load(prior / f'fold{fold}_rows.npz') as rows:
            assert np.array_equal(rows['validation_ids'], ids[va])
        extras = np.vstack([pd.read_parquet(prior / f'fold{fold}_{part}.parquet').loc[ix].to_numpy() for part, ix in (('train', ids), ('test', tids))])
        tab = np.column_stack([static, np.arcsinh(extras)])
        tab = (tab - tab[core].mean(0)) / tab[core].std(0).clip(0.09)
        aux = torch.as_tensor(tab.clip(-7, 7), dtype=torch.float32, device='cpu')
        mu = patch[core].mean((0, 1, 2))
        std = patch[core].std((0, 1, 2)).clip(0.04)
        x = torch.as_tensor(((patch - mu) / std).clip(-7, 7), device='cpu')
        m = Factors(x.shape[-1], aux.shape[1]).to('cpu')
        opt = torch.optim.AdamW(m.parameters(), lr=0.0008, weight_decay=0.028)

        def predict(ix):
            m.eval()
            with torch.no_grad():
                return torch.cat([torch.stack(m(x[j], aux[j])[:2], 1) for j in np.array_split(ix, max(1, len(ix) // 256))]).cpu().numpy()
        best, epoch_best = (-1, 0)
        for ep in range(32):
            m.train()
            for ix in np.array_split(rng.permutation(core), max(1, len(core) // 224)):
                opt.zero_grad(set_to_none=True)
                old, new, g = m(x[ix], aux[ix])
                loss = F.binary_cross_entropy_with_logits(old, y[ix]) + F.binary_cross_entropy_with_logits(new, y[ix]) + (0.65 if ep < 4 else 0.23) * consistency(g, y[ix])
                loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), 2.0)
                opt.step()
            pred = predict(inner)
            scores = [roc_auc_score(yt[inner], p) for p in pred.T]
            quality = np.mean(scores)
            if quality > best + 8e-05:
                best, epoch_best = (quality, ep)
                torch.save(m.state_dict(), checkpoint)
            if ep in (10, 19):
                for group in opt.param_groups:
                    group['lr'] *= 0.55
            if ep >= 10 and ep - epoch_best >= 6:
                break
        m.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        pv, pt = (predict(va), predict(n + np.arange(len(tids))))
        np.savez_compressed(record, ids=ids[va].astype(str), tids=tids.astype(str), pv=pv, pt=pt)
        oof[va] = pv
        tests.append(pt)
        print(fold, 'physical vs with repeatable factors', [round(roc_auc_score(yt[va], s), 6) for s in pv.T], flush=True)
    test = np.mean(tests, 0)
    for part, ix, values in (('train', ids, oof), ('test', tids, test)):
        frame = pd.DataFrame(np.column_stack([values, values[:, 1] - values[:, 0]]), index=pd.Index(ix, name='id'), columns=['physical', 'combined', 'stable_increment'])
        assert frame.index.is_unique and np.isfinite(frame).all().all()
        frame.to_parquet(OUT / f'stable_factors_{part}.parquet')
    print('full paired', [roc_auc_score(yt, s) for s in oof.T], flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        torch.set_num_threads(1)
        run()
