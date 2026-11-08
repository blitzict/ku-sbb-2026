# CPU port; original experiments/weight_space/white_view_network.py, SHA256 b9cb98a61b13b527b14a322ade2d1ab69aa02e4dac6612aa6786326a98d84652
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Estimate one break probability from grouped, genuinely observed views."""
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
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits
from partition_evidence import ROOT, OUT
TABLES = ('ar_law', 'bounds', 'curve', 'noise_tests', 'partition', 'periodic', 'phase_law', 'trend', 'innov_v2', 'mixture_law')

def features(views):
    rows = []
    with np.load(ROOT / 'tune_out/var/mat_ns3_c7.npz', allow_pickle=True) as q:
        ids, tids = (q['ids'].astype(str), q['tid'].astype(str))
        for view in ('', *views):
            root = OUT / ('view_' + view) if view else OUT
            if view:
                assert (root / 'READY').exists(), root
            a = []
            for part, ix, k in (('train', ids, 'V_' if view else 'X'), ('test', tids, 'VT_' if view else 'Xt')):
                terms = [q[k + view].astype(np.float32)]
                for tag in TABLES:
                    df = pd.read_parquet(root / f'{tag}_{part}.parquet')
                    assert np.array_equal(df.index, ix)
                    terms.append(df.to_numpy(dtype=np.float32))
                for stream in ('pulse', 'state'):
                    h = pd.read_parquet(OUT / f'jump_law_{stream}{view}_{part}.parquet').loc[ix]
                    terms.append(h.to_numpy(dtype=np.float32))
                a.append(np.column_stack(terms))
            rows.append(np.vstack(a))
    return (ids, tids, np.stack(rows, axis=1))

def ranks(raw, core):
    p = np.linspace(0.002, 0.998, 47)
    q = np.nanquantile(raw[core, 0, :], p, axis=0)
    out = np.zeros_like(raw, dtype=np.float16)
    for k, knots in enumerate(q.T):
        knots = knots[np.isfinite(knots)]
        v, i, c = np.unique(knots, return_inverse=True, return_counts=True)
        if len(v) < 2:
            continue
        levels = np.bincount(i, weights=p[:len(knots)]) / c
        arr = raw[:, :, k].copy()
        arr = np.where(np.isfinite(arr), arr, np.median(v))
        out[:, :, k] = (2 * np.interp(arr, v, levels) - 1).astype(np.float16)
    return (out, q)

class GroupedViews(nn.Module):

    def __init__(self, d, views):
        super().__init__()
        widths = 8
        self.register_buffer('cuts', torch.linspace(-0.93, 0.93, widths))
        self.proj = nn.Linear(d * widths, 196, bias=False)
        self.skip = nn.Linear(d, 196)
        self.views = nn.Parameter(torch.randn(views, 196) * 0.03)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(196), nn.Linear(196, 268), nn.GELU(), nn.Dropout(0.23), nn.Linear(268, 196), nn.Dropout(0.11)) for _ in range(2)])
        self.aux = nn.Sequential(nn.LayerNorm(196), nn.Linear(196, 1))
        self.compare = nn.Sequential(nn.LayerNorm(196 * 3), nn.Linear(196 * 3, 112), nn.GELU(), nn.Dropout(0.23), nn.Linear(112, 1))
        nn.init.zeros_(self.compare[-1].weight)
        nn.init.zeros_(self.compare[-1].bias)

    def forward(self, x):
        encoded = ((x[:, :, :, None] - self.cuts) * 4.1 + 0.5).clamp(0, 1) - 0.5
        h = self.proj(encoded.flatten(2)) + self.skip(x) + self.views
        for layer in self.blocks:
            h = h + layer(h)
        aux = self.aux(h)[:, :, 0]
        if self.training:
            weights = (torch.rand(h.shape[:2], device=h.device) > 0.22).float()
            weights[:, 0] = 1
        else:
            weights = torch.ones_like(aux)
        weights = weights / weights.sum(1, keepdim=True)
        pooled = (weights[:, :, None] * h).sum(1)
        scatter = ((h - pooled[:, None]).abs() * weights[:, :, None]).sum(1)
        correction = self.compare(torch.cat([h[:, 0], pooled, scatter], 1))[:, 0]
        return (correction + 0.5 * aux[:, 0] + 0.5 * (weights * aux).sum(1), aux)

def run(views):
    ids, tids, raw = features(views)
    n = len(ids)
    te = n + np.arange(len(tids))
    y = pd.read_csv(ROOT / 'data/train_labels.csv', dtype={'id': str}).set_index('id').loc[ids, 'label'].to_numpy()
    yy = torch.tensor(y, device='cpu', dtype=torch.float32)
    oof = np.empty(n)
    pt = []
    name = 'viewnet_' + '_'.join(views)
    folder = OUT / (name + '_models')
    folder.mkdir(exist_ok=True)
    for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(ids, y)):
        ck = folder / f'fold{fold}.pt'
        record = folder / f'fold{fold}.npz'
        if record.exists():
            with np.load(record) as a:
                assert np.array_equal(a['validation_ids'], ids[va])
                oof[va] = a['pv']
                pt.append(a['pt'])
            continue
        core, inner = train_test_split(tr, test_size=0.11, stratify=y[tr], random_state=891 + fold)
        views_law = []
        for v in ('', *views):
            d = OUT / ('view_' + v) if v else OUT
            p = d / 'law_outer0'
            with np.load(p / f'fold{fold}_rows.npz') as k:
                assert np.array_equal(k['validation_ids'], ids[va]) and np.array_equal(k['train_ids'], ids[tr])
            view = np.vstack([pd.read_parquet(p / f'fold{fold}_{part}.parquet').loc[ix].to_numpy(dtype=np.float32) for part, ix in (('train', ids), ('test', tids))])
            views_law.append(view)
        values, knots = ranks(np.concatenate([raw, np.stack(views_law, 1)], axis=2), core)
        x = torch.tensor(values, device='cpu', dtype=torch.float32)
        del values
        torch.manual_seed(297 + fold)
        rng = np.random.default_rng(297 + fold)
        model = GroupedViews(x.shape[2], x.shape[1]).to('cpu')
        opt = torch.optim.AdamW(model.parameters(), lr=0.00063, weight_decay=0.025)

        def predict(idx):
            model.eval()
            with torch.no_grad(), _cpu_context.nullcontext():
                return torch.cat([model(x[j])[0].float().sigmoid() for j in np.array_split(idx, max(1, len(idx) // 300))]).cpu().numpy()
        best, bep = (-1, -1)
        for ep in range(42):
            model.train()
            for j in np.array_split(rng.permutation(core), max(1, len(core) // 180)):
                opt.zero_grad(set_to_none=True)
                with _cpu_context.nullcontext():
                    z, s = model(x[j])
                    label = yy[j]
                    loss = F.binary_cross_entropy_with_logits(z.float(), label) + 0.38 * F.binary_cross_entropy_with_logits(s.float(), label[:, None].expand_as(s))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.7)
                opt.step()
            score = roc_auc_score(y[inner], predict(inner))
            if score > best + 4e-05:
                best, bep = (score, ep)
                torch.save(model.state_dict(), ck)
            if ep in (9, 21):
                for g in opt.param_groups:
                    g['lr'] *= 0.46
            if ep - bep >= 7:
                break
        model.load_state_dict(torch.load(ck, map_location='cpu', weights_only=True))
        oof[va] = predict(va)
        pt.append(predict(te))
        np.savez_compressed(record, validation_ids=ids[va], tr_ids=ids[tr], pv=oof[va], pt=pt[-1], knots=knots)
        print(name, fold, bep, roc_auc_score(y[va], oof[va]), flush=True)
    np.savez_compressed(OUT / (name + '.npz'), ids=ids, tids=tids, oof=oof, test=np.mean(pt, 0))
    print(name, 'out of fold', roc_auc_score(y, oof), flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('views', nargs='+', choices=('c1', 'c2', 'rev'))
    with threadpool_limits(limits=1):
        torch.set_num_threads(1)
        run(p.parse_args().views)
