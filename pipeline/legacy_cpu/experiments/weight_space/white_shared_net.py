# CPU port; original experiments/weight_space/white_shared_net.py, SHA256 2311cbe6f4598f71f09e27a7f3b3abb506322ec1389ad1ce1c8c2a93f9c5bed2
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Shared-weight tabular ensemble on observed physics, following the TabM idea."""
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
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from scipy.special import ndtri
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits
from white_mech_tree import extra
from shape_bank import matrix
from partition_evidence import OUT, ROOT

def inputs(part):
    x = matrix(part, 'state')
    terms = [x, extra(part, x.index, 'both')]
    for n in ('walk_jump', 'conditional_scores', 'trend_population'):
        terms.append(pd.read_parquet(OUT / f'{n}_{part}.parquet').loc[x.index].add_prefix(n))
    return pd.concat(terms, axis=1)

def normalize(x, train):
    prob = np.linspace(0.002, 0.998, 95)
    knots = np.nanquantile(x[train], prob, axis=0)
    result = np.zeros(x.shape, np.float32)
    for i, q in enumerate(knots.T):
        unique, rev, counts = np.unique(q, return_inverse=True, return_counts=True)
        if len(unique) < 2 or not np.isfinite(unique).all():
            continue
        level = np.bincount(rev, weights=prob) / counts
        v = x[:, i]
        ok = np.isfinite(v)
        result[ok, i] = ndtri(np.interp(v[ok], unique, level))
    return (np.ascontiguousarray(result), knots)

class SharedLayer(nn.Module):

    def __init__(self, indim, outdim, k, first):
        super().__init__()
        self.weight = nn.Linear(indim, outdim, bias=False)
        r = torch.randint(0, 2, (k, indim)).float() * 2 - 1 if first else torch.ones(k, indim)
        self.input_scale = nn.Parameter(r)
        self.output_scale = nn.Parameter(torch.ones(k, outdim))
        self.bias = nn.Parameter(torch.zeros(k, outdim))

    def forward(self, x):
        return self.weight(x * self.input_scale) * self.output_scale + self.bias

class Net(nn.Module):

    def __init__(self, dim, k=8, width=160):
        super().__init__()
        self.blocks = nn.ModuleList([SharedLayer(dim, width, k, True), SharedLayer(width, width, k, False), SharedLayer(width, width, k, False)])
        self.norm = nn.ModuleList([nn.LayerNorm(width) for _ in self.blocks])
        self.drop = nn.Dropout(0.16)
        self.readout = nn.Parameter(torch.randn(k, width) / np.sqrt(width))
        self.bias = nn.Parameter(torch.zeros(k))

    def forward(self, x):
        x = x[:, None, :]
        for layer, norm in zip(self.blocks, self.norm):
            x = self.drop(F.gelu(norm(layer(x))))
        return (x * self.readout).sum(-1) + self.bias

@torch.no_grad()
def predict(model, x, ix):
    model.eval()
    vals = []
    for start in range(0, len(ix), 384):
        v = x[torch.as_tensor(ix[start:start + 384], device='cpu')]
        with _cpu_context.nullcontext():
            vals.append(model(v).float().sigmoid().mean(1))
    return torch.cat(vals).cpu().numpy()

def run():
    torch.set_num_threads(1)
    a, b = (inputs('train'), inputs('test'))
    assert list(a.columns) == list(b.columns)
    ids, tids = (a.index, b.index)
    y = pd.read_csv(ROOT / 'data/train_labels.csv', dtype={'id': str}).set_index('id').loc[ids, 'label'].to_numpy()
    full = pd.concat([a, b]).to_numpy(np.float32)
    del a, b
    te = len(y) + np.arange(len(tids))
    oof = np.full(len(y), np.nan)
    tests = []
    folder = OUT / 'lawshared_models'
    folder.mkdir(exist_ok=True)
    yy = torch.as_tensor(y, device='cpu', dtype=torch.float32)
    for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(ids, y)):
        part = folder / f'fold{fold}.npz'
        ck = folder / f'fold{fold}.pt'
        if part.exists():
            with np.load(part) as r:
                assert np.array_equal(r['va'], va)
                oof[va] = r['pv']
                tests.append(r['pt'])
            continue
        core, inner = train_test_split(tr, test_size=0.12, stratify=y[tr], random_state=701 + fold)
        torch.manual_seed(451 + fold)
        rng = np.random.default_rng(451 + fold)
        prior = OUT / 'law_outer0'
        with np.load(prior / f'fold{fold}_rows.npz') as r:
            assert np.array_equal(r['validation_ids'], ids[va])
        h = np.vstack([pd.read_parquet(prior / f'fold{fold}_{p}.parquet').loc[ix].to_numpy(dtype=np.float32) for p, ix in (('train', ids), ('test', tids))])
        design, knots = normalize(np.column_stack([full, h]), core)
        x = torch.as_tensor(design, device='cpu')
        del design
        net = Net(x.shape[1]).to('cpu')
        opt = torch.optim.AdamW(net.parameters(), lr=0.00065, weight_decay=0.014)
        best, bep = (-1, -1)
        for ep in range(34):
            net.train()
            order = rng.permutation(core)
            for start in range(0, len(core), 256):
                ix = torch.as_tensor(order[start:start + 256], device='cpu')
                opt.zero_grad(set_to_none=True)
                with _cpu_context.nullcontext():
                    logits = net(x[ix]).float()
                    loss = F.binary_cross_entropy_with_logits(logits, yy[ix, None].expand_as(logits))
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                opt.step()
            auc = roc_auc_score(y[inner], predict(net, x, inner))
            if auc > best + 5e-05:
                best, bep = (auc, ep)
                torch.save(net.state_dict(), ck)
            if ep - bep >= 5:
                break
            if ep in (11, 23):
                for group in opt.param_groups:
                    group['lr'] *= 0.5
        net.load_state_dict(torch.load(ck, map_location='cpu', weights_only=True))
        pv, pt = (predict(net, x, va), predict(net, x, te))
        np.savez_compressed(part, va=va, pv=pv, pt=pt, knots=knots, best_epoch=bep)
        print('shared', fold, bep, round(best, 5), round(roc_auc_score(y[va], pv), 6), flush=True)
        oof[va] = pv
        tests.append(pt)
    np.savez_compressed(OUT / 'lawshared.npz', ids=ids.to_numpy(dtype=str), tids=tids.to_numpy(dtype=str), oof=oof, test=np.mean(tests, axis=0))
    print('shared out of fold', roc_auc_score(y, oof), flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
