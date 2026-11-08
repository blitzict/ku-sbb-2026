# CPU port; original experiments/weight_space/white_weight_ensemble.py, SHA256 e80c03d53781cab2ff39747efe1228a25ef1ce733254acfdbb1be89799e160fa
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Learn distinct multiplicative weight edits for several predictors on identical real sources.

The shared-weight architecture is inspired by TabM (Gorishniy, Kotelnikov and
Babenko, ICLR 2025, https://arxiv.org/abs/2410.24210). It is trained from
scratch using only this project's observed series and labels.
"""
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
from white_crop_growth import LAWS
from white_view_network import ranks
from partition_evidence import ROOT, OUT
VIEWS = ('', 'c1', 'c2')
K = 8

def inputs():
    out = []
    with np.load(ROOT / 'tune_out/var/mat_ns3_c7.npz', allow_pickle=True) as a:
        ids, tids = (a['ids'].astype(str), a['tid'].astype(str))
        for v in VIEWS:
            root = OUT / ('view_' + v) if v else OUT
            t = []
            for part, ix, key in (('train', ids, 'V_' + v if v else 'X'), ('test', tids, 'VT_' + v if v else 'Xt')):
                cols = [a[key]]
                for tag, fields in LAWS.items():
                    cols.append(pd.read_parquet(root / f'{tag}_{part}.parquet', columns=fields.split()).loc[ix].to_numpy())
                for tag in ('bounds', 'trend', 'phase_law', 'state'):
                    tab = pd.read_parquet(root / f'{tag}_{part}.parquet').loc[ix]
                    cols.append(tab.to_numpy())
                t.append(np.column_stack(cols).astype(np.float32))
            out.append(np.vstack(t))
    return (ids, tids, np.stack(out, axis=1))

class SharedWeights(nn.Module):

    def __init__(self, i, o, first=False):
        super().__init__()
        self.proj = nn.Linear(i, o, bias=False)
        scale = torch.empty(K, i).normal_(1, 0.1)
        if first:
            scale = torch.randint(0, 2, (K, i // 5, 1)).float().mul_(2).sub_(1).expand(K, i // 5, 5).reshape(K, i).clone()
        self.input = nn.Parameter(scale)
        self.output = nn.Parameter(torch.empty(K, o).normal_(1, 0.08))
        self.bias = nn.Parameter(torch.zeros(K, o))

    def forward(self, x):
        return self.proj(x * self.input) * self.output + self.bias

class Network(nn.Module):

    def __init__(self, d):
        super().__init__()
        h = 128
        self.register_buffer('knots', torch.linspace(-0.96, 0.96, 5))
        self.start = SharedWeights(5 * d, h, True)
        self.mid = SharedWeights(h, h)
        self.last = SharedWeights(h, h)
        self.layers = nn.ModuleList([nn.LayerNorm(h) for _ in range(3)])
        self.read = nn.Parameter(torch.randn(K, h) * 0.035)
        self.residual = nn.Parameter(torch.zeros(K, h))
        self.condition = nn.Linear(4, h)
        self.offset = nn.Parameter(torch.zeros(K))

    def forward(self, x, z):
        spline = ((x[:, :, :, None] - self.knots) * 2.15 + 0.5).clamp(0, 1) - 0.5
        h = F.dropout(F.silu(self.layers[0](self.start(spline.flatten(2)))), 0.18, self.training)
        h = h + 0.7 * F.dropout(F.silu(self.layers[1](self.mid(h))), 0.22, self.training)
        h = h + 0.7 * F.dropout(F.silu(self.layers[2](self.last(h))), 0.18, self.training)
        standalone = (h * self.read).sum(2) + self.offset
        pre = torch.stack([torch.tanh(z / 2), torch.tanh(z), torch.exp(-0.5 * (z + 1) ** 2), torch.exp(-0.5 * (z - 1) ** 2)], 1)
        adapt = (h * (1 + 0.45 * self.condition(pre).tanh()[:, None, :]) * self.residual).sum(2)
        return (standalone, z[:, None] + adapt)

def run(seed):
    ids, tids, raw = inputs()
    n = len(ids)
    te = n + np.arange(len(tids))
    y = pd.read_csv(ROOT / 'data/train_labels.csv', dtype={'id': str}).set_index('id').loc[ids, 'label'].to_numpy()
    yy = torch.tensor(y.copy(), device='cpu', dtype=torch.float32)
    out = np.empty((n, 2))
    tests = []
    name = f'law_weight_ensemble_s{seed}'
    dst = OUT / (name + '_models')
    dst.mkdir(exist_ok=True)
    cache = OUT / 'frozen_local_rank_family_agreement'
    with np.load(cache / 'full.npz') as a:
        zt = a['test']
    for f, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(ids, y)):
        ck = dst / f'fold{f}.pt'
        record = dst / f'fold{f}.npz'
        if record.exists():
            with np.load(record) as a:
                assert np.array_equal(a['validation_ids'], ids[va])
                out[va] = a['pv']
                tests.append(a['pt'])
            continue
        core, inner = train_test_split(tr, test_size=0.12, stratify=y[tr], random_state=933 + f)
        z = np.empty(n + len(tids))
        with np.load(cache / f's{seed}_f{f}.npz') as a:
            assert np.array_equal(ids[tr], a['train_ids']) and np.array_equal(ids[va], a['validation_ids'])
            z[tr], z[va], z[te] = (a['oof_train'], a['valid'], zt)
        offsets = torch.tensor(z, device='cpu', dtype=torch.float32)
        prior = []
        for view in VIEWS:
            path = (OUT / ('view_' + view) if view else OUT) / f'law_outer{seed}'
            with np.load(path / f'fold{f}_rows.npz') as a:
                assert np.array_equal(ids[va], a['validation_ids'])
            prior.append(np.vstack([pd.read_parquet(path / f'fold{f}_{part}.parquet').loc[ix].to_numpy(np.float32) for part, ix in (('train', ids), ('test', tids))]))
        arr, knots = ranks(np.concatenate([raw, np.stack(prior, 1)], axis=2), core)
        x = torch.as_tensor(arr, device='cpu', dtype=torch.float32)
        del arr
        torch.manual_seed(6017 + f)
        rng = np.random.default_rng(6017 + f)
        net = Network(x.shape[2]).to('cpu')
        opt = torch.optim.AdamW(net.parameters(), lr=0.0008, weight_decay=0.028)

        def predict(ix):
            net.eval()
            val = []
            with torch.no_grad(), _cpu_context.nullcontext():
                for j in np.array_split(ix, max(1, len(ix) // 240)):
                    p = []
                    for v in range(3):
                        h = x[j, v][:, None, :].expand(-1, K, -1)
                        p.append(torch.stack([w.float().sigmoid().mean(1) for w in net(h, offsets[j])], 1))
                    val.append((p[0] + 0.13 * p[1] + 0.13 * p[2]) / 1.26)
            return torch.cat(val).cpu().numpy()
        best = -1
        bep = 0
        for ep in range(32):
            net.train()
            for j in np.array_split(rng.permutation(core), max(1, len(core) // 176)):
                choice = rng.choice(3, (len(j), K), p=[0.63, 0.185, 0.185])
                h = x[j[:, None], choice]
                opt.zero_grad(set_to_none=True)
                with _cpu_context.nullcontext():
                    v, r = net(h, offsets[j])
                    label = yy[j, None].expand_as(v)
                    loss = 0.61 * F.binary_cross_entropy_with_logits(v, label) + 0.39 * F.binary_cross_entropy_with_logits(r, label)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                opt.step()
            v = predict(inner)
            sc = 0.65 * roc_auc_score(y[inner], v[:, 0]) + 0.35 * roc_auc_score(y[inner], v[:, 1])
            if sc > best + 7e-05:
                best, bep = (sc, ep)
                torch.save(net.state_dict(), ck)
            if ep in (10, 19):
                for q in opt.param_groups:
                    q['lr'] *= 0.42
            if ep - bep >= 7:
                break
        net.load_state_dict(torch.load(ck, map_location='cpu', weights_only=True))
        out[va] = predict(va)
        tests.append(predict(te))
        np.savez_compressed(record, validation_ids=ids[va], pv=out[va], pt=tests[-1], knots=knots)
        print(name, f, bep, [round(roc_auc_score(y[va], out[va, k]), 6) for k in (0, 1)], flush=True)
    for k, suffix in enumerate(('', '_res')):
        np.savez_compressed(OUT / (name + suffix + '.npz'), ids=ids, tids=tids, oof=out[:, k], test=np.mean(tests, 0)[:, k])
    print(name, [roc_auc_score(y, out[:, k]) for k in (0, 1)], flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('seed', type=int)
    a = p.parse_args()
    with threadpool_limits(limits=1):
        torch.set_num_threads(1)
        run(a.seed)
