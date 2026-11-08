# CPU port; original experiments/weight_space/white_conditional_weights.py, SHA256 ba52fc78f699e1bbbe5a903acf7043fe988cfa7a30178da3a0d63263f1b3c35e
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Add context-dependent low-rank task vectors to a frozen sequence checkpoint."""
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
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from white_physical_net import signals, prepare_input
from white_block_kernel import phys
from sequence_edits import seq
from partition_evidence import ROOT, OUT

def streams(part):
    ids, raw, n, b = signals(part)
    with np.load(OUT / f'hetero_path_{part}.npz') as a:
        assert np.array_equal(ids, a['ids']) and np.array_equal(n - 2, a['n'])
        u = a['u']
        h = a['common']
    alt = raw.copy()
    shape = raw.copy()
    for i, (end, j) in enumerate(zip(n, b)):
        for sl, row in ((slice(2, j), slice(0, j - 2)), (slice(j, end), slice(j - 2, end - 2))):
            v = u[i, row]
            m = np.median(v)
            alt[i, 1, sl] = m + (v - m) / h[i, row]
        for ch in (0, 1):
            for sl in (slice(0, j), slice(j, end)):
                v = raw[i, ch, sl]
                mid = np.median(v)
                s = max(np.median(abs(v - mid)) * 1.4826, 0.18 * v.std(), 1e-05)
                shape[i, ch, sl] = (v - mid) / s
    return (ids, (raw, alt, shape), n, b)

@torch.no_grad()
def capture(net, data, n, b):
    out = []

    def hook(mod, args):
        out.append(args[0].detach().to(torch.float16))
    handle = net.head.register_forward_pre_hook(hook)
    rows = []
    indices = np.argsort(n.cpu().numpy())
    views = ((False, False), (True, True))
    for v in data:
        branches = []
        for rev, flip in views:
            out.clear()
            for ix in np.array_split(indices, max(1, len(indices) // 200)):
                j = torch.as_tensor(ix, device='cpu')
                no = n[j]
                bo = b[j]
                rv = torch.full_like(no, rev, dtype=torch.bool)
                fl = torch.full_like(no, flip, dtype=torch.bool)
                x, p, q, qs = prepare_input(v[j], no, bo, rv, fl)
                with _cpu_context.nullcontext():
                    net(x, p, q, x[:, 0, :0], qs)
            branches.append(torch.cat(out)[np.argsort(indices)].cpu().numpy())
        rows.append(np.stack(branches, 1))
    handle.remove()
    return np.stack(rows, 1)

class TaskVectors(nn.Module):

    def __init__(self, head, dim, ntask=5, rank=7):
        super().__init__()
        self.base = head.requires_grad_(False)
        in_dim = head[0].in_features
        self.V = nn.Parameter(torch.randn(ntask, in_dim, rank) * (0.25 / np.sqrt(in_dim)))
        self.U = nn.Parameter(torch.zeros(ntask, rank, 256))
        self.second_in = nn.Parameter(torch.randn(ntask, 256, rank) * 0.018)
        self.second_out = nn.Parameter(torch.zeros(ntask, rank, 64))
        self.aux = nn.Sequential(nn.Linear(dim, 92), nn.LayerNorm(92), nn.SiLU(), nn.Dropout(0.14), nn.Linear(92, 64), nn.SiLU())
        self.gates = nn.Linear(64, ntask)
        self.views = nn.Linear(64, 3)
        self.cal = nn.Linear(64, 2)
        self.read = nn.Linear(64, 1, bias=False)
        nn.init.zeros_(self.read.weight)
        nn.init.zeros_(self.views.weight)
        self.views.bias.data.copy_(torch.tensor([0.9, -0.1, -0.6]))
        nn.init.zeros_(self.cal.weight)
        nn.init.zeros_(self.cal.bias)

    def forward(self, x, c):
        ctx = self.aux(c)
        g = torch.softmax(self.gates(ctx), 1) * 2
        z = x.flatten(1, 2)
        change = torch.einsum('bvi,tir->bvtr', z, self.V)
        change = torch.einsum('bvtr,tro,bt->bvo', change, self.U, g)
        h = F.gelu(self.base[0](z) + change)
        h = F.dropout(h, 0.12, self.training)
        d = torch.einsum('bvi,tir->bvtr', h, self.second_in)
        d = torch.einsum('bvtr,tro,bt->bvo', d, self.second_out, g)
        h = F.gelu(self.base[3](h) + d)
        out = (self.base[6](h) + self.read(h)).reshape(len(c), 3, 2).mean(2)
        bias, scale = self.cal(ctx).unbind(1)
        return (out * torch.softmax(self.views(ctx), 1)).sum(1) * (1 + 0.48 * scale.tanh()) + bias

def run():
    seq.a.hq = 0
    ta, te = (streams('train'), streams('test'))
    ids, tids = (ta[0], te[0])
    nrow = len(ids)
    test = nrow + np.arange(len(tids))
    data = [torch.tensor(np.concatenate([u, v]), device='cpu') for u, v in zip(ta[1], te[1])]
    n, b = [torch.tensor(np.r_[ta[k], te[k]], device='cpu') for k in (2, 3)]
    del ta, te
    aux = np.vstack([phys('train', ids), phys('test', tids)])

    def feature(tag):
        return np.vstack([pd.read_parquet(OUT / f'{tag}_{p}.parquet').loc[ix].to_numpy() for p, ix in (('train', ids), ('test', tids))])
    aux = np.column_stack([aux, np.arcsinh(feature('volatility_law').clip(-22, 40))])
    y = pd.read_csv(ROOT / 'data/train_labels.csv', dtype={'id': str}).set_index('id').loc[ids, 'label'].to_numpy()
    yy = torch.tensor(y, device='cpu', dtype=torch.float32)
    pred = np.empty(nrow)
    ts = []
    dest = OUT / 'conditional_weight_models'
    dest.mkdir(exist_ok=True)
    for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(ids, y)):
        path = OUT / f'physical_ft_8/fold{fold}.pt'
        state = torch.load(path, map_location='cpu', weights_only=True)
        base = seq.Net().to('cpu').eval()
        base.load_state_dict({k[9:]: v for k, v in state.items() if k.startswith('sequence.')})
        core, inner = train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=1000 * fold)
        with np.load(OUT / f'physical_ft_8/fold{fold}.npz') as f:
            assert np.array_equal(f['va'], va)
        hpath = dest / f'hidden{fold}.npz'
        if not hpath.exists():
            H = capture(base, data, n, b)
            np.savez(hpath, ids=np.r_[ids, tids], h=H)
        else:
            with np.load(hpath) as v:
                assert np.array_equal(v['ids'], np.r_[ids, tids])
                H = v['h']
        x = torch.tensor(H, dtype=torch.float32, device='cpu')
        del H
        p = OUT / 'law_outer0'
        f = np.vstack([pd.read_parquet(p / f'fold{fold}_{part}.parquet').loc[ix].to_numpy() for part, ix in (('train', ids), ('test', tids))])
        c = np.column_stack([aux, np.arcsinh(f)])
        mu = c[core].mean(0)
        sd = c[core].std(0).clip(0.09)
        c = torch.tensor(((c - mu) / sd).clip(-7, 7), device='cpu', dtype=torch.float32)
        torch.manual_seed(941 + fold)
        rng = np.random.default_rng(941 + fold)
        model = TaskVectors(base.head, c.shape[1]).to('cpu')
        opt = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad], lr=0.00078, weight_decay=0.025)

        def output(ix):
            model.eval()
            with torch.no_grad(), _cpu_context.nullcontext():
                return torch.cat([model(x[j], c[j]).float().sigmoid() for j in np.array_split(ix, max(1, len(ix) // 400))]).cpu().numpy()
        best, bep = (-1, -1)
        ck = dest / f'fold{fold}.pt'
        for ep in range(32):
            model.train()
            model.base.eval()
            for j in np.array_split(rng.permutation(core), max(1, len(core) // 200)):
                opt.zero_grad(set_to_none=True)
                with _cpu_context.nullcontext():
                    z = model(x[j], c[j]).float()
                    loss = F.binary_cross_entropy_with_logits(z, yy[j])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
            auc = roc_auc_score(y[inner], output(inner))
            if auc > best + 5e-05:
                best, bep = (auc, ep)
                torch.save(model.state_dict(), ck)
            if ep in (7, 15):
                for g in opt.param_groups:
                    g['lr'] *= 0.45
            if ep - bep >= 6:
                break
        model.load_state_dict(torch.load(ck, map_location='cpu', weights_only=True))
        pred[va] = output(va)
        ts.append(output(test))
        np.savez_compressed(dest / f'fold{fold}_recipe.npz', center=mu, scale=sd, validation_ids=ids[va], parent_sha256=np.array(hashlib.sha256(path.read_bytes()).hexdigest()))
        print('conditional sequence weights', fold, bep, roc_auc_score(y[va], pred[va]), flush=True)
    np.savez_compressed(OUT / 'lawvectors.npz', ids=ids, tids=tids, oof=pred, test=np.mean(ts, 0))
    print('conditional sequence weight OOF', roc_auc_score(y, pred), flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        torch.set_num_threads(1)
        run()
