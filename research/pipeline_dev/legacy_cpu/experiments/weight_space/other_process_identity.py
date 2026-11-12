# CPU port; original experiments/weight_space/other_process_identity.py, SHA256 15540844790e40b49ba234a47541d233a899dcdb5a1270e8dd6b1800e816be73
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Learn which process measurements repeat across real blocks of an observed segment.

The auxiliary retrieval loss uses the identity of a source segment, without
assigning structural-break labels to any constructed pair.
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
import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits
from white_block_kernel import phys
from partition_evidence import OUT, ROOT

def block(x):
    med = np.median(x)
    sd = max(x.std(), 0.0001)
    q = np.quantile(x, [0.05, 0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85, 0.95])
    scale = max(0.001, (q[-3] - q[2]) / 1.349, 0.16 * sd)
    z = (x - med) / scale
    v = np.tanh(0.75 * z)
    u = np.tanh(abs(z) - 0.85)
    diff = np.diff(x)
    loc = np.r_[np.arcsinh(q), np.log(sd), np.log(scale), np.arcsinh(x.mean()), np.log(diff.std() + 0.001)]
    shape = np.r_[np.quantile(z, [0.02, 0.08, 0.18, 0.32, 0.68, 0.82, 0.92, 0.98]).clip(-8, 8), np.cos(z[:, None] * [1.0, 2.0, 3.5]).mean(0), np.sin(z[:, None] * [1.0, 2.0]).mean(0)]

    def correlations(a, b):
        return [np.mean(a[k:] * b[:-k]) for k in (1, 2, 3, 4, 6, 8)]
    ar = np.r_[correlations(v, v), correlations(np.sign(v), np.sign(v)), np.tanh(np.median(abs(diff)) / scale)]
    vol = np.r_[correlations(u, u), correlations(v, u), np.mean(u)]
    return np.stack([loc, shape, ar, vol])

def blocks(part):
    dest = OUT / f'process_identity_blocks_{part}.npz'
    if not dest.exists():
        with np.load(ROOT / f'tune_out/seq_cache_{part}.npz', allow_pickle=True) as a:
            ids, x, n, b = [a[k] for k in ('ids', 'x', 'n', 'b')]
        result = np.empty((len(x), 2, 6, 4, 13), np.float32)
        for i, (r, end, cut) in enumerate(zip(x, n, b)):
            for side, seg in enumerate((r[:cut], r[cut:end])):
                result[i, side] = np.stack([block(c.astype(float)) for c in np.array_split(seg, 6)])
        assert np.isfinite(result).all()
        np.savez_compressed(dest, ids=ids.astype(str), x=result)
        print('identity blocks', part, result.shape, flush=True)
    with np.load(dest) as a:
        return (a['ids'], a['x'])

class Identify(nn.Module):

    def __init__(self, c):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(4, 13, 32) * 0.15)
        self.w2 = nn.Parameter(torch.randn(4, 32, 12) * 0.12)
        self.b = nn.Parameter(torch.zeros(4, 32))
        self.core = nn.Sequential(nn.Linear(c, 96), nn.LayerNorm(96), nn.GELU(), nn.Dropout(0.16), nn.Linear(96, 72), nn.LayerNorm(72), nn.GELU())
        self.base = nn.Linear(72, 1)
        self.read = nn.Sequential(nn.Linear(72 + 4 * 12 * 4 + 4 * 4, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.25), nn.Linear(128, 1))
        nn.init.zeros_(self.read[-1].weight)
        nn.init.zeros_(self.read[-1].bias)

    def forward(self, x, c):
        e = F.gelu(torch.einsum('bstgi,gih->bstgh', x, self.w1) + self.b)
        e = torch.einsum('bstgh,ghq->bstgq', e, self.w2)
        halves = F.normalize(torch.stack([e[:, :, ::2].mean(2), e[:, :, 1::2].mean(2)], dim=2), dim=-1)
        h = F.normalize(e.mean(2), dim=-1)
        diff = h[:, 1] - h[:, 0]
        within = 1 - (halves[:, :, 0] * halves[:, :, 1]).sum(-1)
        between = 1 - (h[:, 0] * h[:, 1]).sum(-1)
        repeat = ((halves[:, 1, 0] - halves[:, 0, 0]) * (halves[:, 1, 1] - halves[:, 0, 1])).sum(-1)
        stats = torch.cat([h[:, 0].flatten(1), h[:, 1].flatten(1), diff.abs().flatten(1), diff.square().flatten(1), within.flatten(1), between, repeat], dim=1)
        base = self.core(c)
        p = self.base(base)[:, 0]
        change = self.read(torch.cat([base.detach(), stats], dim=1))[:, 0]
        return (torch.stack([p, p.detach() + change], dim=1), halves, torch.cat([between, repeat, within.flatten(1)], 1))

def run(seed):
    ids, X = blocks('train')
    tids, T = blocks('test')
    raw = np.concatenate([X, T])
    y = pd.read_csv(ROOT / 'data/train_labels.csv', dtype={'id': str}).set_index('id').loc[ids, 'label'].to_numpy()
    yy = torch.tensor(y.copy(), device='cpu', dtype=torch.float32)
    c = np.vstack([phys('train', ids), phys('test', tids)])
    name = f'process_identity_s{seed}'
    dst = OUT / (name + '_models')
    dst.mkdir(exist_ok=True)
    oof = np.empty((len(y), 18))
    test = []
    for f, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(ids, y)):
        path = dst / f'fold{f}.npz'
        if not path.exists():
            core, inner = train_test_split(tr, stratify=y[tr], test_size=0.14, random_state=2923 + f)
            mid = raw[core].mean((0, 1, 2))
            sd = raw[core].std((0, 1, 2)).clip(0.04)
            x = torch.tensor(((raw - mid) / sd).clip(-5, 5), device='cpu')
            cm = c[core].mean(0)
            cs = c[core].std(0).clip(0.06)
            cx = torch.tensor(((c - cm) / cs).clip(-6, 6), device='cpu')
            torch.manual_seed(9001 + f)
            rng = np.random.default_rng(9001 + f)
            net = Identify(c.shape[1]).to('cpu')
            opt = torch.optim.AdamW(net.parameters(), lr=0.0012, weight_decay=0.035)

            def predict(ix):
                net.eval()
                out = []
                with torch.no_grad():
                    for j in np.array_split(ix, max(1, len(ix) // 320)):
                        p, _, d = net(x[j], cx[j])
                        out.append(torch.cat([p, d], 1).cpu().numpy())
                return np.vstack(out)
            best = -1.0
            epbest = -1
            ck = dst / f'fold{f}.pt'
            for ep in range(40):
                net.train()
                for j in np.array_split(rng.permutation(core), max(1, len(core) // 160)):
                    p, halves, _ = net(x[j], cx[j])
                    side = int(rng.integers(2))
                    a, b = halves[:, side].unbind(1)
                    similarity = torch.einsum('igk,jgk->gij', a, b) / 0.13
                    target = torch.arange(len(j), device='cpu').repeat(4)
                    retrieval = 0.5 * (F.cross_entropy(similarity.flatten(0, 1), target) + F.cross_entropy(similarity.transpose(1, 2).reshape(-1, len(j)), target))
                    loss = F.binary_cross_entropy_with_logits(p, yy[j, None].expand_as(p)) + 0.075 * retrieval
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                    opt.step()
                val = predict(inner)
                measure = 0.75 * roc_auc_score(y[inner], val[:, 1]) + 0.25 * roc_auc_score(y[inner], val[:, 0])
                if measure > best + 5e-05:
                    best, epbest = (measure, ep)
                    torch.save(net.state_dict(), ck)
                if ep in (12, 25):
                    for g in opt.param_groups:
                        g['lr'] *= 0.45
                if ep - epbest >= 8:
                    break
            net.load_state_dict(torch.load(ck, map_location='cpu', weights_only=True))
            np.savez_compressed(path, train_ids=ids[tr], valid_ids=ids[va], pv=predict(va), tids=tids, pt=predict(len(y) + np.arange(len(tids))), mid=mid, sd=sd, cm=cm, cs=cs)
        with np.load(path) as a:
            assert np.array_equal(a['train_ids'], ids[tr]) and np.array_equal(a['valid_ids'], ids[va]) and np.array_equal(tids, a['tids'])
            oof[va] = a['pv']
            test.append(a['pt'])
        print(name, f, [round(roc_auc_score(y[va], oof[va, k]), 6) for k in (0, 1)], flush=True)
    np.savez_compressed(OUT / (name + '.npz'), ids=ids, tids=tids, oof=oof, test=np.mean(test, axis=0))
    print(name, 'OOF', [roc_auc_score(y, oof[:, k]) for k in (0, 1)], flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        torch.set_num_threads(1)
        for seed in [int(s) for s in sys.argv[1:]] or [7, 29]:
            run(seed)
