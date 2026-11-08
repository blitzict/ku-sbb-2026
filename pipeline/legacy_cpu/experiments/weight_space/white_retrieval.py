# CPU port; original experiments/weight_space/white_retrieval.py, SHA256 53496df96959ac69d5dc8d938016c924a62b2bd9034f6e6d10ef0200dae0c6fa
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Retrieve real training series by their measured evidence of a break.

The learned-distance mechanism is inspired by Gorishniy et al., ICLR 2024,
https://proceedings.iclr.cc/paper_files/paper/2024/hash/4ef594af0d9a519db8fb292452c461fa-Abstract-Conference.html
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
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits
from white_view_network import ranks
from partition_evidence import OUT, ROOT

def inputs():
    tables = ['ar_law', 'curve', 'state', 'periodic', 'phase_law', 'trend', 'bounds', 'innov_v2', 'mixture_law', 'noise_tests', 'walk_jump', 'volatility_law', 'native_law_state']
    with np.load(ROOT / 'tune_out/var/mat_ns3_c7.npz', allow_pickle=True) as source:
        ids, tids = (source['ids'].astype(str), source['tid'].astype(str))
    views = []
    for v in ('', 'c1', 'c2'):
        root = OUT / f'view_{v}' if v else OUT
        blocks = []
        for part, ix in (('train', ids), ('test', tids)):
            terms = []
            for name in tables:
                tab = pd.read_parquet(root / f'{name}_{part}.parquet').loc[ix]
                if name == 'native_law_state':
                    tab = tab[[c for c in tab if '_pair' not in c and '_from' not in c and ('_to' not in c)]]
                terms.append(tab.to_numpy(dtype=np.float32))
            for stream in ('pulse', 'state', 'prestate', 'nullstate', 'smoothvol'):
                tab = pd.read_parquet(root / f'jump_law_{stream}_{part}.parquet').loc[ix]
                cols = [c for c in tab if '_law' not in c and '_from' not in c and ('_to' not in c)]
                terms.append(tab[cols].to_numpy(dtype=np.float32))
            blocks.append(np.column_stack(terms))
        views.append(np.vstack(blocks))
    return (ids, tids, np.stack(views, 1))

class Retrieval(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(dim, 192), nn.LayerNorm(192), nn.SiLU(), nn.Linear(192, 112), nn.LayerNorm(112), nn.SiLU())
        self.key = nn.Linear(112, 3 * 24)
        self.offset = nn.Sequential(nn.Dropout(0.17), nn.Linear(112, 64), nn.SiLU(), nn.Dropout(0.12), nn.Linear(64, 1))
        self.temperature = nn.Parameter(torch.full((3,), 3.0))
        self.update = nn.Sequential(nn.Linear(112 + 15, 64), nn.LayerNorm(64), nn.SiLU(), nn.Dropout(0.17), nn.Linear(64, 1))

    def encode(self, x):
        h = self.encoder(x)
        k = F.normalize(self.key(h).reshape(-1, 3, 24), dim=2)
        return (h, k, self.offset(h)[:, 0])

    def forward(self, x, own, bank):
        keys, score, y, ids = bank
        h, q, z = self.encode(x)
        sim = torch.bmm(q.transpose(0, 1).contiguous(), keys)
        sim = sim.masked_fill((own[:, None] == ids[None, :])[None, :, :], -100.0)
        val, near = sim.topk(48, dim=2)
        w = F.softmax((val - val[:, :, :1]) * self.temperature.clamp(1.0, 5.0).exp()[:, None, None], 2)
        labels = y[near]
        res = labels - score[near].sigmoid()

        def mean(a):
            return (w * a).sum(2).T
        fields = torch.cat([mean(labels), mean(res), mean(val), (w * w).sum(2).T, mean(score[near]).clamp(-8, 8)], 1)
        return (z + self.update(torch.cat([h, fields], 1))[:, 0], z)

def train():
    ids, tids, raw = inputs()
    n = len(ids)
    test = n + np.arange(len(tids))
    y = pd.read_csv(ROOT / 'data/train_labels.csv', dtype={'id': str}).set_index('id').loc[ids, 'label'].to_numpy()
    yy = torch.tensor(y, dtype=torch.float32, device='cpu')
    pv = np.empty(n)
    pt = []
    dest = OUT / 'lawretrieval_models'
    dest.mkdir(exist_ok=True)
    for f, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(ids, y)):
        core, inner = train_test_split(tr, test_size=0.12, stratify=y[tr], random_state=1006 + f)
        assert set(core).isdisjoint(va) and set(core).isdisjoint(inner)
        torch.manual_seed(1841 + f)
        rng = np.random.default_rng(1841 + f)
        v, knots = ranks(raw, core)
        x = torch.as_tensor(v, device='cpu')
        del v
        net = Retrieval(x.shape[-1]).to('cpu')
        opt = torch.optim.AdamW(net.parameters(), lr=0.00065, weight_decay=0.028)

        def bank():
            net.eval()
            with torch.no_grad():
                out = [net.encode(x[j, 0].float()) for j in np.array_split(core, max(1, len(core) // 700))]
                return (torch.cat([b[1] for b in out]).permute(1, 2, 0).contiguous(), torch.cat([b[2] for b in out]), yy[core], torch.as_tensor(core, device='cpu'))

        def predict(rows, ref):
            net.eval()
            result = []
            with torch.no_grad():
                for j in np.array_split(rows, max(1, len(rows) // 210)):
                    js = torch.as_tensor(j, device='cpu')
                    scores = [net(x[js, k].float(), js, ref)[0] for k in range(3)]
                    result.append(((scores[0] + 0.1 * (scores[1] + scores[2])) / 1.2).sigmoid().cpu().numpy())
            return np.concatenate(result)
        best, bep = (-1, -1)
        ck = dest / f'fold{f}.pt'
        for ep in range(36):
            ref = bank()
            net.train()
            for j in np.array_split(rng.permutation(core), max(1, len(core) // 192)):
                j = torch.as_tensor(j, device='cpu')
                views = torch.multinomial(torch.tensor([0.72, 0.14, 0.14], device='cpu'), len(j), replacement=True)
                opt.zero_grad(set_to_none=True)
                p, z = net(x[j, views].float(), j, ref)
                loss = F.binary_cross_entropy_with_logits(p, yy[j]) + 0.32 * F.binary_cross_entropy_with_logits(z, yy[j])
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                opt.step()
            sc = roc_auc_score(y[inner], predict(inner, bank()))
            if sc > best + 6e-05:
                best, bep = (sc, ep)
                torch.save(net.state_dict(), ck)
            if ep in (9, 18):
                for g in opt.param_groups:
                    g['lr'] *= 0.5
            if ep - bep >= 7:
                break
        net.load_state_dict(torch.load(ck, map_location='cpu', weights_only=True))
        ref = bank()
        pv[va] = predict(va, ref)
        pt.append(predict(test, ref))
        np.savez_compressed(dest / f'fold{f}.npz', bank_ids=ids[core], validation_ids=ids[va], knots=knots)
        print('training-neighbor recovery', f, bep, roc_auc_score(y[va], pv[va]), flush=True)
    np.savez_compressed(OUT / 'lawretrieval.npz', ids=ids, tids=tids, oof=pv, test=np.mean(pt, 0))
    print('training-neighbor OOF', roc_auc_score(y, pv), flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        torch.set_num_threads(1)
        train()
