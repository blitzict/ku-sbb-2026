# CPU port; original experiments/weight_space/other_source_measurements.py, SHA256 d7109613645f6a1df25950a9990a474e4416d5e30257dcc511799cb0c9470fa1
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Paired source classifier on observed trajectories and later physical measurements."""
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from white_block_kernel import phys
from white_view_network import ranks
from partition_evidence import OUT, ROOT
TABLES = ('kernel_timing', 'transition_neighbors', 'phase_coupling', 'shock_ringing', 'mode_switching', 'time_arrow', 'value_bands', 'conditional_noise_law', 'strict_unit_memory', 'other_memory_horizons', 'transition_catalog', 'state_tests', 'seasonal_memory', 'wavelet_envelopes', 'quantile_alignment', 'volatility_law', 'asymmetric_response', 'shock_spacing', 'conditional_drift', 'cold_start_effect', 'boundary_law_state')

def source():
    with np.load(ROOT / 'tune_out/var/mat_ns3_c7.npz', allow_pickle=True) as a:
        ids, tids = (a['ids'].astype(str), a['tid'].astype(str))
        base = np.vstack([np.column_stack([a[k], phys(p, i)]) for k, p, i in (('X', 'train', ids), ('Xt', 'test', tids))])
    tables = []
    for name in TABLES:
        tr, te = [pd.read_parquet(OUT / f'{name}_{p}.parquet').loc[i] for p, i in (('train', ids), ('test', tids))]
        assert tr.columns.equals(te.columns)
        tables.append(np.vstack([tr, te]).astype(np.float32))
    labels = pd.read_csv(ROOT / 'data/train_labels.csv', dtype={'id': str}).set_index('id').loc[ids, 'label'].to_numpy()
    return (ids, tids, labels, np.column_stack([base, *tables]).astype(np.float32), base.shape[1])

class Readout(nn.Module):

    def __init__(self, d, base):
        super().__init__()
        self.base = base

        def encoding(width, hidden):
            return nn.Sequential(nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.18), nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.primary = encoding(base, 120)
        self.extension = encoding(d - base, 120)
        self.common = nn.Linear(120, 1)
        self.change = nn.Sequential(nn.Linear(240, 80), nn.GELU(), nn.Dropout(0.25), nn.Linear(80, 1))
        nn.init.zeros_(self.change[-1].weight)
        nn.init.zeros_(self.change[-1].bias)

    def forward(self, x):
        h = self.primary(x[:, :self.base])
        p = self.common(h)[:, 0]
        t = self.extension(x[:, self.base:])
        delta = self.change(torch.cat([h.detach(), t], dim=1))[:, 0]
        return torch.stack([p, p.detach() + delta], dim=1)

def run(seed):
    ids, tids, y, raw, b = source()
    name = f'measured_source_s{seed}'
    output = OUT / (name + '_models')
    output.mkdir(exist_ok=True)
    print(name, 'base dimensions', b, 'later dimensions', raw.shape[1] - b, flush=True)
    labels = torch.tensor(y.copy(), dtype=torch.float32, device='cpu')
    oof = np.empty((len(y), 2))
    tests = []
    for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(y, y)):
        path = output / f'fold{fold}.npz'
        if not path.exists():
            core, inner = train_test_split(tr, stratify=y[tr], test_size=0.13, random_state=7301 + fold)
            h, knots = ranks(raw[:, None, :], core)
            x = torch.tensor(h[:, 0, :], device='cpu', dtype=torch.float32)
            del h
            torch.manual_seed(5021 + fold)
            rng = np.random.default_rng(5021 + fold)
            net = Readout(raw.shape[1], b).to('cpu')
            opt = torch.optim.AdamW(net.parameters(), lr=0.00085, weight_decay=0.05)

            def predict(ix):
                net.eval()
                with torch.no_grad():
                    return np.vstack([net(x[j]).float().cpu().numpy() for j in np.array_split(ix, max(1, (len(ix) + 511) // 512))])
            epbest = -1
            best = -1.0
            file = output / f'fold{fold}.pt'
            for ep in range(37):
                net.train()
                for j in np.array_split(rng.permutation(core), max(1, len(core) // 224)):
                    opt.zero_grad(set_to_none=True)
                    scores = net(x[j])
                    yy = labels[j, None].expand_as(scores)
                    loss = F.binary_cross_entropy_with_logits(scores, yy) + 0.008 * (scores[:, 0].detach() - scores[:, 1]).square().mean()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 2.5)
                    opt.step()
                p = predict(inner)
                quality = np.mean([roc_auc_score(y[inner], p[:, k]) for k in range(2)])
                if quality > best + 4e-05:
                    epbest, best = (ep, quality)
                    torch.save(net.state_dict(), file)
                if ep in (12, 22):
                    for g in opt.param_groups:
                        g['lr'] *= 0.42
                if ep - epbest >= 8:
                    break
            net.load_state_dict(torch.load(file, map_location='cpu', weights_only=True))
            val = predict(va)
            test = predict(len(y) + np.arange(len(tids)))
            np.savez_compressed(path, valid_ids=ids[va], train_ids=ids[tr], tids=tids, pv=val, pt=test, knots=knots, epoch=epbest)
        with np.load(path) as a:
            assert np.array_equal(ids[va], a['valid_ids']) and np.array_equal(ids[tr], a['train_ids']) and np.array_equal(tids, a['tids'])
            oof[va] = a['pv']
            tests.append(a['pt'])
        print(name, fold, [round(roc_auc_score(y[va], oof[va, k]), 6) for k in range(2)], flush=True)
    np.savez_compressed(OUT / (name + '.npz'), ids=ids, tids=tids, oof=oof, test=np.mean(tests, axis=0))
    print(name, 'pooled OOF', [roc_auc_score(y, oof[:, k]) for k in range(2)], flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        torch.set_num_threads(1)
        for seed in [int(s) for s in sys.argv[1:]] or [7, 29]:
            run(seed)
