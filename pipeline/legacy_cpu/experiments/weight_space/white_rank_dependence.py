# CPU port; original experiments/weight_space/white_rank_dependence.py, SHA256 320d235bf54fdcb2816f9f8e2bd11155f712ee05cc79e321e9766f08a2343cb5
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Learn which pairs of observation ranks change at a boundary in a real series."""
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
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits
from white_block_kernel import phys
from partition_evidence import OUT, ROOT
BINS = 12
LAGS = (1, 3, 8, 16)

def histogram(codes, lag):
    pair = codes[:-lag] * BINS + codes[lag:]
    return np.bincount(pair, minlength=BINS ** 2).reshape(BINS, BINS) / len(pair)

def charts(part):
    path = OUT / f'rank_dependence_{part}.npz'
    if not path.exists():
        frames = []
        for tag in ('pulse', 'state'):
            with np.load(OUT / f'{tag}_streams_{part}.npz') as a:
                current, values, lengths, cuts = (a[k] for k in ('ids', 'separate', 'n', 'b'))
            if frames:
                assert np.array_equal(ids, current)
            ids = current
            block = np.empty((len(ids), 2 * len(LAGS), 2, BINS, BINS), dtype=np.float16)
            for i, (n, cut) in enumerate(zip(lengths, cuts)):
                v = values[i, :n]

                def bins(x):
                    return ((rankdata(x) - 0.5) * BINS / len(x)).astype(np.int32)
                pooled = bins(v)
                for side, sl in enumerate((slice(0, cut), slice(cut, n))):
                    for mode, codes in enumerate((pooled[sl], bins(v[sl]))):
                        for j, lag in enumerate(LAGS):
                            block[i, mode * len(LAGS) + j, side] = np.sqrt(histogram(codes, lag))
            frames.append(block)
        x = np.concatenate(frames, axis=1)
        assert np.isfinite(x).all()
        np.savez_compressed(path, ids=ids, x=x, n=lengths, b=cuts)
    with np.load(path) as a:
        return tuple((a[k] for k in ('ids', 'x', 'n', 'b')))

class RankPairs(nn.Module):

    def __init__(self, auxiliary):
        super().__init__()
        self.filters = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.SiLU(), nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.SiLU(), nn.AvgPool2d(2), nn.Flatten(), nn.Linear(144, 28), nn.LayerNorm(28))
        self.aux = nn.Sequential(nn.Linear(auxiliary, 80), nn.LayerNorm(80), nn.SiLU())
        self.read = nn.Sequential(nn.Linear(16 * 28 + 80, 128), nn.LayerNorm(128), nn.SiLU(), nn.Dropout(0.22), nn.Linear(128, 48), nn.SiLU(), nn.Dropout(0.12), nn.Linear(48, 1))

    def forward(self, x, c, n):
        pre, post = x.unbind(2)
        v = torch.stack((9 * (pre - 0.08), 9 * (post - 0.08), (post - pre) * n[:, None, None, None]), 2)
        f = self.filters(v.flatten(0, 1)).reshape(len(x), -1)
        return self.read(torch.cat((f, self.aux(c)), 1))[:, 0]

def run(seed):
    ids, a, n1, b1 = charts('train')
    tids, b, n2, b2 = charts('test')
    y = pd.read_csv(ROOT / 'data/train_labels.csv', dtype={'id': str}).set_index('id').loc[ids, 'label'].to_numpy()
    chart = torch.as_tensor(np.concatenate((a, b)), device='cpu', dtype=torch.float16)
    lengths, cuts = (np.r_[n1, n2], np.r_[b1, b2])
    scale = torch.as_tensor(np.sqrt(cuts * (lengths - cuts) / lengths), device='cpu', dtype=torch.float32)
    physical = np.vstack((phys('train', ids), phys('test', tids)))
    y_gpu = torch.tensor(y, device='cpu', dtype=torch.float32)
    test = len(ids) + np.arange(len(tids))
    oof = np.empty(len(ids))
    outputs = []
    tag = f'law_rank_dependence_s{seed}'
    dest = OUT / (tag + '_models')
    dest.mkdir(exist_ok=True)
    for f, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(ids, y)):
        record = dest / f'fold{f}.npz'
        if record.exists():
            with np.load(record) as a:
                assert np.array_equal(a['ids'], ids[va])
                oof[va] = a['pred']
                outputs.append(a['test'])
            continue
        core, inner = train_test_split(tr, test_size=0.12, stratify=y[tr], random_state=2401 + f)
        path = OUT / f'law_outer{seed}'
        with np.load(path / f'fold{f}_rows.npz') as a:
            assert np.array_equal(ids[va], a['validation_ids'])
        prior = np.vstack([pd.read_parquet(path / f'fold{f}_{p}.parquet').loc[ix].to_numpy() for p, ix in (('train', ids), ('test', tids))])
        features = np.column_stack((physical, np.arcsinh(prior)))
        mu, sd = (features[core].mean(0), features[core].std(0).clip(0.04))
        c = torch.as_tensor(((features - mu) / sd).clip(-8, 8), device='cpu', dtype=torch.float32)
        torch.manual_seed(971 + f)
        rng = np.random.default_rng(971 + f)
        model = RankPairs(c.shape[1]).to('cpu')
        opt = torch.optim.AdamW(model.parameters(), lr=0.00075, weight_decay=0.025)
        ck = dest / f'fold{f}.pt'

        def predict(rows):
            model.eval()
            with torch.no_grad(), _cpu_context.nullcontext():
                return torch.cat([model(chart[j], c[j], scale[j]).float().sigmoid() for j in np.array_split(rows, max(1, len(rows) // 180))]).cpu().numpy()
        best, best_epoch = (-1.0, 0)
        for epoch in range(32):
            model.train()
            for j in np.array_split(rng.permutation(core), max(1, len(core) // 150)):
                opt.zero_grad(set_to_none=True)
                with _cpu_context.nullcontext():
                    logits = model(chart[j], c[j], scale[j]).float()
                    loss = nn.functional.binary_cross_entropy_with_logits(logits, y_gpu[j])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
            score = roc_auc_score(y[inner], predict(inner))
            if score > best + 0.0001:
                best, best_epoch = (score, epoch)
                torch.save(model.state_dict(), ck)
            if epoch in (9, 18):
                for group in opt.param_groups:
                    group['lr'] *= 0.45
            if epoch - best_epoch >= 6:
                break
        model.load_state_dict(torch.load(ck, map_location='cpu', weights_only=True))
        oof[va] = predict(va)
        outputs.append(predict(test))
        np.savez_compressed(record, ids=ids[va], pred=oof[va], test=outputs[-1], mu=mu, sd=sd)
        print(tag, f, best_epoch, roc_auc_score(y[va], oof[va]), flush=True)
    np.savez_compressed(OUT / (tag + '.npz'), ids=ids, tids=tids, oof=oof, test=np.mean(outputs, axis=0))
    print(tag, roc_auc_score(y, oof), flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('seed', type=int)
    with threadpool_limits(limits=1):
        torch.set_num_threads(1)
        run(parser.parse_args().seed)
