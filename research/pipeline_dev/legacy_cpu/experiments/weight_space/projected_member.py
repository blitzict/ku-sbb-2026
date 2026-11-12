# CPU port; original experiments/weight_space/projected_member.py, SHA256 721a4b987b958d3a9130a2e33e6a72b70e48c51a88aafd975fbb61e207f86399
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""A shared neural readout of before/after projected quantile curves on Intel Arc."""
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
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch.nn import functional as F
from threadpoolctl import threadpool_limits
from projected_curves import OUT, ROOT, Q, ODD, EVEN
DEVICE = 'cpu'
N_PROJ = ODD.shape[1] + EVEN.shape[1]
SAVE = OUT / 'projected_member'

def inputs(curves, context, flip):
    if flip:
        odd = -curves[:, :, :ODD.shape[1], :].flip(-1)
        curves = torch.cat([odd, curves[:, :, ODD.shape[1]:, :]], dim=2)
    before, after = (curves[:, 0], curves[:, 1])
    mid = (before + after) * 0.5
    q = torch.as_tensor(Q, dtype=torch.float32, device=curves.device)
    width = (mid[:, :, 13] - mid[:, :, 3]).clamp(0.05, 20)[:, :, None]
    lo = torch.arange(len(Q), device=curves.device).sub(1).clamp_min(0)
    hi = (lo + 2).clamp_max(len(Q) - 1)
    slope = (mid[:, :, hi] - mid[:, :, lo]) / (q[hi] - q[lo])
    den = (slope * (q * (1 - q)).sqrt()).clamp_min(0.06 * width)
    n = context[:, :2].exp()
    eff = (n.prod(1) / n.sum(1)).sqrt()[:, None, None]
    delta = after - before
    stat = torch.asinh((eff * delta / den).clamp(-80, 80))
    relative = delta / width
    return torch.cat([mid.clamp(-5, 5), stat, stat.abs(), torch.asinh(relative)], dim=-1)

class DistributionNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.quantile = nn.Sequential(nn.Linear(4 * len(Q), 64), nn.LayerNorm(64), nn.SiLU(), nn.Linear(64, 20), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(N_PROJ * 20 + 48, 192), nn.LayerNorm(192), nn.SiLU(), nn.Dropout(0.3), nn.Linear(192, 96), nn.LayerNorm(96), nn.SiLU(), nn.Dropout(0.2), nn.Linear(96, 1))

    def forward(self, x, c):
        h = self.quantile(x)
        features = torch.cat([h.flatten(1), h.mean(1), h.amax(1), c], dim=1)
        return self.head(features)[:, 0]

def load():
    values, flipped, contexts, orders = ([], [], [], [])
    for part in ('train', 'extra', 'test'):
        with np.load(OUT / f'projected_{part}.npz') as z:
            assert np.array_equal(z['q'], Q) and np.array_equal(z['odd'], ODD)
            orders.append(z['ids'])
            curves = z['curve']
            ctx = z['context']
            for flip, store in [(False, values), (True, flipped)]:
                blocks = []
                for start in range(0, len(curves), 1024):
                    c = torch.as_tensor(curves[start:start + 1024], device=DEVICE).float()
                    aux = torch.as_tensor(ctx[start:start + 1024], device=DEVICE)
                    blocks.append(inputs(c, aux, flip).half())
                store.append(torch.cat(blocks))
            contexts.append(torch.as_tensor(ctx, device=DEVICE))
    assert np.array_equal(orders[0], orders[1])
    for part, ids in zip(['train', 'test'], [orders[0], orders[2]]):
        index = pd.read_parquet(ROOT / f'clean/features_{part}.parquet', columns=[]).index
        assert np.array_equal(ids, index)
    return (torch.cat(values), torch.cat(flipped), torch.cat(contexts), orders[0], orders[2])

@torch.inference_mode()
def predict(model, X, Xf, C, indices):
    model.eval()
    scores = []
    for batch in indices.split(1024):
        with _cpu_context.nullcontext():
            orig = model(X[batch].float(), C[batch]).float()
            flip = model(Xf[batch].float(), C[batch]).float()
        scores.append(((orig + flip) / 2).cpu().numpy())
    return np.concatenate(scores)

def fit_fold(X, Xf, C, y, tr, va, test, seed, fold):
    rseed = 28349 + seed * 197 + fold
    torch.manual_seed(rseed)
    torch.manual_seed(rseed)
    core, inner = train_test_split(tr, test_size=0.12, random_state=rseed, stratify=y[tr])
    assert set(core).isdisjoint(va) and set(inner).isdisjoint(va)
    dev = lambda a: torch.as_tensor(a, device=DEVICE)
    core_ix = dev(core)
    aug_ix = dev(core + len(y))
    train_idx = torch.cat([core_ix, aug_ix])
    mu = C[train_idx].mean(0)
    sd = C[train_idx].std(0).clamp_min(0.08)
    ctx = ((C - mu) / sd).clamp(-8, 8)
    target = dev(np.r_[y, y].astype(np.float32))
    model = DistributionNet().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0012, weight_decay=0.025)
    width = 384
    epochs = 36
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=0.00012)
    gen = torch.Generator(device=DEVICE).manual_seed(rseed)
    val_ix = dev(inner)
    best_auc, best_epoch, best = (-1.0, 0, None)
    for epoch in range(epochs):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), device=DEVICE, generator=gen)]
        for batch in order.split(width):
            flip = torch.rand((len(batch), 1, 1), generator=gen, device=DEVICE) < 0.5
            xx = torch.where(flip, Xf[batch], X[batch]).float()
            opt.zero_grad(set_to_none=True)
            with _cpu_context.nullcontext():
                z = model(xx, ctx[batch]).float()
                loss = F.binary_cross_entropy_with_logits(z, target[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
        scheduler.step()
        auc = roc_auc_score(y[inner], predict(model, X, Xf, ctx, val_ix))
        if auc > best_auc + 0.0001:
            best_auc, best_epoch = (auc, epoch)
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= 7:
            break
    assert best is not None
    model.load_state_dict(best)
    pv = predict(model, X, Xf, ctx, dev(va))
    pt = predict(model, X, Xf, ctx, dev(test))
    assert np.isfinite(pv).all() and np.isfinite(pt).all()
    key = f'seed{seed}_fold{fold}'
    torch.save({'state': best, 'context_mean': mu.cpu(), 'context_sd': sd.cpu()}, SAVE / f'{key}.pt')
    ck = torch.load(SAVE / f'{key}.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(ck['state'])
    replay_ctx = ((C - ck['context_mean'].to(DEVICE)) / ck['context_sd'].to(DEVICE)).clamp(-8, 8)
    assert np.array_equal(pv, predict(model, X, Xf, replay_ctx, dev(va)))
    return (pv, pt, {'fold': fold, 'seed': seed, 'inner_auc': best_auc, 'epochs': epoch + 1, 'best_epoch': best_epoch + 1, 'outer_auc': float(roc_auc_score(y[va], pv))})

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    SAVE.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    X, Xf, C, ids, tids = load()
    y = pd.read_csv(ROOT / 'data/train_labels.csv', dtype={'id': str}).set_index('id').loc[ids, 'label'].to_numpy()
    test = np.arange(2 * len(y), 2 * len(y) + len(tids))
    folds = StratifiedKFold(5, shuffle=True, random_state=0).split(y, y)
    oof = np.full(len(y), np.nan)
    tests, info = ([], [])
    for f, (tr, va) in enumerate(folds):
        key = f'seed{args.seed}_fold{f}'
        archive = SAVE / f'{key}.npz'
        if args.resume and archive.exists() and (SAVE / f'{key}.pt').exists():
            with np.load(archive) as z:
                assert np.array_equal(z['ids'], ids[va]) and np.array_equal(z['tids'], tids)
                pv, pt, meta = (z['validation'], z['test'], json.loads(str(z['meta'])))
        else:
            start = time.time()
            pv, pt, meta = fit_fold(X, Xf, C, y, tr, va, test, args.seed, f)
            np.savez(archive, ids=ids[va], tids=tids, validation=pv, test=pt, meta=json.dumps(meta))
            print(key, meta, f'{time.time() - start:.0f}s', flush=True)
        oof[va] = pv
        tests.append(pt)
        info.append(meta)
    assert np.isfinite(oof).all()
    prediction = np.mean(tests, axis=0)
    np.savez(SAVE / f'seed{args.seed}.npz', ids=ids, tids=tids, oof=oof, test=prediction, meta=json.dumps(info))
    print('Full member AUC', roc_auc_score(y, oof), flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
