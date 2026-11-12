# CPU port; original experiments/weight_space/sensitivity.py, SHA256 01a60f4c709bd5566bc9cc12dee94ba52766c4e9aa52260e4aba4b9465cc143c
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Decompose each held-out MLP score into contributions of feature families.

The reference input is the median of the checkpoint's own training fold in
rank-Gaussian coordinates. Gauss-Legendre quadrature integrates local input
gradients along the line to the observed input (Sundararajan et al., 2017).
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
import gc
import json
import re
import time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
import expert_fusion as ef
from neural_head_edit import MLP, rank_reference, transform, KINDS, VIEW_WEIGHTS

def family(col):
    if '__' in col:
        block, rest = col.split('__')
        if block == 'dd':
            return 'dd_' + rest.split('_')[1]
        primary = rest.split('_')[0]
        return block + '_' + primary
    start = col.split('_')[0]
    if start in {'n', 'boundary', 'log'} and col in {'n_pre', 'n_post', 'boundary_frac', 'log_n'}:
        return 'span'
    if start in {'raw', 'abs', 'ar1', 'ar2', 'ar4', 'ar8', 'd1', 'own4', 'vroll16', 'vroll48', 'vewm16'}:
        return start
    if re.search('acf|pacf|chow|lag|persist|freq|spect|period|peak', col):
        return 'dependence'
    if re.search('std|spread|scale|iqr|mad|var|lev[e]?ne', col):
        return 'scale'
    if re.search('kurt|skew|entropy|tail|quant|wass|ks|q[0-9]', col):
        return 'shape'
    return 'misc'

def contributions(model, batch, groupmap):
    nodes, weights = np.polynomial.legendre.leggauss(5)
    gradient = torch.zeros_like(batch)
    for node, weight in zip((nodes + 1) / 2, weights / 2):
        point = (node * batch).requires_grad_(True)
        pred = model(point)
        gradient += float(weight) * torch.autograd.grad(pred.sum(), point)[0]
    contribution = gradient * batch @ groupmap
    with torch.no_grad():
        error = (contribution.sum(1) - (model(batch) - model(torch.zeros_like(batch[:1])))).abs()
    return (contribution.detach().cpu().numpy(), float(error.mean().cpu()))

def run():
    torch.set_num_threads(2)
    raw = np.load(ef.O / 'var/mat_cggd_hv.npz', allow_pickle=True)
    X = {k: raw['X' if k == 'orig' else f'V_{k}'] for k in KINDS}
    T = {k: raw['Xt' if k == 'orig' else f'VT_{k}'] for k in KINDS}
    A, E, y, ids, tids = (raw[k] for k in ('A', 'E', 'y', 'ids', 'tid'))
    index = pd.Index(ids)
    assert index.equals(pd.read_parquet(ef.ROOT / 'clean/features_train.parquet', columns=[]).index)
    assert pd.Index(tids).equals(pd.read_parquet(ef.ROOT / 'clean/features_test.parquet', columns=[]).index)
    sources, extra = (index.get_indexer(raw[k]) for k in ('src', 'Esrc'))
    assert min(sources.min(), extra.min()) >= 0
    labels = np.array([family(c) for c in raw['cols']])
    unique, counts = np.unique(labels, return_counts=True)
    labels[np.isin(labels, unique[counts < 6])] = 'misc'
    labels = np.unique(labels, return_inverse=True)
    names, indices = labels
    groupmap = torch.as_tensor(np.eye(len(names), dtype=np.float32)[indices], device='cpu')
    nfeat = len(names)
    save = ef.OUT / 'sensitivity'
    save.mkdir(exist_ok=True)
    seed_count = 2
    cols = [f'{s}_{k}' for s in ('average', 'view_sd', 'original') for k in names]
    oof = np.full((len(ids), len(cols)), np.nan)
    test = np.zeros((len(tids), len(cols)))
    folds = StratifiedKFold(5, shuffle=True, random_state=0).split(ids, y)
    for fold, (tr, va) in enumerate(folds):
        path = save / f'fold{fold}.npz'
        if path.exists():
            with np.load(path) as z:
                assert np.array_equal(z['va'], va) and np.array_equal(z['cols'], cols)
                oof[va], test = (z['val'], test + z['test'] / 5)
            continue
        tick = time.monotonic()
        core = np.random.RandomState(fold).permutation(tr)[len(tr) // 10:]
        keep = np.zeros(len(ids), bool)
        keep[core] = True
        fit = np.vstack([X['orig'][core], A[keep[sources]], *[X[k][core] for k in KINDS[1:]], E[keep[extra]]])
        ref, n = rank_reference(fit)
        del fit
        models = []
        for seed in range(seed_count):
            file = ef.PACKAGE / f'neural_head/head_v2_MLPHV_big/fold{fold}_seed{seed}.pt'
            m = MLP(len(indices)).to('cpu')
            m.load_state_dict(torch.load(file, map_location='cpu', weights_only=True))
            models.append(m.eval().requires_grad_(False))
        total = len(va) + len(tids)
        contrib = np.zeros((len(KINDS), total, nfeat))
        scores = np.zeros((seed_count, len(KINDS), total))
        errors = []
        for j, k in enumerate(KINDS):
            rows = np.vstack([X[k][va], T[k]])
            for part in (slice(0, len(va)), slice(len(va), total)):
                xx = transform(rows[part], ref, n)
                for s, m in enumerate(models):
                    with torch.no_grad(), _cpu_context.nullcontext():
                        h = m.net[:-1](xx).float()
                    with torch.no_grad():
                        scores[s, j, part] = m.net[-1](h)[:, 0].cpu().numpy()
            for lo in range(0, total, 1024):
                sl = slice(lo, min(lo + 1024, total))
                x = transform(rows[sl], ref, n)
                for s, m in enumerate(models):
                    a, e = contributions(m, x, groupmap)
                    contrib[j, sl] += a / seed_count
                    errors.append(e)
        hist = ef.OLD / f'head_v2_MLPHV_big/head_fold{fold}.npz'
        with np.load(hist) as h:
            assert np.array_equal(va, h['va'])
            reference = np.column_stack([h['edited'][:seed_count], h['edited_test'][:seed_count]])
            re = np.abs((scores * VIEW_WEIGHTS[None, :, None]).sum(1) - reference)
            assert re.max() < 0.0001, re.max()
        mean = np.tensordot(VIEW_WEIGHTS, contrib, axes=1)
        sd = np.sqrt(np.tensordot(VIEW_WEIGHTS, (contrib - mean) ** 2, axes=1))
        features = np.column_stack([mean, sd, contrib[0]])
        assert np.isfinite(features).all() and np.mean(errors) < 0.08, np.mean(errors)
        val, ts = (features[:len(va)], features[len(va):])
        np.savez(path, va=va, cols=cols, val=val, test=ts, replay_error=re.max(), integration_error=np.mean(errors))
        oof[va], test = (val, test + ts / 5)
        print(f'fold {fold} replay {re.max():.5g}, quadrature {np.mean(errors):.5g}, {time.monotonic() - tick:.1f}s', flush=True)
        del models, ref
        gc.collect()
        None
    assert np.isfinite(oof).all()
    for part, array, index in [('train', oof, ids), ('test', test, tids)]:
        pd.DataFrame(array, index=index, columns=cols).to_parquet(ef.OUT / f'sensitivity_{part}.parquet')
if __name__ == '__main__':
    with threadpool_limits(limits=2):
        run()
