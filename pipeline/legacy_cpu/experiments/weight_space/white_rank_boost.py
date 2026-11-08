# CPU port; original experiments/weight_space/white_rank_boost.py, SHA256 9d0cc27f2033ae24c3f4be5384474cee9829f3b8207c8f0e2a5cb560f09ab827
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Fit trees to weighted positive-negative comparisons on genuine source folds."""
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
import lightgbm as lgb
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from white_second_tree import input_frame
from white_view_network import TABLES
from partition_evidence import OUT, ROOT
GRID = np.linspace(-12, 12, 361)

def rank_objective(y, weights, white):
    delta = (GRID[None, :] - GRID[:, None]) / 0.82
    logistic = expit(delta)
    curvature = logistic * (1 - logistic) / 0.82

    def obj(z, dataset):
        p = expit(z)
        grad = 0.31 * (p - y)
        hess = 0.31 * p * (1 - p)
        cell = np.rint((z - GRID[0]) / (GRID[1] - GRID[0])).astype(int).clip(0, len(GRID) - 1)
        for mass, member in ((0.49, np.ones(len(y))), (0.2, white)):
            a, b = [np.bincount(cell, weights=weights * member * (y == k), minlength=len(GRID)) for k in (0, 1)]
            a /= a.sum()
            b /= b.sum()
            gp = np.interp(z, GRID, logistic @ a)
            gn = np.interp(z, GRID, (1 - logistic) @ b)
            hp = np.interp(z, GRID, curvature @ a)
            hn = np.interp(z, GRID, curvature @ b)
            grad += mass * member * np.where(y, -gp, gn)
            hess += mass * member * np.where(y, hp, hn)
        return (weights * grad, weights * hess.clip(0.012))
    return obj

def crop_data(view, part, ids, x):
    root = OUT / f'view_{view}' if view else OUT
    blocks = [x]
    for name in (*TABLES, 'volatility_law', 'walk_jump', 'native_law_state', 'state', *['jump_law_' + s for s in ('pulse', 'state', 'nullstate', 'prestate', 'smoothvol')]):
        a = pd.read_parquet(root / f'{name}_{part}.parquet').loc[ids]
        if name.startswith(('jump_law', 'native_law')):
            a = a[[c for c in a if '_pair' not in c and '_law' not in c]]
        blocks.append(a.to_numpy(dtype=np.float32))
    return np.column_stack(blocks)

def run(mode):
    with np.load(ROOT / 'tune_out/var/mat_ns3_c7.npz', allow_pickle=True) as q:
        ids, tids = (q['ids'].astype(str), q['tid'].astype(str))
        y = q['y'].astype(bool)
        views = ['', 'c1', 'c2', 'rev'] if mode == 'crops' else ['']
        arr = []
        tests = []
        gates = []
        for v in views:
            root = OUT / f'view_{v}' if v else OUT
            if mode == 'crops':
                arr.append(crop_data(v, 'train', ids, q['V_' + v if v else 'X']))
                tests.append(crop_data(v, 'test', tids, q['VT_' + v if v else 'Xt']))
            else:
                names = ['native_law_state', 'filtered_step', 'markov_copula', 'component_moves_state', 'component_moves_pulse']
                arr.append(input_frame('train', pd.Index(ids), names).to_numpy(dtype=np.float32))
                tests.append(input_frame('test', pd.Index(tids), names).to_numpy(dtype=np.float32))
            gates.append(pd.read_parquet(root / 'innov_v2_train.parquet').loc[ids, 'pre_white'].to_numpy().clip(0, 1))
    labels = pd.read_csv(ROOT / 'data/train_labels.csv', dtype={'id': str}).set_index('id').loc[ids, 'label'].to_numpy()
    assert np.array_equal(y, labels)
    oof = np.empty(len(y))
    test = []
    name = 'lawrank_' + mode
    dest = OUT / (name + '_models')
    dest.mkdir(exist_ok=True)
    for f, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(ids, y)):
        X = []
        V = []
        T = []
        for v, a, b in zip(views, arr, tests):
            root = OUT / f'view_{v}' if v else OUT
            d = root / 'law_outer0'
            with np.load(d / f'fold{f}_rows.npz') as rows:
                assert np.array_equal(rows['train_ids'], ids[tr]) and np.array_equal(rows['validation_ids'], ids[va])
            h = pd.read_parquet(d / f'fold{f}_train.parquet').loc[ids].to_numpy(dtype=np.float32)
            X.append(np.column_stack([a[tr], h[tr]]))
            V.append(np.column_stack([a[va], h[va]]))
            T.append(np.column_stack([b, pd.read_parquet(d / f'fold{f}_test.parquet').loc[tids].to_numpy(dtype=np.float32)]))
        ww = np.repeat([1.0, *[0.38 for v in views[1:]]], len(tr))
        yy = np.tile(y[tr], len(views))
        gate = np.concatenate([g[tr] for g in gates])
        prior = rank_objective(yy, ww, gate)
        ck = dest / f'fold{f}.txt'
        if ck.exists():
            model = lgb.Booster(model_file=str(ck))
        else:
            model = lgb.train(dict(objective=prior, metric='None', learning_rate=0.035, num_leaves=25, max_depth=6, min_data_in_leaf=85, lambda_l2=13.0, feature_fraction=0.58, verbosity=-1, deterministic=True, force_col_wise=True, num_threads=2, max_bin=71, seed=3501 + f), lgb.Dataset(np.vstack(X), label=yy), num_boost_round=730)
            model.save_model(str(ck))
        w = np.array([1.0, *[0.11 for v in views[1:]]])
        w /= w.sum()
        pred = lambda frames: expit(sum((a * model.predict(b, raw_score=True, num_threads=2) for a, b in zip(w, frames))))
        oof[va] = pred(V)
        test.append(pred(T))
        print(name, f, roc_auc_score(y[va], oof[va]), flush=True)
    np.savez_compressed(OUT / f'{name}.npz', oof=oof, test=np.mean(test, 0), ids=ids, tids=tids)
    print(name, 'OOF', roc_auc_score(y, oof), flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('original', 'crops'))
    with threadpool_limits(limits=2):
        run(p.parse_args().mode)
