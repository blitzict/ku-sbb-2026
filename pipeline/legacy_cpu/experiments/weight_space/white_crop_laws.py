# CPU port; original experiments/weight_space/white_crop_laws.py, SHA256 fdf43a694f489656772dab5e211093a52e6476e093923e18c46d455cf05f3439
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Evaluate the new physical likelihoods on the project's approved real views."""
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
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from partition_evidence import ROOT, OUT
from white_jump_priors import extract
from state_filter import state

def make(part, view):
    paths = [OUT / f'{tag}{view}_streams_{part}.npz' for tag in ('pulse', 'state')]
    par = OUT / f'view_state_{view}_{part}.parquet'
    if par.exists() and all((p.exists() for p in paths)):
        return
    with np.load(OUT / f'pulse_streams_{part}.npz') as q:
        ids, x, n, b = (q[k] for k in ('ids', 'separate', 'n', 'b'))
    df = pd.read_parquet(OUT / f'trend_{part}.parquet').loc[ids, 'df'].to_numpy().clip(4, 60)
    if view == 'rev':
        new_n = n.copy()
        new_b = n - b
    else:
        frame = pd.read_parquet(ROOT / f'tune_out/var/clean_{part}_{view}.parquet', columns=['n_pre', 'n_post'])
        index = [str(k) + view for k in ids]
        r = frame.loc[index, ['n_pre', 'n_post']].to_numpy(dtype=int)
        new_b = r[:, 0]
        new_n = r.sum(1)
        assert np.all(new_b <= b) and np.all(new_n - new_b <= n - b)
    levels = np.zeros_like(x)
    innov = np.zeros_like(x)
    rows = []
    for i, k in enumerate(ids):
        if view == 'rev':
            v = x[i, :n[i]][::-1].astype(float)
        else:
            start = b[i] - new_b[i]
            v = x[i, start:start + new_n[i]].astype(float)
        levels[i, :new_n[i]] = v
        _, iv, m = state(v, new_b[i], df[i])
        innov[i, :new_n[i]] = iv
        rows.append(m)
    for path, arr in zip(paths, (levels, innov)):
        np.savez(path, ids=ids, n=new_n, b=new_b, separate=arr)
    pd.DataFrame(rows, index=pd.Index(ids, name='id')).to_parquet(par)
    print('verified', part, view, 'retained real ranges', len(ids), flush=True)

def design(ns, ids, part, view):
    cols = ['ar1', 'ar2', 'pre_ar1', 'pre_ar2', 'pre2', 'walk', 'clustering', 'pre_cluster', 'post_cluster', 'prior_scale', 'post_scale', 'ar_change', 'vol_split_bf']
    sp = OUT / (f'view_state_{view}_{part}.parquet' if view else f'state_{part}.parquet')
    fields = [ns, pd.read_parquet(sp, columns=cols).loc[ids].to_numpy()]
    for tag in ('pulse', 'state'):
        a = pd.read_parquet(OUT / f'jump_law_{tag}{view}_{part}.parquet').loc[ids]
        fields.append(a.to_numpy())
    return np.ascontiguousarray(np.column_stack(fields), dtype=np.float32)

def train(views):
    with np.load(ROOT / 'tune_out/var/mat_ns3_c7.npz', allow_pickle=True) as q:
        ids = q['ids'].astype(str)
        tids = q['tid'].astype(str)
        y = q['y']
        X = [design(q['X'], ids, 'train', '')] + [design(q[f'V_{v}'], ids, 'train', v) for v in views]
        T = [design(q['Xt'], tids, 'test', '')] + [design(q[f'VT_{v}'], tids, 'test', v) for v in views]
    ycheck = pd.read_csv(ROOT / 'data/train_labels.csv', dtype={'id': str}).set_index('id').loc[ids, 'label'].to_numpy()
    assert np.array_equal(y, ycheck)
    name = 'lawcrop_' + '_'.join(views)
    dest = OUT / (name + '_models')
    dest.mkdir(exist_ok=True)
    pv = np.empty((len(y), len(X)))
    pt = []
    for f, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(ids, y)):
        ck = dest / f'fold{f}.txt'
        if ck.exists():
            m = lgb.Booster(model_file=str(ck))
        else:
            data = np.vstack([v[tr] for v in X])
            labels = np.tile(y[tr], len(X))
            wt = np.repeat([1.0, *[0.55 for _ in views]], len(tr))
            params = dict(objective='binary', verbosity=-1, num_threads=2, learning_rate=0.028, num_leaves=40, max_depth=7, min_data_in_leaf=90, lambda_l2=18.0, feature_fraction=0.72, max_bin=100, deterministic=True, force_col_wise=True, seed=162 + f)
            m = lgb.train(params, lgb.Dataset(data, label=labels, weight=wt), num_boost_round=900)
            m.save_model(str(ck))
        pv[va] = np.stack([m.predict(v[va], num_threads=2) for v in X], 1)
        pt.append(np.stack([m.predict(v, num_threads=2) for v in T], 1))
        print('crop tree', f, round(roc_auc_score(y[va], pv[va, 0]), 6), flush=True)
    wt = np.r_[0.65, np.full(len(views), 0.35 / len(views))]
    pred = pv @ wt
    test = np.mean(pt, axis=0) @ wt
    np.savez_compressed(OUT / (name + '.npz'), ids=ids, tids=tids, oof=pred, test=test, oof_views=pv, test_views=np.mean(pt, axis=0))
    print(name, 'raw / view-mean', roc_auc_score(y, pv[:, 0]), roc_auc_score(y, pred), flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('views', nargs='+', choices=('c1', 'c2', 'rev'))
    p.add_argument('--train', action='store_true')
    a = p.parse_args()
    with threadpool_limits(limits=1), torch.inference_mode():
        torch.set_num_threads(1)
        for view in a.views:
            for part in ('train', 'test'):
                make(part, view)
            for tag in ('pulse', 'state'):
                for part in ('train', 'test'):
                    extract(part, tag + view)
        if a.train:
            train(a.views)
