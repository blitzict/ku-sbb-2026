# CPU port; original experiments/weight_space/white_symmetric_tree.py, SHA256 08275ef14235bda4a3266a6289a1a182ce1cf6f83180aafbfbd6f7002382c304
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""An oblivious-tree member trained on the exact same source-grouped real views."""
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
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
ROOT = _cpu_workspace()
OUT = ROOT / 'tune_out/weight_space'
BASE = ('ar_law', 'curve', 'phase_law', 'trend', 'periodic', 'noise_tests', 'innov_v2', 'mixture_law', 'bounds', 'state')

def read(view, part, ids, ns, extras):
    root = OUT / f'view_{view}' if view else OUT
    if view:
        assert (root / ('EXTRA_READY' if extras else 'STATES_READY')).exists()
    blocks = [ns]
    for name in BASE:
        a = pd.read_parquet(root / f'{name}_{part}.parquet').loc[ids]
        blocks.append(a.to_numpy(dtype=np.float32))
    for tag in ('pulse', 'state', 'nullstate', 'prestate', 'smoothvol'):
        a = pd.read_parquet(root / f'jump_law_{tag}_{part}.parquet').loc[ids]
        if tag not in ('pulse', 'state'):
            a = a[[k for k in a if '_law' not in k and '_from' not in k and ('_to' not in k)]]
        blocks.append(a.to_numpy(dtype=np.float32))
    if extras:
        for name in ('walk_jump', 'volatility_law', 'native_law_state'):
            a = pd.read_parquet(root / f'{name}_{part}.parquet').loc[ids]
            if name == 'native_law_state':
                a = a[[c for c in a if '_pair' not in c]]
            blocks.append(a.to_numpy(dtype=np.float32))
    return np.column_stack(blocks)

def train(views, extras):
    name = 'symtree' + ('_extra' if extras else '') + '_' + '_'.join(['orig', *views])
    folder = OUT / (name + '_models')
    folder.mkdir(exist_ok=True)
    with np.load(ROOT / 'tune_out/var/mat_ns3_c7.npz', allow_pickle=True) as q:
        ids, tids = (q['ids'].astype(str), q['tid'].astype(str))
        y = q['y']
        X = []
        T = []
        for v in ('', *views):
            X.append(read(v, 'train', ids, q['V_' + v if v else 'X'], extras))
            T.append(read(v, 'test', tids, q['VT_' + v if v else 'Xt'], extras))
    ycheck = pd.read_csv(ROOT / 'data/train_labels.csv', dtype={'id': str}).set_index('id').loc[ids, 'label'].to_numpy()
    assert np.array_equal(y, ycheck)
    pv = np.empty(len(y))
    pt = []
    for f, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(ids, y)):
        snap = folder / f'fold{f}.npz'
        ck = folder / f'fold{f}.cbm'
        if snap.exists():
            with np.load(snap) as a:
                assert np.array_equal(ids[va], a['ids'])
                pv[va] = a['pv']
                pt.append(a['pt'])
            continue
        a = []
        b = []
        for v, x, t in zip(('', *views), X, T):
            root = OUT / f'view_{v}' if v else OUT
            prior = root / 'law_outer0'
            with np.load(prior / f'fold{f}_rows.npz') as rr:
                assert np.array_equal(rr['train_ids'], ids[tr]) and np.array_equal(rr['validation_ids'], ids[va])
            a.append(np.column_stack([x, pd.read_parquet(prior / f'fold{f}_train.parquet').loc[ids].to_numpy(dtype=np.float32)]))
            b.append(np.column_stack([t, pd.read_parquet(prior / f'fold{f}_test.parquet').loc[tids].to_numpy(dtype=np.float32)]))
        model = CatBoostClassifier(iterations=1080, depth=6, learning_rate=0.034, l2_leaf_reg=8.5, rsm=0.62, random_strength=1.5, bootstrap_type='Bernoulli', subsample=0.84, border_count=64, thread_count=2, task_type='CPU', random_seed=1805 + f, allow_writing_files=False, verbose=False)
        w = np.repeat([1.0, *[0.47 for _ in views]], len(tr))
        model.fit(np.vstack([h[tr] for h in a]), np.tile(y[tr], len(a)), sample_weight=w)
        model.save_model(str(ck))
        weights = np.r_[1.0, np.repeat(0.18, len(views))]
        weights /= weights.sum()
        a_pred = np.stack([model.predict(h[va], prediction_type='RawFormulaVal', thread_count=2) for h in a], 1)
        b_pred = np.stack([model.predict(h, prediction_type='RawFormulaVal', thread_count=2) for h in b], 1)
        pv[va] = expit(a_pred @ weights)
        pt.append(expit(b_pred @ weights))
        np.savez_compressed(snap, ids=ids[va], pv=pv[va], pt=pt[-1], ov=a_pred, tv=b_pred)
        print(name, f, roc_auc_score(y[va], pv[va]), flush=True)
    np.savez_compressed(OUT / (name + '.npz'), ids=ids, tids=tids, oof=pv, test=np.mean(pt, 0))
    print(name, 'OOF', roc_auc_score(y, pv), flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--views', nargs='*', default=[])
    p.add_argument('--extras', action='store_true')
    a = p.parse_args()
    with threadpool_limits(limits=2):
        train(a.views, a.extras)
