# CPU port; original experiments/weight_space/transfer_member.py, SHA256 3f4bc49af9dc11dc25ea943646a0378aa486b8df5414a8124e50526068242681
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Learn a noise-domain break expert from real, fold-isolated series."""
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
import sys
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from partition_evidence import ROOT, OUT

def prepare():
    sys.path.insert(0, str(ROOT / 'experiments/v6'))
    import features_lr, features_lm, features_sn
    for part in ('train', 'test'):
        with np.load(OUT / f'innov_v2_streams_{part}.npz') as a:
            ids, x, n, b = (a[k] for k in ('ids', 'separate', 'n', 'b'))
        index = pd.read_parquet(ROOT / f'clean/features_{part}.parquet', columns=[]).index
        assert np.array_equal(ids, index)
        frames = []
        for start in range(0, len(x), 700):
            rows = [(ids[j], x[j, :n[j]], int(b[j])) for j in range(start, min(start + 700, len(x)))]
            frames.append(pd.concat([m.block(rows) for m in (features_lr, features_lm, features_sn)], axis=1))
        frame = pd.concat(frames).replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
        assert frame.index.equals(index) and frame.columns.is_unique
        frame.to_parquet(OUT / f'noise_tests_{part}.parquet')
        print(part, frame.shape, flush=True)

def train():

    def fields(part):
        tests = pd.read_parquet(OUT / f'noise_tests_{part}.parquet')
        dynamics = pd.read_parquet(OUT / f'innov_v2_{part}.parquet').loc[tests.index]
        periodic = pd.read_parquet(OUT / f'periodic_{part}.parquet').loc[tests.index]
        x = pd.concat([tests, dynamics.add_prefix('innovation_'), periodic.add_prefix('periodic_')], axis=1)
        assert x.columns.is_unique and np.isfinite(x).all().all()
        return (x, list(tests.columns), periodic.gate.to_numpy(), dynamics.walk.to_numpy())
    tr, cols, g, w = fields('train')
    test, _, _, _ = fields('test')
    assert list(tr.columns) == list(test.columns)
    y = pd.read_csv(ROOT / 'data/train_labels.csv', dtype={'id': str}).set_index('id').loc[tr.index, 'label'].to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(tr, y))
    save = OUT / 'transfer_model'
    save.mkdir(exist_ok=True)
    oof, all_test = ({}, {})
    for mode in ('pooled', 'noise', 'stationary'):
        active = list(tr.columns) if mode == 'pooled' else cols
        xr, xt = (tr[active], test[active])
        p = np.full(len(y), np.nan)
        ts = []
        for f, (idx, va) in enumerate(folds):
            core = idx if mode != 'stationary' else idx[(g[idx] < 0.25) & (w[idx] < 0.25)]
            assert len(core) > 8000
            pars = dict(objective='binary', metric='None', verbosity=-1, learning_rate=0.035, max_depth=5, num_leaves=22, min_data_in_leaf=85, lambda_l2=15.0, max_bin=63, feature_fraction=0.8, num_threads=2, seed=538 + f, deterministic=True, force_col_wise=True)
            model = lgb.train(pars, lgb.Dataset(xr.iloc[core], label=y[core]), num_boost_round=410)
            ck = save / f'{mode}_fold{f}.txt'
            model.save_model(str(ck))
            p[va] = model.predict(xr.iloc[va], raw_score=True, num_threads=2)
            pred = model.predict(xt, raw_score=True, num_threads=2)
            replay = lgb.Booster(model_file=str(ck)).predict(xt, raw_score=True, num_threads=2)
            assert np.array_equal(replay, pred)
            ts.append(pred)
        print(mode, roc_auc_score(y, p), flush=True)
        oof[mode], all_test[mode] = (p, np.mean(ts, axis=0))
    pd.DataFrame(oof, index=tr.index).to_parquet(OUT / 'transfer_member_train.parquet')
    pd.DataFrame(all_test, index=test.index).to_parquet(OUT / 'transfer_member_test.parquet')
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prepare', action='store_true')
    a = p.parse_args()
    with threadpool_limits(limits=2):
        if a.prepare:
            prepare()
        train()
