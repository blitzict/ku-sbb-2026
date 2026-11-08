# CPU port; original experiments/weight_space/shape_bank.py, SHA256 ef08e3dffa36f3aa0f8540500e40197a47fb2e1a06b3043dcc3c29d4c99938b8
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Transfer the existing statistical detectors to nuisance-adjusted real signals."""
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
import sys
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from partition_evidence import OUT, ROOT

def prepare():
    sys.path.insert(0, str(ROOT / 'experiments/v6'))
    import features_lr, features_lm, features_sn
    for part in ('train', 'test'):
        with np.load(OUT / f'curve_streams_{part}.npz') as q:
            ids, n, b, x = (q[k] for k in ('ids', 'n', 'b', 'separate'))
        with np.load(OUT / f'periodic_streams_{part}.npz') as q:
            assert np.array_equal(ids, q['ids'])
            seasonal = q['separate']
        p = pd.read_parquet(OUT / f'periodic_{part}.parquet')
        assert np.array_equal(ids, p.index)
        g = p.gate.to_numpy() * expit((p.period.to_numpy() - 9) / 1.5) * expit((38 - p.period.to_numpy()) / 2)
        x += g[:, None] * (seasonal - x)
        frames = []
        for start in range(0, len(ids), 650):
            rows = [(ids[j], x[j, :n[j]], int(b[j])) for j in range(start, min(start + 650, len(ids)))]
            frames.append(pd.concat([m.block(rows) for m in (features_lr, features_lm, features_sn)], axis=1))
        tests = pd.concat(frames).replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
        assert tests.index.equals(p.index) and tests.columns.is_unique
        tests.to_parquet(OUT / f'shape_tests_{part}.parquet')
        print('prepared', part, tests.shape, flush=True)

def matrix(part, mode):
    names = ['curve', 'trend', 'phase_law', 'periodic', 'mixture_law', 'ar_law', 'innov_v2', 'noise_tests']
    if mode in ('adjusted', 'state'):
        names.append('shape_tests')
    if mode == 'state':
        names.extend(['pulse', 'state', 'clock', 'state_tests'])
    k = pd.read_parquet(ROOT / f'clean/features_{part}.parquet', columns=['n_pre', 'n_post', 'acf1_pre', 'kurt_pre', 'skew_pre'])
    terms = [k]
    for p in names:
        terms.append(pd.read_parquet(OUT / f'{p}_{part}.parquet').loc[k.index].add_prefix(p + '_'))
    for p in ('lm', 'gh', 'lr5', 'sn'):
        terms.append(pd.read_parquet(ROOT / f'tune_out/{p}_{part}.parquet').loc[k.index].add_prefix(p + '_'))
    x = pd.concat(terms, axis=1).replace([np.inf, -np.inf], np.nan).astype(np.float32)
    assert x.columns.is_unique and x.index.is_unique
    return x

def train(mode):
    tr, te = (matrix('train', mode), matrix('test', mode))
    assert list(tr.columns) == list(te.columns)
    y = pd.read_csv(ROOT / 'data/train_labels.csv', dtype={'id': str}).set_index('id').loc[tr.index, 'label'].to_numpy()
    dest = OUT / ('shape_bank_' + mode)
    dest.mkdir(exist_ok=True)
    oof = np.full(len(y), np.nan)
    tests = []
    for f, (ix, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(tr, y)):
        params = dict(objective='binary', verbosity=-1, num_threads=2, learning_rate=0.032, max_depth=5, num_leaves=25, min_data_in_leaf=70, lambda_l2=18.0, feature_fraction=0.8, max_bin=95, deterministic=True, force_col_wise=True, seed=723 + f)
        m = lgb.train(params, lgb.Dataset(tr.iloc[ix], label=y[ix]), num_boost_round=620)
        file = dest / f'fold{f}.txt'
        m.save_model(str(file))
        oof[va] = m.predict(tr.iloc[va], raw_score=True, num_threads=2)
        result = m.predict(te, raw_score=True, num_threads=2)
        assert np.array_equal(result, lgb.Booster(model_file=str(file)).predict(te, raw_score=True, num_threads=2))
        tests.append(result)
    print(mode, roc_auc_score(y, oof), flush=True)
    for part, scores, index in (('train', oof, tr.index), ('test', np.mean(tests, axis=0), te.index)):
        pd.DataFrame({'score': scores}, index=index).to_parquet(OUT / f'shape_bank_{mode}_{part}.parquet')
    (dest / 'columns.json').write_text(json.dumps(list(tr.columns)))
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['base', 'adjusted', 'state', 'prepare'])
    with threadpool_limits(limits=2):
        m = p.parse_args().mode
        if m == 'prepare':
            prepare()
        else:
            train(m)
