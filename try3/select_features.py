"""try3 step 1: how few features does it take to get close to the 0.879?

python try3/select_features.py --work <training work folder> --out select_out

#all label-free feature tables the cpu rebuild made (~5,100 columns)
#each fold: rank features on that fold's training rows only, keep the top K, refit, score the held-out rows
#so the selection never sees the rows it is scored on
"""
import os
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[k] = '1'
import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

#priors fitted on the training labels: leave out, they would leak inside cv
LABEL_FITTED = {"transition_odds", "transport_state"}
PARAMS = dict(objective='binary', metric='auc', device_type='cpu', learning_rate=.025, num_leaves=27,
              min_data_in_leaf=58, lambda_l2=6., feature_fraction=.62, max_bin=127, seed=7, deterministic=True,
              force_col_wise=True, verbosity=-1)
ROUNDS = 720
KS = [25, 50, 100, 200, 400, 800, 1600]


def tables(work):
    out = ['clean/features_{}.parquet']
    for d in ('tune_out', 'tune_out/weight_space'):
        for p in sorted((work / d).glob('*_train.parquet')):
            name = p.name[:-len('_train.parquet')]
            if p.name.startswith('aug_') or name in LABEL_FITTED or not (work / d / f'{name}_test.parquet').exists():
                continue
            out.append(f'{d}/{name}_{{}}.parquet')
    return out


def matrix(work, split, tabs, ids=None):
    frames = []
    for t in tabs:
        f = pd.read_parquet(work / t.format(split))
        f.index = f.index.astype(str)
        if ids is not None:
            f = f.loc[ids]
        stem = t.split('/')[-1].replace('_{}.parquet', '')
        frames.append(f.astype(np.float32).add_prefix(f'{stem}__'))
    m = pd.concat(frames, axis=1).replace([np.inf, -np.inf], np.nan)
    return m.loc[:, ~m.columns.duplicated()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--threads', type=int, default=8)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    params = dict(PARAMS, num_threads=a.threads)
    tabs = tables(a.work)
    lab = pd.read_csv(a.work / 'data/train_labels.csv', dtype={'id': str}).set_index('id')
    ids = pd.read_parquet(a.work / 'clean/features_train.parquet', columns=[]).index.astype(str)
    y = lab.loc[ids, 'label'].to_numpy()
    X = matrix(a.work, 'train', tabs, ids)
    print('matrix', X.shape, 'from', len(tabs), 'tables', flush=True)
    oof = {k: np.zeros(len(y)) for k in ['all', *KS]}
    gains = []
    for f, (fit, val) in enumerate(StratifiedKFold(5, shuffle=True, random_state=7).split(X, y)):
        t = time.time()
        m = lgb.train(params, lgb.Dataset(X.iloc[fit], label=y[fit]), ROUNDS)
        oof['all'][val] = m.predict(X.iloc[val])
        g = pd.Series(m.feature_importance('gain'), index=X.columns)
        gains.append(g)
        order = g.sort_values(ascending=False).index
        for k in KS:
            cols = order[:k]
            mk = lgb.train(params, lgb.Dataset(X.iloc[fit][cols], label=y[fit]), ROUNDS)
            oof[k][val] = mk.predict(X.iloc[val][cols])
        print(f'fold {f} done {time.time() - t:.0f}s  ' +
              '  '.join(f'{k}:{roc_auc_score(y[val], oof[k][val]):.4f}' for k in ['all', *KS]), flush=True)
    res = {str(k): float(roc_auc_score(y, v)) for k, v in oof.items()}
    imp = pd.concat(gains, axis=1).mean(1).sort_values(ascending=False)
    imp.to_csv(a.out / 'importance.csv', header=['gain'])
    fam = imp.groupby(imp.index.str.split('__').str[0]).sum().sort_values(ascending=False)
    fam.to_csv(a.out / 'family_gain.csv', header=['gain'])
    (a.out / 'results.json').write_text(json.dumps(dict(auc=res, n_features=X.shape[1], tables=tabs), indent=1))
    pd.DataFrame({'id': ids, **{f'oof_{k}': v for k, v in oof.items()}}).to_parquet(a.out / 'oof.parquet')
    print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
    main()
