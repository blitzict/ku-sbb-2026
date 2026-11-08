"""train the 0.870 lightgbm (5 folds) on the cpu feature tables and save everything needed to predict later.

python compact_0870/train_compact.py --work B:/AI_cpu_087910_20261108/work --out compact_0870/model
"""
import os
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[k] = '1'
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

#same tables as the 0.8699 probe
STAT = ('grid', 'grid2', 'dd', 'lr5', 'lrz', 'lrz6', 'gh', 'lm', 'bf', 'sn', 'harm')
PHYS = ('periodic', 'innov_v2', 'trend', 'curve', 'ar_law', 'phase_law', 'mixture_law', 'partition', 'pulse', 'state',
        'volatility_law', 'bounds', 'noise_tests', 'jump_law_pulse', 'jump_law_state', 'jump_law_nullstate',
        'jump_law_prestate', 'jump_law_smoothvol')
TABLES = ['clean/features_{}.parquet'] + [f'tune_out/{n}_{{}}.parquet' for n in STAT] + [f'tune_out/weight_space/{n}_{{}}.parquet' for n in PHYS]
PARAMS = dict(objective='binary', metric='auc', device_type='cpu', num_threads=6, learning_rate=.025, num_leaves=27,
              min_data_in_leaf=58, lambda_l2=6., feature_fraction=.62, max_bin=127, seed=7, deterministic=True,
              force_col_wise=True, verbosity=-1)
ROUNDS = 720


def matrix(work, split, ids=None, columns=None):
    #glue all feature tables side by side, prefix = table number so names never clash
    frames = []
    for i, t in enumerate(TABLES):
        f = pd.read_parquet(Path(work) / t.format(split))
        f.index = f.index.astype(str)
        if ids is not None:
            f = f.loc[ids]
        frames.append(f.astype(np.float32).add_prefix(f'f{i}__'))
    m = pd.concat(frames, axis=1).replace([np.inf, -np.inf], np.nan)
    return m if columns is None else m[columns]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    lab = pd.read_csv(a.work / 'data/train_labels.csv', dtype={'id': str}).set_index('id')
    ids = pd.read_parquet(a.work / 'clean/features_train.parquet', columns=[]).index.astype(str)
    y = lab.loc[ids, 'label'].to_numpy()
    X = matrix(a.work, 'train', ids)
    print('train matrix', X.shape, flush=True)
    oof = np.zeros(len(y))
    for k, (fit, val) in enumerate(StratifiedKFold(5, shuffle=True, random_state=7).split(X, y)):
        m = lgb.train(PARAMS, lgb.Dataset(X.iloc[fit], label=y[fit]), num_boost_round=ROUNDS)
        oof[val] = m.predict(X.iloc[val])
        m.save_model(str(a.out / f'fold{k}.txt'))
        print('fold', k, roc_auc_score(y[val], oof[val]), flush=True)
    auc = roc_auc_score(y, oof)
    (a.out / 'columns.json').write_text(json.dumps(list(X.columns)))
    (a.out / 'model.json').write_text(json.dumps(dict(model='lightgbm 5-fold average', oof_auc=auc, rounds=ROUNDS,
                                                       params=PARAMS, tables=TABLES), indent=2))
    pd.DataFrame({'id': ids, 'label': y, 'oof': oof}).to_csv(a.out / 'oof.csv', index=False)
    #test predictions straight from the tables the training run made
    T = matrix(a.work, 'test', columns=list(X.columns))
    pred = np.mean([lgb.Booster(model_file=str(a.out / f'fold{k}.txt')).predict(T) for k in range(5)], axis=0)
    pd.DataFrame({'id': T.index, 'break_probability': pred}).to_csv(a.out / 'submission_from_tables.csv', index=False)
    print('oof auc', auc, flush=True)


if __name__ == '__main__':
    main()
