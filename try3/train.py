"""try3 training: few features + fold-wise noise-law prior + a small blend.

python try3/train.py --work <training work folder> --out try3/model --k 100

#folds = the seed-7 folds the law priors were fitted on (tune_out/weight_space/law_outer7/fold*_rows.npz)
#per fold: rank features on the fold's training rows, keep top k, add that fold's law-prior columns
#(prior fitted on the fold's training labels only), fit a few models, score the held-out rows
"""
import os
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[k] = '1'
import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_features import PARAMS, matrix, tables  # noqa: E402

WS = 'tune_out/weight_space'


def law(work, seed, fold, part, ids):
    f = pd.read_parquet(work / WS / f'law_outer{seed}/fold{fold}_{part}.parquet')
    f.index = f.index.astype(str)
    return f.loc[ids].add_prefix('law__').astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--k', type=int, default=100)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--seed', type=int, default=7)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    lp = dict(PARAMS, num_threads=a.threads)
    tabs = tables(a.work)
    lab = pd.read_csv(a.work / 'data/train_labels.csv', dtype={'id': str}).set_index('id')['label']
    X = matrix(a.work, 'train', tabs)
    T = matrix(a.work, 'test', tabs)[X.columns]
    y = lab.loc[X.index].to_numpy()
    pos = pd.Series(np.arange(len(X)), index=X.index)
    names = ['lgb', 'lgb_law', 'xgb_law', 'lgb_law_s2']
    oof = {n: np.zeros(len(y)) for n in names}
    test = {n: np.zeros(len(T)) for n in names}
    chosen = []
    for f in range(5):
        t0 = time.time()
        rows = np.load(a.work / WS / f'law_outer{a.seed}/fold{f}_rows.npz')
        fit, val = pos.loc[rows['train_ids'].astype(str)].to_numpy(), pos.loc[rows['validation_ids'].astype(str)].to_numpy()
        #rank on this fold's training rows only
        m0 = lgb.train(lp, lgb.Dataset(X.iloc[fit], label=y[fit]), 720)
        cols = pd.Series(m0.feature_importance('gain'), index=X.columns).sort_values(ascending=False).index[:a.k]
        chosen.append(list(cols))
        Ltr, Lte = law(a.work, a.seed, f, 'train', X.index), law(a.work, a.seed, f, 'test', T.index)
        A, B = pd.concat([X[cols], Ltr], axis=1), pd.concat([T[cols], Lte], axis=1)
        fits = {
            'lgb': lambda: lgb.train(lp, lgb.Dataset(X.iloc[fit][cols], label=y[fit]), 720),
            'lgb_law': lambda: lgb.train(lp, lgb.Dataset(A.iloc[fit], label=y[fit]), 720),
            'lgb_law_s2': lambda: lgb.train(dict(lp, seed=29, learning_rate=.015, feature_fraction=.5), lgb.Dataset(A.iloc[fit], label=y[fit]), 1300),
        }
        for n, make in fits.items():
            m = make()
            src_val = X.iloc[val][cols] if n == 'lgb' else A.iloc[val]
            src_te = T[cols] if n == 'lgb' else B
            oof[n][val] = m.predict(src_val)
            test[n] += m.predict(src_te) / 5
            m.save_model(str(a.out / f'{n}_fold{f}.txt'))
        xp = dict(objective='binary:logistic', eval_metric='auc', tree_method='hist', device='cuda', max_depth=6,
                  learning_rate=.02, subsample=.8, colsample_bytree=.6, min_child_weight=4, reg_lambda=6, seed=7)
        d = xgb.DMatrix(A.iloc[fit], label=y[fit])
        mx = xgb.train(xp, d, 1100)
        oof['xgb_law'][val] = mx.predict(xgb.DMatrix(A.iloc[val]))
        test['xgb_law'] += mx.predict(xgb.DMatrix(B)) / 5
        mx.save_model(str(a.out / f'xgb_law_fold{f}.json'))
        (a.out / f'columns_fold{f}.json').write_text(json.dumps({'top': list(cols), 'law': list(Ltr.columns)}))
        print(f'fold {f} {time.time() - t0:.0f}s ' + ' '.join(f'{n}:{roc_auc_score(y[val], oof[n][val]):.4f}' for n in names), flush=True)
    res = {n: float(roc_auc_score(y, v)) for n, v in oof.items()}
    blend_names = ['lgb_law', 'xgb_law', 'lgb_law_s2']
    #rank-average blend, ranks inside each fold's validation set would differ; use pooled ranks (monotone per model)
    blend = np.mean([rankdata(oof[n]) for n in blend_names], axis=0)
    res['blend'] = float(roc_auc_score(y, blend))
    tb = np.mean([rankdata(test[n]) / len(T) for n in blend_names], axis=0)
    uniq = sorted({c for cs in chosen for c in cs})
    res.update(k=a.k, distinct_features_used=len(uniq), law_columns=int(Ltr.shape[1]))
    (a.out / 'results.json').write_text(json.dumps(res, indent=1))
    (a.out / 'features.json').write_text(json.dumps({'per_fold': chosen, 'union': uniq}, indent=1))
    pd.DataFrame({'id': T.index, 'break_probability': tb}).to_csv(a.out / 'submission.csv', index=False)
    #rank blend needs the raw per-model test scores, kept so new data can be checked against them
    pd.DataFrame({'id': T.index, **{n: test[n] for n in names}}).to_parquet(a.out / 'test_scores.parquet')
    print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
    main()
