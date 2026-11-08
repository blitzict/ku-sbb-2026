"""Development check: can a compact model use the existing label-free features?

Reads newly computed CPU feature tables to avoid repeating extraction during
model screening. This is NOT a from-raw runtime measurement. No old model
predictions, labels-as-features, or learned cross-fold priors are inputs.
"""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.update(CUDA_VISIBLE_DEVICES='', ONEAPI_DEVICE_SELECTOR='*:cpu')
import argparse
import json
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
PHYSICAL = ('periodic', 'innov_v2', 'trend', 'curve', 'ar_law', 'phase_law',
            'mixture_law', 'partition', 'pulse', 'state', 'volatility_law',
            'bounds', 'noise_tests', 'jump_law_pulse', 'jump_law_state',
            'jump_law_nullstate', 'jump_law_prestate', 'jump_law_smoothvol')
STATISTICAL = ('grid', 'grid2', 'dd', 'lr5', 'lrz', 'lrz6', 'gh', 'lm', 'bf', 'sn', 'harm')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--folds', type=int, default=5, choices=range(1, 6), help='Number of the fixed five folds to run')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use a fresh development output folder')
    args.output.mkdir(parents=True, exist_ok=True)
    started = monotonic()
    labels = pd.read_csv(args.features / 'data/train_labels.csv', dtype={'id': str}).set_index('id')
    ids = pd.read_parquet(args.features / 'clean/features_train.parquet', columns=[]).index
    target = labels.loc[ids, 'label'].to_numpy()
    paths = ['clean/features_train.parquet']
    paths += [f'tune_out/{name}_train.parquet' for name in STATISTICAL]
    paths += [f'tune_out/weight_space/{name}_train.parquet' for name in PHYSICAL]
    frames = []
    for number, relative in enumerate(paths):
        frame = pd.read_parquet(args.features / relative).loc[ids].astype(np.float32)
        if any(name.lower() in {'label', 'target', 'y', 'break_probability'} for name in frame.columns):
            raise ValueError(f'Forbidden target/prediction column in {relative}')
        frames.append(frame.add_prefix(f'f{number}__'))
    matrix = pd.concat(frames, axis=1).replace([np.inf, -np.inf], np.nan)
    del frames
    print('Label-free feature matrix', matrix.shape, 'built in', round(monotonic() - started, 1), 'seconds', flush=True)
    params = dict(objective='binary', metric='auc', device_type='cpu', num_threads=6,
                  learning_rate=.025, num_leaves=27, min_data_in_leaf=58,
                  lambda_l2=6., feature_fraction=.62, max_bin=127, seed=7,
                  deterministic=True, force_col_wise=True, verbosity=-1)
    predictions = np.full(len(ids), np.nan)
    fold_ids = np.full(len(ids), -1)
    timings, importance = [], []
    for fold, (fit, valid) in enumerate(StratifiedKFold(5, shuffle=True, random_state=7).split(matrix, target)):
        if fold >= args.folds:
            break
        begin = monotonic()
        model = lgb.train(params, lgb.Dataset(matrix.iloc[fit], label=target[fit]), num_boost_round=720)
        predictions[valid] = model.predict(matrix.iloc[valid], num_threads=6)
        fold_ids[valid] = fold
        model.save_model(str(args.output / f'fold{fold}.txt'))
        importance.append(model.feature_importance(importance_type='gain'))
        result = dict(fold=fold, seconds=monotonic() - begin, auc=float(roc_auc_score(target[valid], predictions[valid])))
        timings.append(result)
        print(json.dumps(result), flush=True)
        pd.DataFrame({'id': ids, 'label': target, 'fold': fold_ids, 'prediction': predictions}).to_csv(args.output / 'validation.csv', index=False)
    valid = np.isfinite(predictions)
    report = dict(scope='development using fresh CPU feature tables; raw extraction excluded',
                  model='compact fixed LightGBM baseline', features=matrix.shape[1],
                  folds=timings, completed_validation_rows=int(valid.sum()),
                  complete_five_fold_validation=bool(valid.all()),
                  auc=float(roc_auc_score(target[valid], predictions[valid])),
                  elapsed_seconds=monotonic() - started, params=params, rounds=720,
                  input_tables=paths, target_auc=.879,
                  runtime_under_one_hour_from_raw_verified=False)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    pd.DataFrame({'feature': matrix.columns, 'gain': np.mean(importance, axis=0)}).sort_values('gain', ascending=False).to_csv(args.output / 'feature_importance.csv', index=False)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
