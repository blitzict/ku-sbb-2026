"""load the saved 0.870 lightgbm and score new raw series.

python compact_0870/predict_compact.py --series new_series.parquet --csv submission.csv

--ref is the training work folder: it has the frozen feature code and the training-side reference tables
some feature formulas need (train series, train features, fitted streams). nothing is retrained.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_compact import matrix  # noqa: E402

#test-side feature steps from pipeline/full_cpu_stages.json that the 0.870 tables need, in training order
STAGES = ['features_004_features_grid', 'features_006_features_grid2', 'features_009_features_dd',
          'features_015_features_lr', 'features_018_features_lrz', 'features_021_features_lrz',
          'features_024_features_garch', 'features_027_features_lm', 'features_030_features_bf',
          'features_extra_sn_test', 'mechanism_periodic_test', 'mechanism_innov_v2_test', 'mechanism_trend_test',
          'mechanism_curve_test', 'mechanism_ar_law_test', 'mechanism_phase_law_test', 'mechanism_mixture_law_test',
          'mechanism_partition_test', 'mechanism_harmonic_test', 'mechanism_pulse_test', 'mechanism_state_test',
          'mechanism_volatility_law_test', 'mechanism_jump_law_pulse_test', 'mechanism_jump_law_state_test',
          'mechanism_noise_tests', 'mechanism_bounds', 'mechanism_alt_innov_test', 'mechanism_jump_law_nullstate_test',
          'mechanism_jump_law_prestate_test', 'mechanism_jump_law_smoothvol_test', 'features_034_copy_table']
#training-side files the test formulas read (copied, never changed)
REFERENCE_STAGES = ['mechanism_periodic_train', 'mechanism_innov_v2_train', 'mechanism_noise_tests', 'mechanism_bounds']
REFERENCE_FILES = ['data/train_series.parquet', 'data/train_labels.csv', 'clean/features_train.parquet']
SKIP_DIRS = {'stages', 'logs', 'weights', 'weights_experiments', 'data', 'tune_out', 'new weights', 'submissions', 'outputs'}


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    shutil.copy2(src, dst)       #real copy so the training folder can never be touched


def setup(ref, out, stages):
    #frozen feature source code
    for p in ref.rglob('*.py'):
        rel = p.relative_to(ref)
        if rel.parts[0] not in SKIP_DIRS:
            copy(p, out / rel)
    for p in (ref / 'tune_out').glob('*.csv'):
        copy(p, out / 'tune_out' / p.name)
    #training-side reference tables
    files = list(REFERENCE_FILES)
    for s in REFERENCE_STAGES:
        files += [o for o in stages[s]['outputs'] if '_train' in o or 'train_' in o]
    for s in STAGES:
        files += [i for i in stages[s]['inputs'] if 'train' in i]
    for rel in dict.fromkeys(files):
        copy(ref / rel, out / rel)
    (out / 'CPU_REBUILD_WORKSPACE.json').write_text(json.dumps({'mode': 'prediction'}))


def run_stage(out, stage):
    #same environment the training runner gives each step
    env = {**os.environ, 'KU_SBB_ROOT': str(out), 'KU_SBB_CODE_ROOT': str(out), 'KU_SBB_THREADS': '2',
           'KU_SBB_RAW_ONLY': '1', 'KU_SBB_ORIGINAL_ROOT': str(ROOT), 'KU_SBB_OFFLINE': '1', 'CUDA_VISIBLE_DEVICES': '',
           'ONEAPI_DEVICE_SELECTOR': '*:cpu', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'OPENBLAS_NUM_THREADS': '1', 'PYTHONUNBUFFERED': '1',
           'PYTHONPATH': os.pathsep.join([str(ROOT / 'pipeline/offline_hooks'), str(out)])}
    env.update({k: str(v) for k, v in stage.get('env', {}).items()})
    (out / 'logs').mkdir(exist_ok=True)
    with open(out / 'logs' / f"{stage['id']}.log", 'w') as log:
        r = subprocess.run([sys.executable, '-B', str(out / stage['argv'][0]), *stage['argv'][1:]], cwd=out, env=env,
                           stdout=log, stderr=subprocess.STDOUT)
    if r.returncode:
        raise SystemExit(f"{stage['id']} failed, see {out / 'logs' / (stage['id'] + '.log')}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--series', type=Path, required=True)
    p.add_argument('--model', type=Path, default=Path(__file__).resolve().parent / 'model')
    p.add_argument('--ref', type=Path, default=Path('B:/AI_cpu_087910_20261108/work'))
    p.add_argument('--out', type=Path, default=Path('predict_0870'))
    p.add_argument('--csv', type=Path, default=Path('submission.csv'))
    p.add_argument('--workers', type=int, default=2)
    a = p.parse_args()
    ref, out = a.ref.resolve(), a.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit('use an empty --out folder')
    stages = {s['id']: s for s in json.loads((ROOT / 'pipeline/full_cpu_stages.json').read_text())['stages']}
    setup(ref, out, stages)

    from pipeline.inference import common, raw_features
    common.bind(out, ref)
    #1 raw series -> 977 base features
    print('base features', flush=True)
    raw_features.foundation(ref, out, a.series, a.workers)
    #2 generator segment table (training rows kept, new rows measured)
    print('generator', flush=True)
    raw_features.generator(ref, out)
    #3 the other feature blocks, same scripts as training, test part only
    for s in STAGES:
        print('stage', s, flush=True)
        run_stage(out, stages[s])
    #4 load the 5 saved fold models and average
    cols = json.loads((a.model / 'columns.json').read_text())
    X = matrix(out, 'test', columns=cols)
    pred = np.mean([lgb.Booster(model_file=str(a.model / f'fold{k}.txt')).predict(X) for k in range(5)], axis=0)
    back = {m['internal_id']: m['id'] for m in json.loads((out / 'query_ids.json').read_text())}
    pd.DataFrame({'id': [back[i] for i in X.index], 'break_probability': pred}).to_csv(a.csv, index=False)
    print('wrote', a.csv, len(pred), 'rows', flush=True)


if __name__ == '__main__':
    main()
