"""Small CPU-only feature profiling, isolated from the paused training run."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.update(CUDA_VISIBLE_DEVICES='', ONEAPI_DEVICE_SELECTOR='*:cpu')
import argparse
import cProfile
import importlib
import json
from pathlib import Path
import pstats
import shutil
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]


def workspace():
    work = ROOT / 'cpu_088_rebuild/cpu_speedup_v1'
    work.mkdir(parents=True, exist_ok=True)
    (work / 'CPU_REBUILD_WORKSPACE.json').write_text(json.dumps({'kind': 'cpu_performance_test_only'}))
    for source, relative in [
        (ROOT / 'pipeline/features_v4.py', 'clean/features.py'),
        (ROOT / 'pipeline/legacy_cpu/ku_workspace.py', 'ku_workspace.py'),
        *[(ROOT / 'pipeline/legacy_cpu' / path, path) for path in (
            'experiments/v5/features_grid.py', 'experiments/v5/features_grid2.py', 'experiments/v6/features_dd.py')],
    ]:
        target = work / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
    os.environ.update(KU_SBB_ROOT=str(work), KU_SBB_CODE_ROOT=str(work))
    sys.path[:0] = [str(work), str(work / 'clean'), str(work / 'experiments/v5'), str(work / 'experiments/v6')]
    return work


def sample_rows(count):
    import pyarrow.parquet as pq
    frame = next(pq.ParquetFile(ROOT / 'data/train_series.parquet').iter_batches(
        batch_size=max(65536, (count + 1) * 1024), columns=['id', 'time', 'period', 'value'])).to_pandas()
    jobs = []
    for ident, series in frame.groupby('id', sort=False, observed=True):
        series = series.sort_values('time')
        values = series.value.to_numpy(float)
        period = series.period.to_numpy()
        jobs.append((str(ident), values[period == 0], values[period == 1]))
        if len(jobs) == count:
            break
    if len(jobs) != count:
        raise ValueError('Insufficient complete sample rows')
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=32)
    args = parser.parse_args()
    work = workspace()
    jobs = sample_rows(args.count)
    families = [('clean', 'features', 'series_features'), ('grid', 'features_grid', 'one'),
                ('grid2', 'features_grid2', 'one'), ('dd', 'features_dd', 'one')]
    timing = {}
    for name, module, function in families:
        calculate = getattr(importlib.import_module(module), function)
        profile = cProfile.Profile()
        started = perf_counter()
        profile.enable()
        for ident, a, b in jobs:
            calculate(ident, a, b)
        profile.disable()
        timing[name] = perf_counter() - started
        with (work / (name + '_profile.txt')).open('w') as output:
            pstats.Stats(profile, stream=output).sort_stats('cumulative').print_stats(35)
        print(name, round(timing[name], 3), 'seconds', flush=True)
    (work / 'profile_timings.json').write_text(json.dumps({'count': args.count, 'seconds_with_profiler': timing}, indent=2))


if __name__ == '__main__':
    main()
