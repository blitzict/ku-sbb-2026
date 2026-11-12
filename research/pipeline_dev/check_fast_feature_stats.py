"""Compare every feature column, plus edge cases, on CPU without fitting."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import importlib
import json
from time import perf_counter, process_time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from pipeline.profile_cpu_features import workspace, sample_rows
from pipeline.fast_feature_stats import FeatureStats, install


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=128)
    parser.add_argument('--paired', action='store_true', help='Alternate baseline/fast blocks; record CPU time to identify contention')
    args = parser.parse_args()
    work = workspace()
    jobs = sample_rows(args.count)
    modules = [importlib.import_module(name) for name in ('features', 'features_grid', 'features_grid2', 'features_dd')]
    functions = [modules[0].series_features, *[module.one for module in modules[1:]]]
    fast = FeatureStats()
    rng = np.random.default_rng(173)
    checked = 0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for dtype in (np.float64, np.float32, np.int64):
            for n, m in ((1, 2), (8, 9), (17, 31), (80, 80), (310, 451), (1024, 513)):
                for kind in ('normal', 'ties', 'constant', 'tail'):
                    a, b = rng.standard_t(4, size=n), rng.standard_t(4, size=m)
                    if kind == 'ties':
                        a, b = a.round(), b.round()
                    if kind == 'constant':
                        a, b = a * 0, b * 0
                    if kind == 'tail':
                        b *= 100
                    a, b = a.astype(dtype), b.astype(dtype)
                    for method in ('auto', 'asymp', 'exact'):
                        np.testing.assert_equal(fast.ks_2samp(a, b, method=method).statistic,
                                                stats.ks_2samp(a, b, method=method).statistic)
                        checked += 1
        timings = []
        # Baseline and optimized paths use the same raw rows in the same process.
        for family, function in zip(('clean', 'grid', 'grid2', 'dd'), functions):
            baseline = elapsed = baseline_cpu = fast_cpu = 0.
            old_rows, new_rows = [], []
            size = 8 if args.paired else len(jobs)
            for block in range(0, len(jobs), size):
                for use_fast in ((False, True) if (block // size) % 2 == 0 else (True, False)):
                    for module in modules:
                        if hasattr(module, 'stats'):
                            module.stats = stats
                    if use_fast:
                        install(*modules)
                    begin, cpu_begin = perf_counter(), process_time()
                    rows = [function(*job) for job in jobs[block:block + size]]
                    seconds, cpu_seconds = perf_counter() - begin, process_time() - cpu_begin
                    if use_fast:
                        new_rows.extend(rows)
                        elapsed += seconds
                        fast_cpu += cpu_seconds
                    else:
                        old_rows.extend(rows)
                        baseline += seconds
                        baseline_cpu += cpu_seconds
            original = pd.DataFrame(old_rows).set_index('id')
            improved = pd.DataFrame(new_rows).set_index('id')
            pd.testing.assert_frame_equal(original, improved, check_exact=True)
            a = original.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
            b = improved.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
            pd.testing.assert_frame_equal(a, b, check_exact=True)
            timings.append(dict(family=family, rows=len(jobs), columns=a.shape[1],
                                baseline_seconds=baseline, optimized_seconds=elapsed,
                                baseline_cpu_seconds=baseline_cpu, optimized_cpu_seconds=fast_cpu,
                                speedup=baseline / elapsed, raw_and_saved_values_exact=True))
            print(json.dumps(timings[-1]), flush=True)
    report = dict(cpu_only=True, no_model_fit=True, ks_edge_checks=checked, families=timings,
                  overall_speedup=sum(x['baseline_seconds'] for x in timings) / sum(x['optimized_seconds'] for x in timings))
    suffix = '_paired' if args.paired else ''
    (work / f'fast_statistics_{args.count}{suffix}.json').write_text(json.dumps(report, indent=2))
    print('PASS', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
