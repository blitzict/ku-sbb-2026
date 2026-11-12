# CPU port; original experiments/weight_space/other_localization_trial.py, SHA256 2bf38bc0352b9b7409527ddff13b4237f9ea23a196ec78d60716b83647ea0e15
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Evaluate observed change timing and a matched supervised source-model pair."""
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
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from threadpoolctl import threadpool_limits
from partition_evidence import OUT
from other_response_gate import run
from white_local_rank_weights import centered, weights

def setup(mode):

    def factory(ids, tids, y):
        ix = ids.append(tids)
        files = []
        if mode == 'timing':
            parts = []
            for rows, part in ((ids, 'train'), (tids, 'test')):
                path = OUT / f'kernel_timing_{part}.parquet'
                files.append(path)
                parts.append(pd.read_parquet(path).loc[rows])
            df = pd.concat(parts)
            advantage = df.filter(regex='_center_advantage$').to_numpy()
            noise = df.filter(regex='_residual_shape$').to_numpy()
            signal = pd.Series(np.mean(advantage / (1 + noise), axis=1), index=ix)
            data = np.arcsinh(df.to_numpy())
        else:
            models = []
            for s in (7, 29):
                path = OUT / f'measured_source_s{s}.npz'
                files.append(path)
                with np.load(path) as a:
                    assert np.array_equal(a['ids'], ids) and np.array_equal(a['tids'], tids)
                    models.append(np.vstack([a['oof'], a['test']]))
            original, changed = np.mean(models, axis=0).T
            diff = changed - original
            split = np.subtract(*models)
            data = np.column_stack([original, changed, diff, np.tanh(diff), abs(diff), abs(split), diff * abs(split[:, 1])])
            signal = pd.Series(diff, index=ix)
        pairs = {}
        for group in ('other_disagreement', 'other_no_detector'):
            m = pd.read_csv(OUT / f'{group}_pairs.csv', dtype={'case_id': str, 'control_id': str})
            d = signal.loc[m.case_id].to_numpy() - signal.loc[m.control_id].to_numpy()
            wins = sum(d > 0)
            ties = sum(d == 0)
            pairs[group] = dict(wins=int(wins), ties=int(ties), n=len(d), p=float(binomtest(wins, len(d) - ties, alternative='greater').pvalue))
        print(mode, pairs, flush=True)
        (OUT / f'other_{mode}_paired.json').write_text(json.dumps(pairs, indent=2))

        def fit(tr, va, z, v, old, prior):
            mu = data[tr].mean(0)
            sd = data[tr].std(0).clip(0.07)
            h, ref = centered(((data[tr] - mu) / sd).clip(-5, 5), z)
            r, _ = centered(((data[va] - mu) / sd).clip(-5, 5), v, ref)

            def design(X, b):
                t = np.tanh((b + 0.5) / 1.6)
                return np.column_stack([X, 0.6 * t[:, None] * X])
            train, valid = (design(h, z), design(r, v))
            w = 0.65 * weights(train, z, y[tr])
            return (v + valid @ w, valid, w)
        return (fit, [*files, Path(__file__).resolve()])
    return factory
if __name__ == '__main__':
    mode = sys.argv[1]
    assert mode in ('timing', 'source')
    with threadpool_limits(limits=1):
        run(setup(mode), 'local_rank_observed_' + mode)
