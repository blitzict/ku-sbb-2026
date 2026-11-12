# CPU port; original experiments/weight_space/other_repeatability_trial.py, SHA256 07d59293574b69a3e0e38cb962fa0e102178ff2d501c141af377679963283d9a
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Test learned within-segment repeatability or adaptive noise scores on matched cases and full folds."""
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
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from threadpoolctl import threadpool_limits
from other_response_gate import run
from partition_evidence import OUT
from white_local_rank_weights import weights, centered

def factory(mode, ids, tids, y):
    paths = []
    if mode == 'identity':
        models = []
        for s in (7, 29):
            p = OUT / f'process_identity_s{s}.npz'
            paths.append(p)
            with np.load(p) as a:
                assert np.array_equal(ids, a['ids']) and np.array_equal(tids, a['tids'])
                models.append(np.vstack([a['oof'], a['test']]))
        x = np.mean(models, axis=0)
        between, repeat, pre, post = np.hsplit(x[:, 2:], 4)
        within = (pre + post) / 2
        contrast = between - within
        new = x[:, 1] - x[:, 0]
        X = np.column_stack([x[:, :2], new, between, repeat, contrast, np.abs(pre - post)])
        fixed = contrast.mean(1)
    else:
        parts = []
        for part, ix in (('train', ids), ('test', tids)):
            p = OUT / f'noise_score_{part}.parquet'
            paths.append(p)
            parts.append(pd.read_parquet(p).loc[ix])
        frame = pd.concat(parts)
        x = frame.filter(regex='_(repeat_.*|strongest)$')
        fixed = x.filter(regex='_extra_repeat_(level|scale|lag1|mode_spacing)$').mean(axis=1).to_numpy()
        X = x.to_numpy()
    signal = pd.Series(fixed, index=ids.append(tids))
    results = {}
    for group in ('other_disagreement', 'other_no_detector'):
        p = pd.read_csv(OUT / f'{group}_pairs.csv', dtype={'case_id': str, 'control_id': str})
        d = signal.loc[p.case_id].to_numpy() - signal.loc[p.control_id].to_numpy()
        w = int(sum(d > 0))
        tie = int(sum(d == 0))
        results[group] = dict(n=len(d), wins=w, ties=tie, p=float(binomtest(w, len(d) - tie, alternative='greater').pvalue))
    print(mode, results, flush=True)
    (OUT / f'other_{mode}_pairs.json').write_text(json.dumps(results, indent=2))

    def fit(tr, va, z, v, prev, prior):
        mu = X[tr].mean(0)
        sd = X[tr].std(0).clip(0.09)
        h, ref = centered(((X[tr] - mu) / sd).clip(-4, 4), z)
        r, _ = centered(((X[va] - mu) / sd).clip(-4, 4), v, ref)
        w = 0.59 * weights(h, z, y[tr])
        return (v + r @ w, r, w)
    return (fit, [*paths, Path(__file__).resolve()])
if __name__ == '__main__':
    mode = sys.argv[1]
    assert mode in ('identity', 'noise')
    with threadpool_limits(limits=1):
        run(lambda i, t, y: factory(mode, i, t, y), 'local_rank_adaptive_' + mode)
