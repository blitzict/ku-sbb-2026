# CPU port; original experiments/weight_space/other_below_cutoff.py, SHA256 e1003babc1811e0807a231fd55b00db359c2aa8b1649797bcffbe2961faf3ac9
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Test a score correction confined to observable below-cutoff series.

Both sides of each training comparison come from training-fold labels. No
miss-list membership or validation-fold cutoff enters the classifier.
"""
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
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from other_archived_votes import load
from other_miss_witnesses import make
from other_response_gate import run
from partition_evidence import OUT
from white_local_rank_weights import weights

def factory(ids, tids, y):
    older, paths = load(ids, tids)
    banks, files = make(ids, tids)
    physics = np.column_stack([a.mean(1) for a in banks])
    streams = []
    for name, cols in (('transition_neighbors', None), ('memory_law1', ['unit_start'])):
        parts = []
        for part, ix in (('train', ids), ('test', tids)):
            path = OUT / f'{name}_{part}.parquet'
            paths.append(path)
            t = pd.read_parquet(path, columns=cols).loc[ix]
            if cols is None:
                t = t.filter(regex='_held_k12_separation$')
            parts.append(t.mean(axis=1).to_numpy())
        streams.append(np.concatenate(parts))
    features = np.column_stack([*streams, older, physics])

    def fit(tr, va, z, v, p, old):
        baseline = np.r_[z, v]
        n = len(tr)
        ix = np.r_[tr, va]
        controls = tr[y[tr] == 0]
        cutoff = np.quantile(z[y[tr] == 0], 0.8)
        negative_mid, negative_high = np.quantile(z[y[tr] == 0], [0.5, 0.8])

        def percentiles(a):
            return np.column_stack([np.searchsorted(np.sort(a[controls, k]), a[ix, k]) / len(controls) for k in range(a.shape[1])])
        rates = percentiles(features)
        classic = rates[:, 2:2 + older.shape[1]]
        recent = rates[:, 2 + older.shape[1]:]
        fade = np.clip((cutoff - baseline) / 0.4, 0, 1)
        fresh = recent.max(1) >= 0.8
        legacy = classic.max(1) >= 0.8
        patterns = np.column_stack([~fresh & ~legacy, fresh & ~legacy, ~fresh & legacy, fresh & legacy])
        rank = np.tanh((baseline - negative_mid) / max(negative_high - negative_mid, 0.1))
        info = np.column_stack([np.ones(len(ix)), rank, rank ** 2, 2 * rates[:, :2] - 1, 2 * classic.max(1) - 1, 2 * recent.max(1) - 1, classic.mean(1) - recent.mean(1)])
        h = (fade[:, None, None] * patterns[:, :, None] * info[:, None, :]).reshape(len(ix), -1)
        w = 0.65 * weights(h[:n], z, y[tr])
        return (v + h[n:] @ w, h[n:], w)
    return (fit, [*paths, *files, Path(__file__).resolve()])
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run(factory, 'local_rank_below_cutoff')
