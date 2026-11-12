# CPU port; original experiments/weight_space/other_neighbor_trial.py, SHA256 5130013de0327703a70c24f445d6598791ec5ae50f5607a8c9cd3c7791d2d86c
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Test whether short-transition geometry identifies missed breaks at fixed parent score."""
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
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from threadpoolctl import threadpool_limits
from other_response_gate import run
from partition_evidence import OUT
from white_local_rank_weights import centered, weights

def source(ids, tids, y):
    paths = [OUT / f'transition_neighbors_{s}.parquet' for s in ('train', 'test')]
    table = pd.concat([pd.read_parquet(p).loc[i] for p, i in zip(paths, (ids, tids))])
    stable = table.filter(regex='_held_k12_separation$').mean(axis=1)
    assert len(table.filter(regex='_held_k12_separation$').columns) == 4
    matched = {}
    for group in ('other_disagreement', 'other_no_detector'):
        pairs = pd.read_csv(OUT / f'{group}_pairs.csv', dtype={'case_id': str, 'control_id': str})
        gap = stable.loc[pairs.case_id].to_numpy() - stable.loc[pairs.control_id].to_numpy()
        tied = gap == 0
        matched[group] = dict(pairs=len(gap), wins=int(sum(gap > 0)), ties=int(sum(tied)), fraction=float(np.mean((gap > 0) + 0.5 * tied)), p=float(binomtest(sum(gap > 0), sum(~tied), alternative='greater').pvalue))
    print('Fixed neighborhood separation', matched, flush=True)
    (OUT / 'other_neighbor_pairs.json').write_text(json.dumps(matched, indent=2))
    features = np.arcsinh(table.to_numpy())

    def fit(tr, va, z, v, prev, valid_prev):
        mu = features[tr].mean(0)
        sd = features[tr].std(0).clip(0.08)
        train, ref = centered(((features[tr] - mu) / sd).clip(-4, 4), z)
        held, _ = centered(((features[va] - mu) / sd).clip(-4, 4), v, ref)

        def design(a, logits):
            near = np.exp(-0.5 * ((logits + 0.7) / 1.35) ** 2)
            return np.column_stack([a, 0.6 * near[:, None] * a])
        x, h = (design(train, z), design(held, v))
        w = 0.6 * weights(x, z, y[tr])
        return (v + h @ w, h, w)
    return (fit, [*paths, Path(__file__).resolve()])
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run(source, 'local_rank_transition_neighbors')
