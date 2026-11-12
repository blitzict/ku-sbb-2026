# CPU port; original experiments/weight_space/other_unit_memory.py, SHA256 a8f5467f098d93268621f635038b452e41a03c5bd5bc1a04796c012edd3fd8fb
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Distinguish evidence for a persistent pre-boundary state from its subsequent release."""
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
from scipy.special import expit
from threadpoolctl import threadpool_limits
from other_miss_witnesses import make
from other_response_gate import run
from partition_evidence import ROOT, OUT
from white_local_rank_weights import weights
import white_loop as wl

def table(ids, tids):
    paths = []
    frames = []
    for part, ix in (('train', ids), ('test', tids)):
        path = OUT / f'memory_law1_{part}.parquet'
        paths.append(path)
        frames.append(pd.read_parquet(path).loc[ix])
    return (pd.concat(frames), paths)

def diagnose():
    _, ids, tids, *_ = wl.load()
    f, _ = table(ids, tids)
    pair = pd.read_csv(OUT / 'other_disagreement_pairs.csv')
    gen = pd.read_parquet(ROOT / 'tune_out/generator_segments.parquet', columns=['id', 'part', 'phi1_ar1'])
    gen = gen[gen.part == 'pre'].set_index('id')
    r = pair[pair.replication]
    records = []
    a = f.loc[r.case_id]
    b = f.loc[r.control_id]
    for key in ('unit_start', 'unit_both', 'unit_end', 'pre_pacf', 'independent_memory', 'release', 'pre_unit_given_free'):

        def sample(df):
            if key == 'release':
                return (df.unit_start - df.unit_both).to_numpy()
            if key == 'pre_unit_given_free':
                return (df.unit_start - df.independent_memory).to_numpy()
            return df[key].to_numpy()
        u, v = (sample(a), sample(b))
        d = u - v
        records.append(dict(variable=key, n=len(d), ties=float(np.mean(d == 0)), win=float(np.mean((d > 0) + 0.5 * (d == 0))), median_case=float(np.median(u)), median_control=float(np.median(v)), mean_case=float(u.mean()), mean_control=float(v.mean()), case_floor=float(np.mean(u <= -79.999)), control_floor=float(np.mean(v <= -79.999))))
    tab = pd.DataFrame(records)
    tab.to_csv(OUT / 'other_unit_memory_components.csv', index=False)
    print(tab.to_string(index=False), flush=True)
    for key, t in (('case', a), ('control', b)):
        print(key, 'near unit pre share', float(np.mean(gen.loc[t.index, 'phi1_ar1'] > 0.9)), flush=True)

def factory(ids, tids, y):
    f, paths = table(ids, tids)
    banks, pp = make(ids, tids)
    votes = np.column_stack([b.mean(1) for b in banks])
    start = f.unit_start.to_numpy()
    both = f.unit_both.to_numpy()
    clean = (np.abs(start) < 79.99) & (np.abs(both) < 79.99)
    release = np.where(clean, start - both, 0.0)
    fixed = np.column_stack([np.arcsinh(start), expit((start + 60) / 12), np.arcsinh(release.clip(-50, 50)), clean])

    def fit(tr, va, z, v, unused_train, unused_valid):
        med = np.median(fixed[tr], axis=0)
        scale = np.std(fixed[tr], axis=0).clip(0.2)
        ix = np.r_[tr, va]
        score = np.r_[z, v]
        n = len(tr)
        neg = y[tr] == 0

        def percentile(ref, val):
            return (np.searchsorted(np.sort(ref), val) + 0.5) / (len(ref) + 1)
        current = percentile(z[neg], score)
        experts = np.column_stack([percentile(votes[tr[neg], j], votes[ix, j]) for j in range(votes.shape[1])])
        difference = experts.max(1) - current
        disagreements = expit((experts.max(1) - 0.7) / 0.12) * expit((0.82 - current) / 0.09) * expit((difference - 0.08) / 0.08)
        h = ((fixed[ix] - med) / scale).clip(-5, 5)
        h = np.column_stack([0.45 * h, h * disagreements[:, None], 0.6 * h * np.tanh(score[:, None] / 1.6)])
        w = 0.6 * weights(h[:n], z, y[tr])
        return (v + h[n:] @ w, h[n:], w)
    return (fit, [*paths, *pp, Path(__file__).resolve()])
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        diagnose()
        run(factory, 'local_rank_persistent_pre_dynamics')
