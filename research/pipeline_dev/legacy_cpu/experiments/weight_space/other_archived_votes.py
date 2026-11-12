# CPU port; original experiments/weight_space/other_archived_votes.py, SHA256 81b24fae79e9cf602b716f65a232f3afbd30ecb8c6e736f620e1a4fa6e257541
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Revisit independent original feature-model votes on the broad current misses."""
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
from scipy.special import logit
from threadpoolctl import threadpool_limits
from other_miss_witnesses import make
from other_missed_breaks import CURRENT, rankings
from other_response_gate import run
from hard_pair_budget import cohorts
from white_local_rank_weights import centered, weights
from partition_evidence import OUT, ROOT
import white_loop as wl
MEMBERS = ('xgb_clean', 'xgb_grid', 'xgb_clean_grid', 'lgbsel6', 'extratrees')

def load(ids, tids):
    sources = []
    parts = []
    for s, index in (('train', ids), ('test', tids)):
        for folder in ('clean', 'features_old'):
            assert pd.read_parquet(ROOT / f'{folder}/features_{s}.parquet', columns=[]).index.equals(index)
        prefix = 'oof' if s == 'train' else 'test'
        vals = []
        for model in MEMBERS:
            file = ROOT / f'tune_out/{prefix}_{model}.npy'
            v = np.load(file)
            assert v.shape == (len(index),) and np.isfinite(v).all() and np.all((v >= 0) & (v <= 1))
            vals.append(logit(np.clip(v.astype(float), 1e-06, 1 - 1e-06)))
            sources.append(file)
        parts.append(np.column_stack(vals))
    return (np.vstack(parts), sources)

def diagnostic():
    _, ids, tids, y, _, _, white = wl.load()
    E, _ = load(ids, tids)
    with np.load(OUT / (CURRENT + '_oof.npz')) as a:
        base = a['s7']
    n = sum(y == 0)
    rank = rankings(y, base)
    missed = (y == 1) & (rank < 0.8 * n) & ~white
    other = cohorts(ids, y)[0] & ~white
    elementary = [rankings(y, E[:len(y), j]) >= 0.8 * n for j in range(E.shape[1])]
    hits = np.sum(elementary, axis=0)
    banks, _ = make(ids, tids)
    physics = np.sum([rankings(y, m[:len(y)].mean(1)) >= 0.8 * n for m in banks], axis=0)
    rows = []
    for label, flag in [*zip(MEMBERS, elementary), *[(f'at_least_{k}', hits >= k) for k in range(1, len(MEMBERS) + 1)]]:
        rows.append(dict(detector=label, additional_other_positives=int(sum(flag & missed)), previously_unflagged_by_six=int(sum(flag & missed & (physics == 0))), caught_other_historical=int(sum(flag & missed & other)), additional_controls=int(sum(flag & (y == 0) & (rank < 0.8 * n))), previously_unflagged_controls=int(sum(flag & (y == 0) & (rank < 0.8 * n) & (physics == 0)))))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'other_archived_votes.csv', index=False)
    print(table.to_string(index=False), flush=True)

def factory(ids, tids, y):
    E, paths = load(ids, tids)
    banks, pp = make(ids, tids)
    physics = np.column_stack([v.mean(1) for v in banks])

    def fit(tr, va, z, v, unused_train, unused_valid):
        ix = np.r_[tr, va]
        score = np.r_[z, v]
        n = len(tr)

        def calibrated(values):
            lo, med, hi = np.quantile(values[tr], [0.15, 0.5, 0.85], axis=0)
            l, m, r = np.quantile(z, [0.15, 0.5, 0.85])
            return (values[ix] - med) * (r - l) / (hi - lo).clip(0.1) + m
        older = calibrated(E)
        fresh = calibrated(physics)
        disagreement = np.arcsinh((older - score[:, None]).clip(-9, 9))
        median = np.median(older, axis=1)
        surprise = np.tanh((np.max(older, axis=1) - np.max(fresh, axis=1)) / 0.8)
        spread = np.tanh((np.max(older, axis=1) - median) / 1.2)
        risk = np.tanh((score + 0.6) / 1.6)
        h = np.column_stack([disagreement, 0.65 * disagreement * risk[:, None], 0.6 * disagreement * surprise[:, None], 0.45 * disagreement * spread[:, None], np.arcsinh(median - score), np.arcsinh(older.max(1) - score) * surprise])
        X, ref = centered(h[:n], z)
        V, _ = centered(h[n:], v, ref)
        w = 0.65 * weights(X, z, y[tr])
        return (v + V @ w, V, w)
    return (fit, [*paths, *pp, Path(__file__).resolve()])
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        diagnostic()
        run(factory, 'local_rank_original_feature_votes')
