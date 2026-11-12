# CPU port; original experiments/weight_space/white_signature_screen.py, SHA256 7d539f1eb929ea3427290eef473ddb85c208201b9930a385002113ea21e4ae27
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Describe the difficult cohort using controls that also fooled all ten original models."""
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
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
from partition_evidence import ROOT, OUT
import white_loop as wl
OLD = ('xgb', 'lgb', 'xgb_clean', 'lgbsel6', 'xgb_grid', 'v5_xgb', 'v5_lgb', 'nn_cnn3', 'nn_mlp', 'extratrees')
MEASURES = ('volatility_law', 'quantile_alignment', 'local_transport_score', 'vol_smoothing', 'latent_duration', 'motif_dictionary', 'mode_residual', 'hard_signals', 'ar_law', 'noise_tests', 'markov_copula', 'asymmetric_response', 'ordinal_motifs', 'shock_spacing', 'tail_edges', 'weak_phase', 'smooth_path', 'value_spacing', 'trend', 'curve', 'bounds', 'variance_ramp', 'wandering_frequency', 'wavelet_modulation', 'boundary_law_state', 'cold_start_effect', 'conditional_drift', 'jump_law_state', 'jump_law_smoothvol')

def pairs():
    q, ids, _, y, _, wn, miss = wl.load()
    old = np.column_stack([rankdata(np.load(ROOT / f'tune_out/oof_{name}.npy')) / len(y) for name in OLD])
    low = (old < 0.5).all(1)
    assert np.array_equal(miss, low & wn & (y == 1))
    active = pd.read_parquet(OUT / 'phase_law_train.parquet').loc[ids, 'df'].to_numpy() > 0
    select = low & wn & ~active
    rows = np.flatnonzero(select)
    hard = np.flatnonzero(y[rows] == 1)
    control = np.flatnonzero(y[rows] == 0)
    lens = pd.read_parquet(ROOT / 'clean/features_train.parquet', columns=['n_pre', 'n_post']).loc[ids]
    prior = pd.read_parquet(OUT / 'innov_v2_train.parquet').loc[ids, 'pre_white'].to_numpy()
    locked = np.empty(len(ids))
    for f in range(5):
        key = f's7_f{f}_'
        locked[q[key + 'va']] = q[key + 'zv']
    raw = np.column_stack([old, old.mean(1), locked, prior, lens])[rows]
    background = rankdata(raw, axis=0) / len(rows)
    background *= np.r_[np.full(10, 0.3), 1.5, 1.5, 1.0, 0.65, 0.65]
    cost = cdist(background[hard], background[control], metric='sqeuclidean')
    left, right = linear_sum_assignment(cost)
    assert len(left) == 808 and len(np.unique(right)) == len(left)
    return (ids, rows[hard[left]], rows[control[right]], dict(eligible_controls=len(control), matched_historical=len(left), matched_controls=len(right), matching_median_absolute_gap={k: float(v) for k, v in zip(('historical_mean_rank', 'locked_logit', 'pre_white', 'pre_len', 'post_len'), np.median(abs(raw[hard[left], -5:] - raw[control[right], -5:]), axis=0))}))

def main():
    ids, a, b, meta = pairs()
    rng = np.random.default_rng(723901)
    discover, replicate = np.array_split(rng.permutation(len(a)), 2)
    flips = (2 * rng.integers(0, 2, (16000, len(replicate))) - 1).astype(np.float64)
    results = []
    for fam in MEASURES:
        frame = pd.read_parquet(OUT / f'{fam}_train.parquet').loc[ids]
        vals = frame.to_numpy(dtype=float)[np.r_[a, b]]
        assert np.isfinite(vals).all()
        ranked = rankdata(vals, axis=0) / len(vals)
        effect = ranked[:len(a)] - ranked[len(a):]
        d = effect[discover]
        t = d.mean(0) / d.std(0).clip(0.015) * np.sqrt(len(d))
        j = int(np.argmax(abs(t)))
        direction = np.sign(t[j])
        v = direction * effect[replicate, j]
        p = (1 + (flips @ v >= v.sum() - 1e-12).sum()) / (1 + len(flips))
        results.append(dict(family=fam, feature=str(frame.columns[j]), features=len(t), direction=int(direction), discovery_gap=float(d[:, j].mean()), replication_gap=float(effect[replicate, j].mean()), discovery_z=float(t[j]), p=float(p)))
    res = pd.DataFrame(results).sort_values('p')
    res['holm_p'] = np.minimum(1, np.maximum.accumulate(res.p.to_numpy() * np.arange(len(res), 0, -1)))
    meta.update(discovery=len(discover), replication=len(replicate), permutations=len(flips), candidates=int(res.features.sum()), families=len(res), holm_pass=int((res.holm_p < 0.05).sum()))
    res.to_csv(OUT / 'signature_screen_replication.csv', index=False)
    (OUT / 'signature_screen_matching.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)
    print(res[['family', 'feature', 'discovery_gap', 'replication_gap', 'holm_p']].head(12).to_string(index=False), flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
