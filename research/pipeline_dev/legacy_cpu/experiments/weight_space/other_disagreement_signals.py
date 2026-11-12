# CPU port; original experiments/weight_space/other_disagreement_signals.py, SHA256 587b5b9aa3eff107311550ced726f079b817d94a0159c6847fa7f30354badaff
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Discover and separately check physical signatures of genuine detector disagreements."""
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
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
from other_missed_breaks import CURRENT, rankings
from other_miss_witnesses import make
from white_signature_screen import OLD
from partition_evidence import ROOT, OUT
import white_loop as wl
FAMILIES = ('ar_law', 'memory_law1', 'phase_law', 'jump_law_smoothvol', 'volatility_law', 'mode_residual', 'noise_tests', 'trend', 'boundary_law_state', 'other_memory_horizons', 'forecast_weight_change', 'tree_channels_named')

def matching(any_vote=True):
    q, ids, tids, y, _, wn, white = wl.load()
    with np.load(OUT / (CURRENT + '_oof.npz')) as a:
        base = a['s7']
    baseline = rankings(y, base) / sum(y == 0)
    banks, _ = make(ids, tids)
    votes = np.column_stack([rankings(y, g[:len(y)].mean(1)) / sum(y == 0) for g in banks])
    agreement = (votes >= 0.8).sum(1)
    low = np.column_stack([rankdata(np.load(ROOT / f'tune_out/oof_{tag}.npy')) / len(y) < 0.5 for tag in OLD]).all(1)
    assert np.array_equal(white, wn & low & (y == 1))
    eligible = (baseline < 0.8) & ((agreement > 0) == any_vote) & ~(wn & low)
    gen = pd.read_parquet(ROOT / 'tune_out/generator_segments.parquet', columns=['id', 'part', 'p_bic', 'phi1_ar1'])
    gen = gen[gen.part == 'pre'].set_index('id').loc[ids]
    phi = gen.phi1_ar1.to_numpy()
    fam = np.select([phi > 0.95, gen.p_bic.to_numpy() == 0, gen.p_bic.to_numpy() >= 5, phi < -0.2, phi > 0.2], [0, 1, 2, 3, 4], 5)
    length = np.log(pd.read_parquet(ROOT / 'clean/features_train.parquet', columns=['n_pre', 'n_post']).loc[ids])
    matrix = np.column_stack([baseline * 12, votes * 2.5, phi * 2, length.to_numpy() * 0.65, agreement * 1.2])
    ca, co = ([], [])
    for family in np.unique(fam):
        a = np.flatnonzero(eligible & (y == 1) & (fam == family))
        b = np.flatnonzero(eligible & (y == 0) & (fam == family))
        i, j = linear_sum_assignment(cdist(matrix[a], matrix[b], 'sqeuclidean'))
        ca.extend(a[i])
        co.extend(b[j])
    ca, co = (np.array(ca), np.array(co))
    assert len(np.unique(co)) == len(co) and (not np.any(white[ca]))
    return (ids, ca, co, dict(eligible_cases=int(sum(eligible & (y == 1))), eligible_controls=int(sum(eligible & (y == 0))), matched=len(ca), median_parent_neg_percentile_gap=float(np.median(abs(baseline[ca] - baseline[co]))), same_number_of_agreeing_groups=float(np.mean(agreement[ca] == agreement[co]))))

def main(any_vote=True):
    ids, a, b, meta = matching(any_vote)
    tag = 'other_disagreement' if any_vote else 'other_no_detector'
    rng = np.random.default_rng(33391 if any_vote else 50821)
    discovery, replication = np.array_split(rng.permutation(len(a)), 2)
    meta.update(discovery=len(discovery), replication=len(replication), scope='Disjoint pairs on the same reused local labelled cohort. Both cases and controls satisfy the same score and original-vote restrictions.')
    flips = (2 * rng.integers(2, size=(24576, len(replication)), dtype=np.int8) - 1).astype(float)
    rows = []
    for name in FAMILIES:
        table = pd.read_parquet(OUT / f'{name}_train.parquet').loc[ids]
        vals = table.to_numpy(dtype=float)[np.r_[a, b]]
        assert np.isfinite(vals).all()
        ranked = rankdata(vals, axis=0) / len(vals)
        effect = ranked[:len(a)] - ranked[len(a):]
        d = effect[discovery]
        t = d.mean(0) / d.std(0).clip(0.02)
        k = int(np.argmax(abs(t)))
        sign = 1 if t[k] >= 0 else -1
        test = effect[replication, k] * sign
        p = (1 + np.sum(flips @ test >= test.sum())) / (1 + len(flips))
        rows.append(dict(family=name, measurement=table.columns[k], direction=int(sign), variables=table.shape[1], discovery_rank_gap=float(sign * d[:, k].mean()), replication_rank_gap=float(test.mean()), replication_pair_win=float(np.mean((test > 0) + 0.5 * (test == 0))), one_sided_p=float(p)))
    tab = pd.DataFrame(rows).sort_values('one_sided_p')
    tab['holm_adjusted_p'] = np.minimum(1, np.maximum.accumulate(tab.one_sided_p.to_numpy() * np.arange(len(tab), 0, -1)))
    pd.DataFrame(dict(case_id=ids[a], control_id=ids[b], replication=np.isin(np.arange(len(a)), replication))).to_csv(OUT / f'{tag}_pairs.csv', index=False)
    tab.to_csv(OUT / f'{tag}_replication.csv', index=False)
    (OUT / f'{tag}_matching.json').write_text(json.dumps(meta, indent=2))
    print(meta, flush=True)
    print(tab.head(8).to_string(index=False), flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main('no_vote' not in sys.argv[1:])
