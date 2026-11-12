# CPU port; original experiments/weight_space/other_missed_breaks.py, SHA256 79f6c11701229b909c27ecd301b49c3a6dc4fc2d90ef77b704ee98d5f58c7d40
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Account for pairwise ranking errors outside the historical white-noise misses."""
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
from hard_pair_budget import cohorts
from partition_evidence import OUT, ROOT
import white_loop as wl
CURRENT = 'local_rank_shared_residual_response'
PREVIOUS = 'local_rank_without_multiview_trees'

def rankings(y, score):
    negative = np.sort(score[y == 0])
    return 0.5 * (np.searchsorted(negative, score, side='left') + np.searchsorted(negative, score, side='right'))

def main():
    q, ids, _, y, _, _, white = wl.load()
    hist, _ = cohorts(ids, y)

    def prediction(name):
        with np.load(OUT / (name + '_oof.npz')) as data:
            assert np.array_equal(data['ids'], ids)
            return data['s7']
    latest = prediction(CURRENT)
    old = np.empty(len(y))
    for f in range(5):
        k = f's7_f{f}_'
        old[q[k + 'va']] = q[k + 'zv']
    ranks, locked, prev = [rankings(y, a) for a in (latest, old, prediction(PREVIOUS))]
    nn = int(sum(y == 0))
    total = nn * int(sum(y == 1))
    need = round(0.88 * total - ranks[y == 1].sum())
    source = pd.read_parquet(ROOT / 'tune_out/generator_segments.parquet', columns=['id', 'part', 'phi1_ar1', 'p_bic'])
    source = source[source.part == 'pre'].set_index('id').loc[ids]
    phi, p = (source.phi1_ar1.to_numpy(), source.p_bic.to_numpy())
    fam = np.select([phi > 0.95, p >= 5, p == 0, phi < -0.2, phi > 0.2], ['near_unit_root', 'higher_order', 'white', 'negative_AR', 'positive_AR'], 'low_AR')
    other = hist & ~white
    tp = (y == 1) & (ranks >= 0.8 * nn)
    populations = {'all_breaks': y == 1, 'historical_821': white, 'other_historical_643': other, 'outside_historical_1464': (y == 1) & ~hist, 'all_positives_outside_821': (y == 1) & ~white}
    populations.update({'other_historical_' + name: other & (fam == name) for name in np.unique(fam[other])})
    populations.update({'still_missed_outside_821_' + name: (y == 1) & ~white & ~tp & (fam == name) for name in np.unique(fam)})
    rows = []
    for label, mask in populations.items():
        actual = int(mask.sum())
        if actual == 0:
            continue
        rows.append(dict(population=label, n=actual, auc_vs_all_neg=ranks[mask].mean() / nn, recovered_10=int((ranks[mask] >= 0.9 * nn).sum()), recovered_20=int(tp[mask].sum()), missed_20=int((mask & ~tp).sum()), wrong_pairs=int((nn - ranks[mask]).sum()), gain_pairs_from_locked=int((ranks[mask] - locked[mask]).sum()), gain_pairs_from_previous=int((ranks[mask] - prev[mask]).sum()), mean_extra_negatives_to_pass_for_goal=need / actual))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / 'other_missed_breaks_audit.csv', index=False)
    assert int(frame.iloc[1:4].wrong_pairs.sum()) == int(frame.iloc[0].wrong_pairs)
    report = dict(auc=float(ranks[y == 1].sum() / total), needed_pairs=int(need), total_pairs=total, interpretation='Oracle arithmetic with all other scores fixed. Missing is defined at 20% false positives. Historical flags only enter this audit.')
    (OUT / 'other_missed_breaks_audit.json').write_text(json.dumps(report, indent=2))
    print(frame.to_string(index=False), flush=True)
if __name__ == '__main__':
    main()
