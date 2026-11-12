# CPU port; original experiments/weight_space/other_miss_witnesses.py, SHA256 bc9561e248e4234ae690ba5d66e811f25aa40228535e8e5e3aacca19fdda10c8
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Count which separate detectors already recognize genuine ensemble misses."""
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
import numpy as np
import pandas as pd
from other_missed_breaks import CURRENT, rankings
from partition_evidence import OUT
from hard_pair_budget import cohorts
from white_joint import read_member
import white_loop as wl
GROUPS = {'crop_trees': ('claude_crop_three_partitions', 'claude_source_loss'), 'physical_trees': ('lawrank_original', 'law_independent_rank_bag_29_7', 'lawvoltree'), 'learned_likelihood': ('lawforecast', 'lawpredictive', 'lawblock', 'lawretrieval'), 'ordered_motion': ('law_ordered_paths', 'law_shock_quantiles'), 'rank_dependence': ('law_rank_dependence_s7', 'law_rank_dependence_s29'), 'physical_heads': ('law_path_control', 'law_shock_quantile_control')}

def make(ids, tids):
    output = []
    sources = []
    for names in GROUPS.values():
        values = []
        for name in names:
            arr, pp = read_member(name, ids, tids)
            sources += pp
            values.append(arr[:, 0])
        output.append(np.column_stack(values))
    return (output, sources)

def main():
    q, ids, tids, y, _, _, white = wl.load()
    hist = cohorts(ids, y)[0]
    with np.load(OUT / (CURRENT + '_oof.npz')) as a:
        assert np.array_equal(a['ids'], ids)
        base = a['s7']
    n = sum(y == 0)
    base_det = rankings(y, base) >= 0.8 * n
    current = (y == 1) & ~base_det
    assert np.sum(current & ~white) == 1857
    banks, _ = make(ids, tids)
    rows = []
    detectors = []
    for name, bank in zip(GROUPS, banks):
        s = bank[:len(y)].mean(1)
        hits = rankings(y, s) >= 0.8 * n
        detectors.append(hits)
        rows.append(dict(name=name, gained_outside_821=int(sum(hits & current & ~white)), gained_other_historical=int(sum(hits & current & hist & ~white)), gained_821=int(sum(hits & current & white)), lost_outside_821=int(sum(~hits & base_det & (y == 1) & ~white)), lost_821=int(sum(~hits & base_det & white)), new_false_alarms=int(sum(hits & ~base_det & (y == 0)))))
    total = np.sum(detectors, axis=0)
    overlap = []
    for k in range(1, len(GROUPS) + 1):
        mask = total >= k
        overlap.append(dict(minimum_agreeing_groups=k, other_actual_misses=int(sum(mask & current & ~white)), other_historical=int(sum(mask & current & hist & ~white)), white_historical=int(sum(mask & current & white)), additional_false_alarms=int(sum(mask & ~base_det & (y == 0)))))
    for tab, k in ((rows, 'individual'), (overlap, 'agreement')):
        frame = pd.DataFrame(tab)
        frame.to_csv(OUT / f'other_miss_witnesses_{k}.csv', index=False)
        print(frame.to_string(index=False), flush=True)
if __name__ == '__main__':
    main()
