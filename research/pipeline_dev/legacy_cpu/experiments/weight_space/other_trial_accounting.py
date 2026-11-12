# CPU port; original experiments/weight_space/other_trial_accounting.py, SHA256 5a51cf9469e5f6768eaeed05612797f8cfc2f5368631946e1912b8c0c03d1f3f
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Count whose actual positive-negative orderings changed under the two focused trials."""
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
from threadpoolctl import threadpool_limits
from other_missed_breaks import CURRENT, rankings
from hard_pair_budget import cohorts
from partition_evidence import OUT
import white_loop as wl

def main():
    _, ids, _, y, _, _, white = wl.load()
    hist = cohorts(ids, y)[0]
    with np.load(OUT / (CURRENT + '_oof.npz')) as a:
        assert np.array_equal(ids, a['ids'])
        base = rankings(y, a['s7'])
    nt = int(sum(y == 0))
    allpos = y == 1
    low = base < 0.8 * nt
    rows = []
    for name in ('quiet_reordering', 'quiet_reordering_serial', 'net_wins_policy', 'adaptive_identity', 'adaptive_noise'):
        with np.load(OUT / ('local_rank_' + name + '_oof.npz')) as a:
            assert np.array_equal(ids, a['ids'])
            rank = rankings(y, a['s7'])
        delta = rank - base
        for label, g in (('all_breaks', allpos), ('white_821', white), ('other_current_misses', allpos & ~white & low), ('other_caught_breaks', allpos & ~white & ~low), ('historical_643', allpos & ~white & hist), ('still_missed_of_643', allpos & ~white & hist & low), ('outside_original_1464_still_missed', allpos & ~hist & low)):
            rows.append(dict(trial=name, positive_group=label, count=int(sum(g)), pairs_won_net=int(delta[g].sum()), moved_up=int(sum(delta[g] > 0)), moved_down=int(sum(delta[g] < 0)), caught_20=int(sum(rank[g] >= 0.8 * nt)), original_caught_20=int(sum(base[g] >= 0.8 * nt))))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / 'other_recovery_pair_accounting.csv', index=False)
    print(frame[frame.trial == 'quiet_reordering_serial'].to_string(index=False), flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
