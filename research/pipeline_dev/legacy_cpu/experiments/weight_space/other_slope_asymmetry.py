# CPU port; original experiments/weight_space/other_slope_asymmetry.py, SHA256 4ebedae05f190720b688f0378003fe7f703b6560e15fb460b483763aa991fca9
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Check whether freeing the lag slope adds more evidence before than after the cut."""
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
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from other_disagreement_signals import matching
from partition_evidence import OUT, ROOT
import white_loop as wl

def main():
    _, ids, _, y, *_ = wl.load()
    f = pd.read_parquet(OUT / 'strict_unit_memory_train.parquet').loc[ids]
    regular = f.grid_pre_only.to_numpy() - f.grid_post_only.to_numpy()
    physical = f.physical_pre_only.to_numpy() - f.physical_post_only.to_numpy()
    table = pd.read_csv(OUT / 'other_disagreement_pairs.csv')
    ix, a, b, meta = matching(any_vote=False)
    assert np.array_equal(ix, ids)
    used = np.r_[ids.get_indexer(table.case_id), ids.get_indexer(table.control_id), a, b]
    assert np.min(used) >= 0 and len(used) == len(np.unique(used))
    train = (y == 0) & ~np.isin(np.arange(len(y)), used)
    lens = pd.read_parquet(ROOT / 'clean/features_train.parquet', columns=['n_pre', 'n_post']).loc[ids]
    pre = pd.read_parquet(ROOT / 'tune_out/generator_segments.parquet', columns=['id', 'part', 'p_bic', 'phi1_ar1'])
    pre = pre[pre.part == 'pre'].set_index('id').loc[ids]
    x = np.column_stack([np.log(lens.n_pre), np.log(lens.n_post), np.log(lens.n_post / lens.n_pre), pre.phi1_ar1, pre.p_bic])

    def residual(v):
        fit = HistGradientBoostingRegressor(learning_rate=0.06, max_iter=75, max_leaf_nodes=12, min_samples_leaf=90, l2_regularization=25, random_state=405, early_stopping=False)
        fit.fit(x[train], v[train])
        return v - fit.predict(x)
    fixed = residual(regular)
    phys = residual(physical)
    confirm = table[table.replication]
    data = {}
    for name, l, r in (('original_confirmation', ids.get_indexer(confirm.case_id), ids.get_indexer(confirm.control_id)), ('disjoint', a, b)):
        rows = []
        for key, values in (('unadjusted', regular), ('adjusted', fixed), ('physical_axis_adjusted', phys)):
            gap = values[l] - values[r]
            rng = np.random.default_rng(4885)
            wins = (gap > 0) + 0.5 * (gap == 0)
            boot = np.concatenate([rng.choice(wins, (128, len(l))).mean(1) for _ in range(96)])
            rows.append(dict(signal=key, matched=len(gap), win=float(wins.mean()), mean=float(gap.mean()), paired_win_interval=np.quantile(boot, [0.025, 0.975]).tolist()))
        data[name] = rows
    data['calibration_sources'] = int(sum(train))
    (OUT / 'other_slope_asymmetry.json').write_text(json.dumps(data, indent=2))
    print(json.dumps(data), flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
