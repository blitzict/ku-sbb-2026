# CPU port; original experiments/weight_space/white_false_alarm_groups.py, SHA256 2adc36ffb073e84c8ffaf0756878befad1aca1b5e570aee3d7a9afdb4a2e8ea6
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Locate no-break controls responsible for lost AUC pairs in the current ensemble."""
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
from partition_evidence import ROOT, OUT
from white_freeze_first_layer import NAME
import white_loop as wl

def main():
    q, ids, _, y, _, wn, miss = wl.load()

    def table(name, cols):
        return pd.read_parquet(OUT / f'{name}_train.parquet', columns=cols).loc[ids]
    g = pd.read_parquet(ROOT / 'tune_out/generator_segments.parquet', columns=['id', 'part', 'p_bic', 'phi1_ar1'])
    g = g[g.part == 'pre'].set_index('id').loc[ids]
    wc = table('periodic', ['gate']).gate.to_numpy()
    trend = table('curve', ['gate']).gate.to_numpy()
    vol = table('volatility_law', ['pre_cluster', 'post_cluster'])
    mode = table('mode_residual', ['gate']).gate.to_numpy()
    fam = np.select([wc > 0.65, g.phi1_ar1 > 0.95, trend > 0.65, vol.pre_cluster > 0.65, mode > 0.8, wn, g.phi1_ar1 < -0.18, g.phi1_ar1 > 0.18], ['periodic', 'random_walk', 'trend', 'volatility', 'two_modes', 'white', 'negative_correlation', 'positive_correlation'], default='other')
    with np.load(OUT / (NAME + '_oof.npz')) as a:
        assert np.array_equal(a['ids'], ids)
        z = a['s7']
    neg = y == 0
    low = np.searchsorted(np.sort(z[neg]), z) / neg.sum()
    hard = np.searchsorted(np.sort(z[miss]), z)
    allpos = np.searchsorted(np.sort(z[y == 1]), z)
    locked = np.empty(len(y))
    for f in range(5):
        k = f's7_f{f}_'
        locked[q[k + 'va']] = q[k + 'zv']
    historic = ids.isin(pd.read_csv(ROOT / 'graphs/consistent_false_alarms.csv', dtype={'id': str}).id)
    groups = {'observed_family': fam, 'past_alarm': np.where(historic, 'old_false_alarm', 'other_negatives'), 'negative_score_band': np.where(low < 0.5, 'bottom_half', np.where(low < 0.8, '50-80', np.where(low < 0.95, '80-95', 'top_5')))}
    result = []
    for partition, category in groups.items():
        for key in np.unique(category):
            sel = neg & (category == key)
            if not np.any(sel):
                continue
            result.append(dict(partition=partition, group=key, n=int(sel.sum()), hard_pair_loss=int(hard[sel].sum()), all_pair_loss=int(allpos[sel].sum()), hard_loss_per_negative=round(hard[sel].mean(), 1), overall_loss_per_negative=round(allpos[sel].mean(), 1), top_5pct=int(np.sum(sel & (low >= 0.95))), median_logit=float(np.median(z[sel])), change_from_locked=float(np.median(z[sel] - locked[sel]))))
    result = pd.DataFrame(result)
    result.to_csv(OUT / 'false_alarm_pair_groups.csv', index=False)
    print(result[['partition', 'group', 'n', 'hard_loss_per_negative', 'overall_loss_per_negative', 'top_5pct']].to_string(index=False))
if __name__ == '__main__':
    main()
