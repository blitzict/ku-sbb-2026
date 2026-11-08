# CPU port; original experiments/weight_space/hard_pair_budget.py, SHA256 b947da23ef4417932cea9a8fe02fa0e6c498a88d171d9fef9007399a78115d03
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Attribute misordered positive-negative pairs to disjoint historical cohorts."""
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
import rebuild_policy as _rebuild
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from partition_evidence import ROOT, OUT
STEM = 'extended_curve_net_bank_both_pulse_joint_head_family_state_net'

def cohorts(ids, y):
    if _rebuild.fixed():
        # Historical id lists were only diagnostic and promotion guardrails.
        # Recompute the diagnostic on fresh OOF scores instead of importing
        # training-row lists or rebuilding ten discarded diagnostic models.
        path = ROOT / 'tune_out/weight_edit_20261104/pruning_oof.npz'
        with np.load(path) as a:
            assert np.array_equal(a['y'], y)
            score = a['s7_unpruned']
        assert score.shape == y.shape and np.isfinite(score).all()
        low = rankdata(score, method='average') / len(y) < 0.5
        miss, alarm = (y == 1) & low, (y == 0) & ~low
        assert 0 < miss.sum() < (y == 1).sum() and 0 < alarm.sum() < (y == 0).sum()
        return miss, alarm
    miss = ids.isin(pd.read_csv(ROOT / 'graphs/consistently_missed_breaks.csv', dtype={'id': str}).id)
    alarm = ids.isin(pd.read_csv(ROOT / 'graphs/consistent_false_alarms.csv', dtype={'id': str}).id)
    assert miss.sum() == 1464 and alarm.sum() == 928 and (y[miss] == 1).all() and (y[alarm] == 0).all()
    return (miss, alarm)

def cells(y, z, miss, alarm):
    report = {}
    for a, positive in [('missed', miss), ('other_pos', (y == 1) & ~miss)]:
        for b, negative in [('alarms', alarm), ('other_neg', (y == 0) & ~alarm)]:
            x = z[positive]
            order = np.sort(z[negative])
            wins = (np.searchsorted(order, x, 'left') + np.searchsorted(order, x, 'right')).sum() / 2
            size = len(x) * len(order)
            report[a + '_' + b] = dict(pairs=size, wrong=size - wins, auc=wins / size)
    return report

def matched_controls(ids, y, miss, z, new):
    names = ('xgb', 'lgb', 'xgb_clean', 'lgbsel6', 'xgb_grid', 'v5_xgb', 'v5_lgb', 'nn_cnn3', 'nn_mlp', 'extratrees')
    old = np.column_stack([rankdata(np.load(ROOT / f'tune_out/oof_{k}.npy')) / len(y) for k in names])
    low = (old < 0.5).all(1)
    assert np.array_equal(miss, low & (y == 1))
    df = pd.read_parquet(ROOT / 'tune_out/generator_segments.parquet', columns=['id', 'part', 'phi1_ar1', 'p_bic'])
    df = df.loc[df.part == 'pre'].set_index('id').loc[ids]
    v, p = (df.phi1_ar1, df.p_bic)
    family = np.select([v > 0.95, p >= 5, p == 0, v < -0.2, v > 0.2], ['RW', 'OSC', 'WN', 'negAR', 'posAR'], 'lowAR')
    bands = np.floor(old.mean(1) / 0.025).astype(int)
    results = {}
    all_rows = []
    for fam in np.unique(family):
        rows = []
        for band in np.unique(bands[low]):
            group = low & (family == fam) & (bands == band)
            negatives = group & (y == 0)
            if negatives.sum() < 5:
                continue
            for j in np.flatnonzero(group & miss):
                rows.append([np.mean((s[j] > s[negatives]) + 0.5 * (s[j] == s[negatives])) for s in (z, new)])
        if rows:
            all_rows.extend(rows)
            results[fam] = {'n': len(rows), 'parent': np.mean(rows, 0)[0], 'state_tree': np.mean(rows, 0)[1]}
    results['all'] = {'n': len(all_rows), 'parent': np.mean(all_rows, 0)[0], 'state_tree': np.mean(all_rows, 0)[1]}
    return results

def run():
    archive = np.load(OUT / f'{STEM}_oof.npz')
    ids = pd.Index(archive['ids'])
    y = pd.read_csv(ROOT / 'data/train_labels.csv', dtype={'id': str}).set_index('id').loc[ids, 'label'].to_numpy()
    miss, alarm = cohorts(ids, y)
    z = archive['s7_35.0']
    pct = (rankdata(z) - 0.5) / len(z)
    npos = (y == 1).sum()
    nneg = (y == 0).sum()
    total = int(npos * nneg)
    scores = cells(y, z, miss, alarm)
    wrong = sum((item['wrong'] for item in scores.values()))
    auc = 1 - wrong / total
    assert abs(roc_auc_score(y, z) - auc) < 1e-12
    gain = (0.9 - auc) * total
    missed_wrong = sum((c['wrong'] for k, c in scores.items() if k.startswith('missed_')))
    alarms_wrong = sum((c['wrong'] for k, c in scores.items() if k.endswith('_alarms')))
    union_wrong = wrong - scores['other_pos_other_neg']['wrong']
    current_miss = 1 - missed_wrong / (miss.sum() * nneg)
    report = dict(primary_auc=auc, total_pairs=total, pairs_to_target=round(gain), cells=scores, currently_above_median=int((pct[miss] > 0.5).sum()), alarm_count_above_median=int((pct[alarm] > 0.5).sum()), median_missed_rank=float(np.median(pct[miss])), median_alarm_rank=float(np.median(pct[alarm])), missed_vs_all_neg=current_miss, needed_missed_vs_all_neg=current_miss + gain / (miss.sum() * nneg), current_pos_vs_alarms=1 - alarms_wrong / (alarm.sum() * npos), needed_pos_vs_alarms=1 - (alarms_wrong - gain) / (alarm.sum() * npos), headroom=dict(missed=missed_wrong / total, alarms=alarms_wrong / total, either=union_wrong / total), fraction_of_group_errors_needed=dict(missed=gain / missed_wrong, alarms=gain / alarms_wrong, either=gain / union_wrong))
    candidate = OUT / 'hard_state_tree_oof.npz'
    if candidate.exists():
        with np.load(candidate) as ar:
            assert np.array_equal(ids, ar['ids'])
            report['matched_controls'] = matched_controls(ids, y, miss, z, ar['s7_expert'])
    (OUT / 'hard_pair_budget.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
if __name__ == '__main__':
    run()
