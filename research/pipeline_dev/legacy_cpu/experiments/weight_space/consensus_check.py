# CPU port; original experiments/weight_space/consensus_check.py, SHA256 7114d4c32037a1740170f6daf04d5236ddc4575d9f52ec86508a71ce82af3377
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Apply a fixed peer correction only where deployable legacy voters agree."""
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import expert_fusion as ef
from hard_pair_budget import STEM, cohorts, cells

def run():
    base = np.load(ef.OUT / f'{STEM}_oof.npz')
    state = np.load(ef.OUT / 'hard_state_tree_oof.npz')
    ids = pd.Index(base['ids'])
    assert np.array_equal(ids, state['ids'])
    y = pd.read_csv(ef.ROOT / 'data/train_labels.csv', dtype={'id': str}).set_index('id').loc[ids, 'label'].to_numpy()
    miss, alarm = cohorts(ids, y)
    w = np.load(ef.O / 'adv_ptest_train.npy')
    w = w / (1 - w).clip(0.001)
    names = ('xgb_clean', 'lgbsel6', 'xgb_grid', 'v5_xgb', 'v5_lgb', 'nn_cnn3', 'nn_mlp', 'extratrees')
    X = np.column_stack([np.load(ef.O / f'oof_{k}.npy') for k in names])
    assert all(((ef.O / f'test_{k}.npy').is_file() for k in names))
    reports = {}
    for seed in (7, 29, 0):
        low = np.zeros(len(y), bool)
        high = low.copy()
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            med = np.median(X[tr], axis=0)
            low[va] = (X[va] < med).all(1)
            high[va] = (X[va] >= med).all(1)
        anchor = state[f's{seed}_expert']
        reports[seed] = {}
        for name in ('state', 'harmonic_tree', 'claude_tree'):
            z = anchor.copy()
            if name != 'state':
                with np.load(ef.OUT / f'hard_{name}_oof.npz') as peer:
                    assert np.array_equal(ids, peer['ids'])
                    delta = peer[f's{seed}_expert'] - base[f's{seed}_35.0']
                z += 0.5 * (low | high) * delta
            d = cells(y, z, miss, alarm)
            row = ef.score(y, z, w)
            row['missed_vs_negative'] = roc_auc_score(y[miss | (y == 0)], z[miss | (y == 0)])
            row['positive_vs_alarm'] = roc_auc_score(y[alarm | (y == 1)], z[alarm | (y == 1)])
            row['unaffected'] = d['other_pos_other_neg']['auc']
            reports[seed][name] = row
        print(seed, reports[seed], flush=True)
    (ef.OUT / 'hard_consensus_check.json').write_text(json.dumps(reports, indent=2))
if __name__ == '__main__':
    run()
