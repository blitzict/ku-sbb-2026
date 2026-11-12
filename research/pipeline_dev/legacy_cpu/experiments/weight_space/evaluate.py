# CPU port; original experiments/weight_space/evaluate.py, SHA256 5e207e229f166446c256e9101f2070e8e6da5de3079d229877b63307893b6f16
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Paired evaluation of direct sequence edits with the incumbent stack held fixed."""
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
from pathlib import Path
import numpy as np
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
ROOT = _cpu_workspace()
sys.path.insert(0, str(ROOT / 'experiments/weight_edit_20261104'))
from finalize import inputs, logistic_fit, fit_views, score
OUT = ROOT / 'tune_out/weight_space'
TAGS = ['W_SEQ_q', 'W_SEQ_q_s1', 'W_SEQ_q_s2', 'W_SEQ_qh', 'W_SEQ_qh_s1']
VARIANTS = ('identity', 'smooth', 'common', 'both')

def sequence_logits(ids, part):
    members = []
    for tag in TAGS:
        fold_logits = []
        combined = np.full((4, len(ids)), np.nan)
        assigned = np.zeros(len(ids), bool)
        for fold in range(5):
            with np.load(OUT / f'{tag}_f{fold}_{part}.npz') as archive:
                assert tuple(archive['names']) == VARIANTS
                idx = ids.get_indexer(archive['ids'])
                assert np.all(idx >= 0) and len(np.unique(idx)) == len(idx)
                pred = archive['predictions'].mean(axis=1)
            if part == 'train':
                assert not assigned[idx].any()
                combined[:, idx] = pred
                assigned[idx] = True
            else:
                assert np.array_equal(idx, np.arange(len(ids)))
                fold_logits.append(pred)
        if part == 'test':
            combined = np.mean(fold_logits, axis=0)
        assert np.isfinite(combined).all()
        members.append(logit(np.clip(expit(combined.astype(float)), 1e-06, 1 - 1e-06)))
    return np.mean(members, axis=0)

def main():
    names, ids, _, y, wt, X, _, V, _, labels, groups, _ = inputs(True)
    idx = labels.index('W_SEQ_qb5')
    members = sequence_logits(ids, 'train')
    assert np.max(np.abs(members[0] - V[:, idx])) < 2e-06
    historic = np.load(ROOT / 'tune_out/weight_edit_20261104/pruning_oof.npz')
    assert np.array_equal(y, historic['y'])
    results, saved = ({}, {})
    for split in (7, 29):
        all_logits = np.empty_like(members)
        base_weight = np.zeros(len(y))
        folds = list(StratifiedKFold(5, shuffle=True, random_state=split).split(X, y))
        for tr, va in folds:
            init = logistic_fit(X[tr], y[tr])
            th = fit_views(V[tr], y[tr], groups, init, 100.0)
            base_weight[va] = th[idx]
            all_logits[:, va] = V[va] @ th[:-1] + th[-1] + th[idx] * (members[:, va] - members[0, va])
        reference = historic[f's{split}_unpruned']
        err = float(np.max(np.abs(reference - all_logits[0])))
        assert abs(score(y, all_logits[0], wt)['auc'] - score(y, reference, wt)['auc']) < 2e-06 and err < 0.01, err
        all_logits = reference + base_weight * (members - members[0])
        results[split] = {}
        for name, values in zip(VARIANTS, all_logits):
            s = score(y, values, wt)
            s['fold_auc'] = [score(y[va], values[va], wt[va])['auc'] for _, va in folds]
            results[split][name] = s
            saved[f's{split}_{name}'] = values
        print(split, json.dumps(results[split]), flush=True)
    np.savez(OUT / 'sequence_oof.npz', ids=ids.to_numpy(dtype=str), **saved)
    (OUT / 'sequence_results.json').write_text(json.dumps(results, indent=2))
if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
