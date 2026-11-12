# CPU port; original experiments/weight_space/readout.py, SHA256 3bce60b6cb8310daf8a805c7e673d0c84dddddbd480adaa99d9694a093c71d84
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Evaluate fixed five-branch heads and a closed-form conditional readout."""
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
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from evaluate import ROOT, OUT, TAGS, inputs, logistic_fit, fit_views, score
from finalize import write_submission
BRANCHES = ['given', 'earlier', 'later', 'before_only', 'after_only']
HEADS = {'location': [1.0, -0.5, -0.5, 0.0, 0.0], 'stability': [0.0, 0.0, 0.0, -0.5, -0.5], 'combined': [1.0, -0.5, -0.5, -0.5, -0.5]}
DEST = ROOT / 'new weights/boundary_readout'

def branches(ids, part):
    folds = []
    oof = np.full((5, 5, len(ids)), np.nan)
    for f in range(5):
        with np.load(OUT / f'boundary_{part}_f{f}.npz') as a:
            idx = ids.get_indexer(a['ids'])
            assert (idx >= 0).all() and list(a['tags']) == TAGS and (list(a['branches']) == BRANCHES)
            if part == 'train':
                assert np.isnan(oof[:, :, idx]).all()
                oof[:, :, idx] = a['logits']
            else:
                assert np.array_equal(idx, np.arange(len(ids)))
                folds.append(a['logits'])
    a = oof if part == 'train' else np.mean(folds, axis=0)
    assert np.isfinite(a).all()
    return logit(np.clip(expit(a.astype(float)), 1e-06, 1 - 1e-06)).mean(axis=1).T

def features(B):
    g, l, r, p, q = np.clip(B, -8.0, 8.0).T
    return np.column_stack([g - (l + r) / 2, l - r, abs(l - r), p, q, abs(p - q), np.maximum(p, q), np.minimum(l, r)])

def means_at(model, z):
    return np.column_stack([np.interp(z, model['knots'], v) for v in np.array(model['means']).T])

def discriminant(B, z, y):
    X = features(B)
    bins = np.array_split(np.argsort(z, kind='stable'), 24)
    model = {'knots': [float(z[b].mean()) for b in bins], 'means': [X[b].mean(0).tolist() for b in bins]}
    X = X - means_at(model, z)
    sd = np.maximum(X.std(0), 0.1)
    X /= sd
    cov = np.zeros((X.shape[1], X.shape[1]))
    diff = np.zeros(X.shape[1])
    mass = 0.0
    for b in bins:
        pos, neg = (X[b[y[b] == 1]], X[b[y[b] == 0]])
        if min(len(pos), len(neg)) < 2:
            continue
        weight = len(pos) * len(neg) / len(b)
        diff += weight * (pos.mean(0) - neg.mean(0))
        cov += weight * (np.cov(pos.T) + np.cov(neg.T)) / 2
        mass += weight
    model['weights'] = (0.8 * np.linalg.solve(cov / mass + 0.15 * np.eye(len(diff)), diff / mass) / sd).tolist()
    return model

def apply(kind, B, z, seq_weight, model=None):
    if kind == 'two_coefficients':
        return z + seq_weight * (B @ np.array(model['head']) + model['bias'])
    if kind == 'conditional':
        return z + (features(B) - means_at(model, z)) @ model['weights']
    return z + seq_weight * (B @ HEADS[kind] - (1.5 if kind in ('stability', 'combined') else 0.0))

def choose_two(B, z, y, wt, seq_weight):
    loc = B @ HEADS['location']
    null = B @ HEADS['stability'] - 1.5
    best = -np.inf
    for a in (0.0, -0.5, 0.5, -1.0, 1.0):
        for b in (0.0, -0.5, 0.5, -1.0, 1.0):
            metrics = score(y, z + seq_weight * (a * loc + b * null), wt)
            utility = (metrics['auc'] + metrics['weighted_auc']) / 2 - 4e-05 * (a * a + b * b)
            if utility > best:
                best = utility
                result = {'a': a, 'b': b, 'head': (a * np.array(HEADS['location']) + b * np.array(HEADS['stability'])).tolist(), 'bias': -1.5 * b}
    return result

def base_logits(path, ids):
    csv = pd.read_csv(path, dtype={'id': str}).set_index('id')
    assert csv.index.is_unique and set(csv.index) == set(ids)
    return logit(csv.loc[ids, 'break_probability'].to_numpy())

def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def replay(bundle, output):
    m = json.loads(bundle.read_text())
    for path, h in m['inputs'].items():
        assert file_hash(ROOT / path) == h, path
    ids = pd.read_parquet(ROOT / 'clean/features_test.parquet', columns=[]).index
    z = base_logits(ROOT / 'new weights/submission.csv', ids)
    B = branches(ids, 'test')
    write_submission(expit(apply(m['kind'], B, z, m['seq_weight'], m['conditional'])), ids, output)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--replay', type=Path)
    parser.add_argument('--output', type=Path)
    a = parser.parse_args()
    if a.replay:
        replay(a.replay, a.output)
        return
    names, ids, tids, y, wt, X, T, V, VT, labels, groups, _ = inputs(True)
    B = branches(ids, 'train')
    idx = labels.index('W_SEQ_qb5')
    assert np.max(np.abs(B[:, 0] - V[:, idx])) < 2e-06
    hist = np.load(ROOT / 'tune_out/weight_edit_20261104/pruning_oof.npz')
    results, saved = ({}, {})
    for split in (7, 29):
        base = hist[f's{split}_unpruned']
        preds = {name: np.empty_like(base) for name in [*HEADS, 'conditional', 'two_coefficients']}
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=split).split(y, y):
            th = fit_views(V[tr], y[tr], groups, logistic_fit(X[tr], y[tr]), 100.0)
            assert np.max(np.abs(V[va] @ th[:-1] + th[-1] - base[va])) < 0.01
            train_z = V[tr] @ th[:-1] + th[-1]
            model = discriminant(B[tr], train_z, y[tr])
            two = choose_two(B[tr], train_z, y[tr], wt[tr], th[idx])
            for name in preds:
                preds[name][va] = apply(name, B[va], base[va], th[idx], two if name == 'two_coefficients' else model)
        preds['incumbent'] = base
        results[str(split)] = {k: score(y, v, wt) for k, v in preds.items()}
        saved.update({f's{split}_{k}': v for k, v in preds.items()})
    eligible = [k for k in preds if all((row[k][m] - row['incumbent'][m] >= 0.0003 for row in results.values() for m in ['auc', 'weighted_auc']))]
    chosen = max(eligible, key=lambda k: sum((row[k]['auc'] for row in results.values()))) if eligible else None
    report = {'scores': results, 'eligible': eligible, 'selected': chosen}
    (OUT / 'boundary_results.json').write_text(json.dumps(report, indent=2))
    np.savez(OUT / 'boundary_oof.npz', ids=ids.to_numpy(dtype=str), **saved)
    print(json.dumps(report), flush=True)
    if a.export and chosen:
        dest = DEST
        dest.mkdir(exist_ok=True)
        incumbent = json.loads((ROOT / 'new weights/ensemble.json').read_text())
        cols = [labels.index(k) for k in incumbent['feature_order']]
        theta = np.asarray(incumbent['coefficients_and_intercept'])
        z = V[:, cols] @ theta[:-1] + theta[-1]
        coef = theta[cols.index(idx)]
        conditional = discriminant(B, z, y) if chosen == 'conditional' else None
        if chosen == 'two_coefficients':
            conditional = choose_two(B, z, y, wt, coef)
        files = [ROOT / 'new weights' / x for x in ('submission.csv', 'model_manifest.json', 'ensemble.json')]
        files.extend((OUT / f'boundary_test_f{f}.npz' for f in range(5)))
        payload = {'kind': chosen, 'seq_weight': coef, 'conditional': conditional, 'scores': results, 'branches': BRANCHES, 'fixed_heads': HEADS, 'inputs': {str(p.relative_to(ROOT)): file_hash(p) for p in files}}
        (dest / 'weights.json').write_text(json.dumps(payload, indent=2))
        baseline = base_logits(files[0], tids)
        Bt = branches(tids, 'test')
        assert np.max(np.abs(Bt[:, 0] - VT[:, idx])) < 2e-05
        expected = expit(VT[:, cols] @ theta[:-1] + theta[-1])
        assert np.max(abs(expected - expit(baseline))) < 1e-12
        write_submission(expit(apply(chosen, Bt, baseline, coef, conditional)), tids, dest / 'submission.csv')
        replay(dest / 'weights.json', dest / 'submission_replay.csv')
        assert file_hash(dest / 'submission.csv') == file_hash(dest / 'submission_replay.csv')
if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
