# Fixed winning-recipe CPU adaptation v1
"""Evaluate a named pair of ID-indexed feature blocks on the verified fusion model."""
import rebuild_policy as _rebuild
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
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
import expert_fusion as ef
ANCHOR = ef.PACKAGE / 'expert_fusion_candidate/weights.json'
HIST = ef.OUT / 'expert_fusion_oof.npz'
MODES = _rebuild.select(('tails',), ('rank', 'tails', 'conditional'))

def baseline(data):
    item, ids, tids, y, wt, X, V, VT, labs, groups, E, D, nc = data
    archive = np.load(HIST)
    assert np.array_equal(archive['ids'], ids)
    folds = {}
    for split in ef.SPLITS:
        folds[split] = []
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=split).split(X, y):
            theta = ef.fit_views(V[tr], y[tr], groups, ef.logistic_fit(X[tr], y[tr]), 100.0)
            zr = V[tr] @ theta[:-1] + theta[-1]
            q = ef.distribution_maps(D[tr])
            dr = ef.ranks(D[tr], q)
            head = ef.fit_head(ef.design('regime', E[tr], dr, zr, nc), zr, y[tr])
            zv = archive[f's{split}_incumbent'][va]
            check = ef.adjust('regime', E[va], ef.ranks(D[va], q), zv, nc, head)
            assert np.max(abs(check - archive[f's{split}_regime'][va])) < 1e-07
            folds[split].append((tr, va, ef.adjust('regime', E[tr], dr, zr, nc, head), check))
    art = json.loads(ANCHOR.read_text())
    cols = [labs.index(k) for k in item['feature_order']]
    theta = np.array(item['coefficients_and_intercept'])
    zz = np.r_[V[:, cols] @ theta[:-1] + theta[-1], VT[:, cols] @ theta[:-1] + theta[-1]]
    full_z = ef.adjust('regime', E, ef.ranks(D, np.array(art['quantiles'])), zz, nc, np.array(art['weights']))
    abi = pd.read_csv(ANCHOR.parent / 'submission.csv', dtype={'id': str}).set_index('id').loc[tids, 'break_probability']
    assert np.max(abs(expit(full_z[len(y):]) - abi)) < 1e-13
    return (folds, full_z)

def frame(tag, part, ids):
    tab = pd.read_parquet(ef.OUT / f'{tag}_{part}.parquet')
    assert tab.index.is_unique and set(tab.index) == set(ids) and tab.columns.is_unique
    tab = tab.loc[ids]
    assert np.isfinite(tab.to_numpy()).all()
    return tab

def matrix(mode, rank, z, context):
    features = [rank, rank * np.tanh(z[:, None] / 2)]
    if mode in ('tails', 'conditional'):
        features += [np.maximum(rank - 0.5, 0) * 2, np.maximum(-rank - 0.5, 0) * 2]
    if mode == 'conditional':
        features.extend([rank * context[:, j, None] for j in (2, 3, 6)])
    return np.column_stack(features)

def replay(recipe, output):
    art = json.loads(recipe.read_text())
    for f, h in art['inputs'].items():
        assert ef.sha(ef.ROOT / f) == h, f
    data = ef.read_data()
    item, ids, tids, y, wt, X, V, VT, labs, groups, E, D, nc = data
    parent = json.loads(ANCHOR.read_text())
    th = np.array(item['coefficients_and_intercept'])
    parent_z = VT[:, [labs.index(k) for k in item['feature_order']]] @ th[:-1] + th[-1]
    context = ef.ranks(D[len(y):], np.array(parent['quantiles']))
    z = ef.adjust('regime', E[len(y):], context, parent_z, nc, np.array(parent['weights']))
    tab = frame(art['tag'], 'test', tids)
    assert list(tab.columns) == art['feature_order']
    F = matrix(art['mode'], ef.ranks(tab.to_numpy(), np.array(art['quantiles'])), z, context)
    theta = np.array(art['coefficients'])
    ef.write_submission(expit(z + F @ theta[:-1] + theta[-1]), tids, output)

def run(tag):
    assert tag.isidentifier()
    data = ef.read_data()
    _, ids, tids, y, wt, _, _, _, _, _, _, D, _ = data
    zr, zt = (frame(tag, 'train', ids), frame(tag, 'test', tids))
    assert list(zr.columns) == list(zt.columns)
    X, T = (zr.to_numpy(), zt.to_numpy())
    folds, full_z = baseline(data)
    scores, predictions = ({}, {})
    for split, parts in folds.items():
        pred = {k: np.empty(len(y)) for k in ['anchor', *MODES]}
        for tr, va, train_z, val_z in parts:
            pred['anchor'][va] = val_z
            q = ef.distribution_maps(X[tr])
            train, val = (ef.ranks(X[tr], q), ef.ranks(X[va], q))
            qc = ef.distribution_maps(D[tr, :11])
            cr, cv = (ef.ranks(D[tr, :11], qc), ef.ranks(D[va, :11], qc))
            for mode in MODES:
                th = ef.fit_head(matrix(mode, train, train_z, cr), train_z, y[tr])
                pred[mode][va] = val_z + matrix(mode, val, val_z, cv) @ th[:-1] + th[-1]
        scores[split] = {k: ef.score(y, v, wt) for k, v in pred.items()}
        predictions.update({f's{split}_{k}': v for k, v in pred.items()})
        print(split, scores[split], flush=True)
    eligible = [m for m in MODES if all((s[m][k] - s['anchor'][k] >= 0.0003 for s in scores.values() for k in ('auc', 'weighted_auc')))]
    chosen = _rebuild.select('tails', max(eligible, key=lambda m: np.mean([s[m]['auc'] for s in scores.values()])) if eligible else None)
    report = _rebuild.annotate({'scores': scores, 'selected': chosen})
    (ef.OUT / f'{tag}_results.json').write_text(json.dumps(report, indent=2))
    np.savez(ef.OUT / f'{tag}_oof.npz', ids=ids.to_numpy(dtype=str), **predictions)
    if chosen:
        dest = ef.PACKAGE / f'{tag}_candidate'
        dest.mkdir(exist_ok=True)
        q = ef.distribution_maps(X)
        context = ef.ranks(D, np.array(json.loads(ANCHOR.read_text())['quantiles']))
        f = matrix(chosen, ef.ranks(X, q), full_z[:len(y)], context[:len(y)])
        th = ef.fit_head(f, full_z[:len(y)], y)
        files = [ANCHOR, ef.OUT / f'{tag}_test.parquet', ef.OUT / f'{tag}_train.parquet', Path(__file__).resolve()]
        hashes = {str(p.relative_to(ef.ROOT)): ef.sha(p) for p in files}
        hashes.update(json.loads(ANCHOR.read_text())['inputs'])
        art = {'tag': tag, 'mode': chosen, 'feature_order': list(zt.columns), 'quantiles': q.tolist(), 'coefficients': th.tolist(), 'inputs': hashes, **report}
        path = dest / 'weights.json'
        path.write_text(json.dumps(art, indent=2))
        test = matrix(chosen, ef.ranks(T, q), full_z[len(y):], context[len(y):])
        ef.write_submission(expit(full_z[len(y):] + test @ th[:-1] + th[-1]), tids, dest / 'submission.csv')
        replay(path, dest / 'submission_replay.csv')
        assert ef.sha(dest / 'submission.csv') == ef.sha(dest / 'submission_replay.csv')
    print('Selected', chosen, flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--tag')
    p.add_argument('--replay', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    with threadpool_limits(limits=2):
        if args.replay:
            replay(args.replay, args.output)
        else:
            run(args.tag)
