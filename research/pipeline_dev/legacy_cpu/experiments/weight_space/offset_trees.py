# CPU port; original experiments/weight_space/offset_trees.py, SHA256 19ce291f3169524fc62e28f8ce0fa614a9dace9ddc8815d9d4138d2f97c2430f
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Train shallow score corrections on top of the frozen fold-local parent."""
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
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits
import expert_fusion as ef
from residual_trial import baseline, ANCHOR
ROUNDS = (30, 100)
MODES = ('expert', 'evidence')

def matrix(mode, E, D, z):
    differences = E - E.mean(axis=1, keepdims=True)
    ctx = D[:, :11] if mode == 'expert' else D
    return np.column_stack([z, E, differences, ctx])

def fit(x, z, y, rounds):
    ds = lgb.Dataset(x, label=y, init_score=z, free_raw_data=False)
    pars = dict(objective='binary', metric='None', max_depth=2, num_leaves=4, min_data_in_leaf=300, lambda_l2=80.0, learning_rate=0.035, max_bin=31, num_threads=2, verbosity=-1, deterministic=True, force_col_wise=True, feature_fraction=0.85, seed=3801, boost_from_average=False)
    return lgb.train(pars, ds, num_boost_round=rounds)

def replay(recipe, path):
    art = json.loads(recipe.read_text())
    for n, h in art['hashes'].items():
        assert ef.sha(ef.ROOT / n) == h, n
    _, ids, tids, y, wt, X, V, VT, labs, groups, E, D, nc = ef.read_data()
    p = pd.read_csv(ANCHOR.parent / 'submission.csv', dtype={'id': str}).set_index('id').loc[tids]
    z = logit(p.break_probability.to_numpy())
    x = matrix(art['mode'], E[len(y):], D[len(y):], z)
    model = lgb.Booster(model_file=str(recipe.parent / 'correction.txt'))
    ef.write_submission(expit(z + model.predict(x, raw_score=True, num_threads=2)), tids, path)

def run():
    data = ef.read_data()
    _, ids, tids, y, wt, X, V, VT, labs, groups, E, D, nc = data
    folds, zall = baseline(data)
    scores, archive = ({}, {})
    for split, parts in folds.items():
        names = [f'{mode}_{r}' for mode in MODES for r in ROUNDS]
        preds = {k: np.empty(len(y)) for k in ['anchor', *names]}
        for tr, va, zr, zv in parts:
            preds['anchor'][va] = zv
            for mode in MODES:
                x = matrix(mode, E[tr], D[tr], zr)
                t = matrix(mode, E[va], D[va], zv)
                model = fit(x, zr, y[tr], max(ROUNDS))
                for r in ROUNDS:
                    adj = model.predict(t, num_iteration=r, raw_score=True, num_threads=2)
                    preds[f'{mode}_{r}'][va] = zv + adj
        scores[split] = {k: ef.score(y, p, wt) for k, p in preds.items()}
        archive.update({f's{split}_{k}': p for k, p in preds.items()})
        print(split, scores[split], flush=True)
    eligible = [k for k in names if all((s[k][m] - s['anchor'][m] >= 0.0003 for s in scores.values() for m in ('auc', 'weighted_auc')))]
    chosen = max(eligible, key=lambda k: np.mean([s[k]['auc'] for s in scores.values()])) if eligible else None
    report = {'scores': scores, 'selected': chosen}
    (ef.OUT / 'offset_trees_results.json').write_text(json.dumps(report, indent=2))
    np.savez(ef.OUT / 'offset_trees_oof.npz', ids=ids.to_numpy(dtype=str), **archive)
    if chosen:
        mode, rounds = chosen.split('_')
        dest = ef.PACKAGE / 'offset_trees_candidate'
        dest.mkdir(exist_ok=True)
        x = matrix(mode, E[:len(y)], D[:len(y)], zall[:len(y)])
        model = fit(x, zall[:len(y)], y, int(rounds))
        model.save_model(str(dest / 'correction.txt'))
        files = [dest / 'correction.txt', Path(__file__).resolve(), ANCHOR, ANCHOR.parent / 'submission.csv']
        hashes = {str(p.relative_to(ef.ROOT)): ef.sha(p) for p in files}
        hashes.update(json.loads(ANCHOR.read_text())['inputs'])
        recipe = dest / 'weights.json'
        recipe.write_text(json.dumps({**report, 'mode': mode, 'hashes': hashes}, indent=2))
        replay(recipe, dest / 'submission.csv')
    print('Selected', chosen, flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--replay', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    with threadpool_limits(limits=2):
        if args.replay:
            replay(args.replay, args.output)
        else:
            run()
