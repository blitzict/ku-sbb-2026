# CPU port; original experiments/weight_space/paired_regimes.py, SHA256 1ebed3345960d29ec35c69244be8e2424cdd66e896d65ef39025e51db9a5e62f
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Test whether two-sided process descriptors improve the validated fusion head."""
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
PAIRS = ['acf1', 'acf2', 'acf5', 'acf10', 'acf20', 'dacf1', 'skew', 'kurt', 'own4_mad_std', 'own4_absmean', 'own4_skew', 'own4_hill']
MODES = ['splines', 'pooled', 'asymmetric']

def dynamics(ids, tids):
    cols = [s + end for end in ['_pre', '_post'] for s in PAIRS]
    rows = []
    for p, index in (('train', ids), ('test', tids)):
        x = pd.read_parquet(ef.ROOT / f'clean/features_{p}.parquet', columns=cols).loc[index, cols]
        assert x.index.equals(index) and x.index.is_unique
        rows.append(x.to_numpy())
    return np.vstack(rows)

def matrix(mode, E, D, H, z, nc):
    base = ef.design('regime', E, D, z, nc)
    context = D[:, :nc]
    d = np.clip(E - E.mean(axis=1, keepdims=True), -4, 4)
    if mode == 'splines':
        G = np.column_stack([np.maximum(abs(context) - 0.55, 0), np.maximum(context - 0.55, 0)])
    else:
        before, after = np.split(H, 2, axis=1)
        G = 0.5 * (before + after)
        if mode == 'asymmetric':
            G = np.column_stack([G, 0.5 * (after - before)])
    return np.column_stack([base, G, G * np.tanh(z[:, None] / 2), (G[:, :, None] * d[:, None, :]).reshape(len(z), -1)])

def normalize(H, ref):
    return np.column_stack([ef.ranks(x, ref) for x in np.split(H, 2, axis=1)])

def replay(path, output, data=None):
    art = json.loads(path.read_text())
    for f, h in art['sources'].items():
        assert ef.sha(ef.ROOT / f) == h, f
    item, ids, tids, y, wt, X, V, VT, labs, groups, E, D, nc = data or ef.read_data()
    H = dynamics(ids, tids)[len(y):]
    cols = [labs.index(c) for c in item['feature_order']]
    theta = np.array(item['coefficients_and_intercept'])
    z = VT[:, cols] @ theta[:-1] + theta[-1]
    F = matrix(art['mode'], E[len(y):], ef.ranks(D[len(y):], np.array(art['quantiles'])), normalize(H, np.array(art['paired_quantiles'])), z, nc)
    th = np.array(art['weights'])
    ef.write_submission(expit(z + F @ th[:-1] + th[-1]), tids, output)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.replay:
        replay(args.replay, args.output)
        return
    data = ef.read_data()
    item, ids, tids, y, wt, X, V, VT, labs, groups, E, D, nc = data
    H = dynamics(ids, tids)
    old_archive = np.load(ef.OUT / 'expert_fusion_oof.npz')
    assert np.array_equal(ids, old_archive['ids'])
    results, saved = ({}, {})
    for split in ef.SPLITS:
        pred = {mode: np.zeros(len(y)) for mode in MODES}
        control = old_archive[f's{split}_regime']
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=split).split(X, y):
            th = ef.fit_views(V[tr], y[tr], groups, ef.logistic_fit(X[tr], y[tr]), 100.0)
            zr = V[tr] @ th[:-1] + th[-1]
            zv = old_archive[f's{split}_incumbent'][va]
            ref = ef.distribution_maps(D[tr])
            pref = ef.distribution_maps(np.vstack(np.split(H[tr], 2, axis=1)))
            Dr, Dv = (ef.ranks(D[tr], ref), ef.ranks(D[va], ref))
            Hr, Hv = (normalize(H[tr], pref), normalize(H[va], pref))
            for mode in MODES:
                F = matrix(mode, E[tr], Dr, Hr, zr, nc)
                head = ef.fit_head(F, zr, y[tr])
                pred[mode][va] = zv + matrix(mode, E[va], Dv, Hv, zv, nc) @ head[:-1] + head[-1]
        pred['regime'] = control
        results[split] = {m: ef.score(y, p, wt) for m, p in pred.items()}
        saved.update({f's{split}_{m}': p for m, p in pred.items()})
        print(split, results[split], flush=True)
    eligible = [m for m in MODES if all((s[m][k] - s['regime'][k] >= 0.0003 for s in results.values() for k in ('auc', 'weighted_auc')))]
    chosen = max(eligible, key=lambda m: np.mean([s[m]['auc'] for s in results.values()])) if eligible else None
    report = {'scores': results, 'selected': chosen}
    (ef.OUT / 'paired_regime_results.json').write_text(json.dumps(report, indent=2))
    np.savez(ef.OUT / 'paired_regime_oof.npz', ids=ids.to_numpy(dtype=str), **saved)
    if chosen:
        dest = ef.PACKAGE / 'paired_regime_candidate'
        dest.mkdir(exist_ok=True)
        parent = ef.PACKAGE / 'expert_fusion_candidate/weights.json'
        refs = json.loads(parent.read_text())['inputs']
        refs[str(Path(__file__).resolve().relative_to(ef.ROOT))] = ef.sha(Path(__file__))
        for f, h in refs.items():
            assert ef.sha(ef.ROOT / f) == h
        theta = np.array(item['coefficients_and_intercept'])
        cols = [labs.index(c) for c in item['feature_order']]
        z = np.r_[V[:, cols] @ theta[:-1] + theta[-1], VT[:, cols] @ theta[:-1] + theta[-1]]
        q = ef.distribution_maps(D[:len(y)])
        pq = ef.distribution_maps(np.vstack(np.split(H[:len(y)], 2, axis=1)))
        F = matrix(chosen, E, ef.ranks(D, q), normalize(H, pq), z, nc)
        th = ef.fit_head(F[:len(y)], z[:len(y)], y)
        art = {'mode': chosen, 'weights': th.tolist(), 'sources': refs, 'quantiles': q.tolist(), 'paired_quantiles': pq.tolist(), **report}
        path = dest / 'weights.json'
        path.write_text(json.dumps(art, indent=2))
        ef.write_submission(expit(z[len(y):] + F[len(y):] @ th[:-1] + th[-1]), tids, dest / 'submission.csv')
        replay(path, dest / 'submission_replay.csv', data)
        assert ef.sha(dest / 'submission.csv') == ef.sha(dest / 'submission_replay.csv')
    print('Selected', chosen, flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
