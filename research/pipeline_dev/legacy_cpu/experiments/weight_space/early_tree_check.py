# CPU port; original experiments/weight_space/early_tree_check.py, SHA256 59affb5a4a7ca0b61fe51d67f4f69ed75fea8c2f514f8589a6a054f0493585e9
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Refit the frozen stack with Claude's harmonic tree present at the first layer."""
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
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
import expert_fusion as ef
import extend_readout as er
import innovation_head as ih
import mechanism_edit as me
import seasonal_head as sh
import trend_870 as trend
import hard_expert_trial as expert
from hard_pair_budget import cohorts, cells
from hard_case_trial import PARENT
from physical_870 import fit

def harmonic_views(ids, tids):
    logs = lambda x: logit(np.clip(x.astype(float), 1e-06, 1 - 1e-06))
    views = []
    tests = []
    inputs = []
    for seed in (0, 1):
        path = ef.O / f'ver_VARC4harm_ns3_s{seed}.npz'
        inputs.append(path)
        with np.load(path, allow_pickle=False) as q:
            kinds = q['kinds']
            assert list(kinds) == ['orig', 'rev', 'flip', 'revflip', 'c1', 'c2', 'c3', 'c4'] or set(kinds) == {'orig', 'rev', 'flip', 'revflip', 'c1', 'c2', 'c3', 'c4'}
            if views:
                assert np.array_equal(kinds, old_kinds)
            old_kinds = kinds.copy()
            views.append(logs(q['oof']))
            tests.append(logs(q['test']))
    weights = np.array([{'orig': 2.0, 'flip': 2.0, 'rev': 1.0, 'revflip': 1.0}.get(str(k), 0.125) for k in kinds])
    weights /= weights.sum()
    V, T = (np.mean(views, 0), np.mean(tests, 0))
    assert V.shape == (len(weights), len(ids)) and T.shape == (len(weights), len(tids))
    for mat, p in ((V, 'oof'), (T, 'test')):
        path = ef.O / f'{p}_VARC4harm_ns3_wc.npy'
        inputs.append(path)
        combined = logs(np.load(path))
        mat += combined - weights @ mat
    return (np.vstack([V.T, T.T]), weights, inputs)

def run(verify):
    data = ef.read_data()
    _, ids, tids, y, wt, X, V, VT, _, groups, E, D, nc = data
    orig = np.vstack([V, VT])
    new, frac, inputs = harmonic_views(ids, tids)
    expanded = np.column_stack([orig, new])
    Vnew = np.column_stack([X, new[:len(y)] @ frac])
    more_groups = groups + [(orig.shape[1], expanded.shape[1], frac)]
    season = json.loads(ih.PARENT.read_text())
    _, features, g, H = sh.read_tables(data, True)
    A = features[season['cols']].to_numpy()
    mtab = me.tables(data, me.inputs(trend.BASE_MODE))
    ttab = me.tables(data, trend.names('net'))
    locked = json.loads(PARENT.read_text())
    mods = er.modules(locked['spec'])
    _, ctab = er.tables(data, mods)
    st = me.tables(data, ['state'])['state']
    train, q = expert.sources('state_tree', 'train', ids)
    test, r = expert.sources('state_tree', 'test', tids)
    tree = [np.r_[a, b] for a, b in zip(train, test)]
    inputs.extend(q + r)

    def fold(tr, va, add):
        M, raw, grp = (expanded, Vnew, more_groups) if add else (orig, X, groups)
        w = ef.fit_views(M[tr], y[tr], grp, ef.logistic_fit(raw[tr], y[tr]), 100.0)
        zr, zv = (M[i] @ w[:-1] + w[-1] for i in (tr, va))
        ranks = ef.distribution_maps(D[tr])
        F, Q = (ef.design('regime', E[i], ef.ranks(D[i], ranks), z, nc) for i, z in ((tr, zr), (va, zv)))
        w = ef.fit_head(F, zr, y[tr])
        zr, zv = (zr + F @ w[:-1] + w[-1], zv + Q @ w[:-1] + w[-1])
        hr, hv = sh.harmonic_parent(H, tr, va, zr, zv, y)
        mode = season['mode']
        zr, zv = (sh.base(mode, zr, hr, g[tr]), sh.base(mode, zv, hv, g[va]))
        ranks = sh.lookup(A, tr, g)
        F, Q = (sh.features(mode, A[i], g[i], z, ranks) for i, z in ((tr, zr), (va, zv)))
        w = sh.fit(F, zr, y[tr])
        zr, zv = (zr + F @ w, zv + Q @ w)
        F, Q = (me.encode(trend.BASE_MODE, mtab, z, g, i) for i, z in ((tr, zr), (va, zv)))
        w = sh.fit(F, zr, y[tr], penalty=40.0)
        zr, zv = (zr + F @ w, zv + Q @ w)
        F, Q = (trend.design('net', ttab, z, i) for i, z in ((tr, zr), (va, zv)))
        w = fit(F, zr, y[tr], 60.0)
        zr, zv = (zr + F @ w, zv + Q @ w)
        F, Q = (er.features(mods, ctab, z, i) for i, z in ((tr, zr), (va, zv)))
        w = fit(F, zr, y[tr], locked['selected'])
        zr, zv = (zr + F @ w, zv + Q @ w)
        F, Q = (expert.features(tree, z, i, st) for i, z in ((tr, zr), (va, zv)))
        w = fit(F, zr, y[tr], 100.0)
        return (zv + Q @ w, zv, Q, w)
    src = ef.OUT / ('hard_state_tree_verified_oof.npz' if verify else 'hard_state_tree_oof.npz')
    miss, alarm = cohorts(ids, y)
    hist = np.load(src)
    assert np.array_equal(ids, hist['ids'])
    scores = {}
    archive = {}
    for seed in (7, 29, 0, 123, 2024, 11041) if verify else (7, 29, 0):
        p = np.zeros(len(y))
        ref = hist[f's{seed}_expert']
        for j, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)):
            if j == 0 and seed == 7:
                check = fold(tr, va, False)[0]
                assert max(abs(check - ref[va])) < 1e-07
            p[va] = fold(tr, va, True)[0]
        scores[seed] = {}
        for tag, z in (('parent', ref), ('early', p)):
            s = ef.score(y, z, wt)
            c = cells(y, z, miss, alarm)
            s.update(missed_vs_negative=1 - sum((v['wrong'] for k, v in c.items() if k.startswith('missed_'))) / (sum(miss) * sum(y == 0)), missed_above_median=int((z[miss] > np.median(z)).sum()))
            scores[seed][tag] = s
        archive[f's{seed}'] = p
        print(seed, scores[seed], flush=True)
    passes = all((s['early']['auc'] - s['parent']['auc'] >= 0.0002 and all((s['early'][k] > s['parent'][k] for k in ('weighted_auc', 'missed_vs_negative'))) for s in scores.values()))
    name = 'hard_early_tree' + ('_verified' if verify else '')
    report = dict(passes=passes, scores=scores)
    (ef.OUT / f'{name}.json').write_text(json.dumps(report, indent=2))
    np.savez_compressed(ef.OUT / f'{name}_oof.npz', ids=ids.to_numpy(dtype=str), **archive)
    if passes and verify:
        _, z, F, w = fold(np.arange(len(y)), np.arange(len(y), len(y) + len(tids)), True)
        folder = ef.PACKAGE / 'hard_early_tree_candidate'
        folder.mkdir(exist_ok=True)
        sub = ef.write_submission(expit(z + F @ w), tids, folder / 'submission.csv')
        np.savez_compressed(folder / 'replay_inputs.npz', parent_logits=z, features=F, ids=tids.to_numpy(dtype=str), order=pd.Index(tids).get_indexer(sub.id))
        inputs.extend([PARENT, Path(__file__).resolve()])
        (folder / 'weights.json').write_text(json.dumps({**report, 'weights': w.tolist(), 'hashes': {str(p.relative_to(ef.ROOT)): ef.sha(p) for p in inputs}}, indent=2))
    print('passes', passes, flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--verify', action='store_true')
    with threadpool_limits(limits=2):
        run(p.parse_args().verify)
