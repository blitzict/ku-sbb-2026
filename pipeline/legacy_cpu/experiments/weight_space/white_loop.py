# Fixed winning-recipe CPU adaptation v1
"""Evaluation harness for the white-noise recovery loop on the locked state-tree parent."""
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
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import expert_fusion as ef
import extend_readout as er
import hard_expert_trial as he
from hard_pair_budget import cohorts
from physical_870 import fit
PARENT = ef.PACKAGE / 'hard_state_tree_candidate/weights.json'
REF = ef.OUT / 'hard_state_tree_verified_oof.npz'
CACHE = ef.OUT / 'white_loop_parent.npz'

def prepare():
    data = ef.read_data()
    ids, tids, y = data[1:4]
    X, _ = he.sources('state_tree', 'train', ids)
    s = pd.read_parquet(ef.OUT / 'state_train.parquet').loc[ids]
    frozen = json.loads(he.PARENT.read_text())
    mods = er.modules(frozen['spec'])
    _, tables = er.tables(data, mods)
    folds, z = er.parent(data, (123, 2024, 11041))
    ref = np.load(REF)
    assert np.array_equal(ref['ids'], ids)
    out = {'ids': ids.to_numpy(dtype=str), 'tids': tids.to_numpy(dtype=str), 'parent_hash': ef.sha(PARENT)}
    for seed, parts in folds.items():
        for n, (tr, va, zr, zv) in enumerate(parts):
            a, b = (er.features(mods, tables, zr, tr), er.features(mods, tables, zv, va))
            w = fit(a, zr, y[tr], frozen['selected'])
            zr, zv = (zr + a @ w, zv + b @ w)
            a, b = (he.features(X, zr, tr, s), he.features(X, zv, va, s))
            w = fit(a, zr, y[tr], 100.0)
            zr, zv = (zr + a @ w, zv + b @ w)
            assert np.max(abs(zv - ref[f's{seed}_expert'][va])) < 1e-07
            key = f's{seed}_f{n}_'
            out.update({key + k: v for k, v in zip(('tr', 'va', 'zr', 'zv'), (tr, va, zr, zv))})
    X = [np.r_[u, v] for u, v in zip(X, he.sources('state_tree', 'test', tids)[0])]
    st = pd.concat([s, pd.read_parquet(ef.OUT / 'state_test.parquet').loc[tids]])
    ix = np.arange(len(z))
    z = z + er.features(mods, tables, z, ix) @ np.array(frozen['weights'])
    z += he.features(X, z, ix, st) @ np.array(json.loads(PARENT.read_text())['weights'])
    p = pd.read_csv(PARENT.parent / 'submission.csv', dtype={'id': str}).set_index('id').loc[tids, 'break_probability']
    assert np.max(abs(expit(z[len(y):]) - p.to_numpy())) < 1e-12
    out['zr_all'] = z[:len(y)]
    out['zt'] = logit(p.to_numpy())
    np.savez_compressed(CACHE, **out)
    print('verified frozen parent on six splits', flush=True)

def load():
    q = np.load(CACHE)
    ids, tids = (pd.Index(q['ids']), pd.Index(q['tids']))
    assert str(q['parent_hash']) == ef.sha(PARENT)
    y = pd.read_csv(ef.ROOT / 'data/train_labels.csv', dtype={'id': str}).set_index('id').loc[ids, 'label'].to_numpy()
    df = pd.read_parquet(ef.O / 'generator_segments.parquet', columns=['id', 'part', 'p_bic', 'phi1_ar1'])
    d = df[df.part == 'pre'].set_index('id').loc[ids]
    wn = (d.p_bic.to_numpy() == 0) & (d.phi1_ar1.to_numpy() <= 0.95)
    miss = cohorts(ids, y)[0] & wn
    if not _rebuild.fixed():
        assert miss.sum() == 821
    wt = _rebuild.diagnostics(ef.O, len(y))
    return (q, ids, tids, y, wt, wn, miss)

def score(y, z, wt, wn, miss):

    def auc(mask):
        return roc_auc_score(y[mask], z[mask])
    return {'auc': roc_auc_score(y, z), 'weighted': roc_auc_score(y, z, sample_weight=wt), 'remaining': auc(~miss), 'target_vs_neg': auc(miss | (y == 0)), 'target_vs_wn_neg': auc(miss | (y == 0) & wn), 'wn': auc(wn), 'target_above_median': int((z[miss] > np.median(z)).sum())}

def run(name, design, penalty=80.0, seeds=(7, 29, 0), dependencies=(), save=True):
    q, ids, tids, y, wt, wn, miss = load()
    scores = {}
    oof = {}
    for seed in seeds:
        base = np.empty(len(y))
        new = base.copy()
        for n in range(5):
            k = f's{seed}_f{n}_'
            tr, va, zr, zv = [q[k + v] for v in ('tr', 'va', 'zr', 'zv')]
            f = design(tr, zr)
            v = design(va, zv)
            w = fit(f, zr, y[tr], penalty)
            new[va] = zv + v @ w
            base[va] = zv
        scores[seed] = {k: score(y, s, wt, wn, miss) for k, s in (('base', base), ('new', new))}
        oof[f's{seed}'] = new
        print(seed, {k: round(scores[seed]['new'][k], 7) for k in ('auc', 'remaining', 'target_vs_neg', 'weighted')}, flush=True)
    accepted = all((s['new']['auc'] > s['base']['auc'] and s['new']['remaining'] >= s['base']['remaining'] and (s['new']['target_vs_neg'] > s['base']['target_vs_neg']) for s in scores.values()))
    report = _rebuild.annotate(dict(name=name, penalty=penalty, accepted=accepted, scores=scores))
    (ef.OUT / f'wn_{name}.json').write_text(json.dumps(report, indent=2))
    np.savez_compressed(ef.OUT / f'wn_{name}_oof.npz', ids=ids.to_numpy(dtype=str), **oof)
    if _rebuild.export(accepted) and save:
        tr = np.arange(len(ids))
        test = len(ids) + np.arange(len(tids))
        w = fit(design(tr, q['zr_all']), q['zr_all'], y, penalty)
        f = design(test, q['zt'])
        d = ef.PACKAGE / f'wn_{name}_candidate'
        d.mkdir(exist_ok=True)
        sub = ef.write_submission(expit(q['zt'] + f @ w), tids, d / 'submission.csv')
        recipe = d / 'weights.json'
        paths = [Path(__file__).resolve(), PARENT, PARENT.parent / 'submission.csv', *dependencies]
        recipe.write_text(json.dumps({**report, 'weights': w.tolist(), 'hashes': {str(p.relative_to(ef.ROOT)): ef.sha(p) for p in paths}}, indent=2))
        np.savez_compressed(d / 'replay_inputs.npz', parent_logits=q['zt'], features=f, ids=tids.to_numpy(dtype=str), order=tids.get_indexer(sub.id))
    print('accepted', accepted, flush=True)
    return report
if __name__ == '__main__':
    with threadpool_limits(limits=2):
        prepare()
