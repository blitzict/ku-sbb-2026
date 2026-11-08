# Fixed winning-recipe CPU adaptation v1
"""Compare matched tree experts against the locked ensemble without using hard IDs to route."""
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
from threadpoolctl import threadpool_limits
import expert_fusion as ef
import extend_readout as er
import mechanism_edit as me
from hard_case_trial import PARENT, diagnostic
from hard_pair_budget import STEM, cohorts
from physical_870 import fit

def sources(kind, part, ids):
    if kind == 'state_tree':
        files = [ef.OUT / f'shape_bank_{k}_{part}.parquet' for k in ('adjusted', 'state')]
        frames = [pd.read_parquet(f).loc[ids] for f in files]
        return ([f.score.to_numpy() for f in frames], files)
    prefix = 'oof' if part == 'train' else 'test'
    if kind == 'harmonic_tree':
        files = [ef.O / f'{prefix}_{k}.npy' for k in ('VARC4hv_ns3_wc', 'VARC4harm_ns3_wc')]
    else:
        files = [ef.O / 'redesign' / f'{prefix}_mech_{k}.npy' for k in ('ns3_harm', 'ns3_allmech')]
    x = [np.load(f) for f in files]
    assert all((s.shape == (len(ids),) and np.isfinite(s).all() and (s >= 0).all() and (s <= 1).all() for s in x))
    if part == 'train':
        with np.load(ef.O / 'var/mat_ns3_plain.npz', allow_pickle=True) as q:
            assert np.array_equal(ids, q['ids'])
    return ([logit(np.clip(s.astype(float), 1e-06, 1 - 1e-06)) for s in x], files)

def features(X, z, ix, state):
    old, new = (q[ix] for q in X)
    gap = (new - old).clip(-8, 8)
    d = (new - z).clip(-8, 8)
    h = state.iloc[ix]
    ar = np.clip(np.maximum(abs(h.pre_ar1), abs(h.pre_ar2)) / 0.5, 0, 1)
    loc = np.tanh(z / 2)
    return np.column_stack([d, gap, d * loc, gap * loc, ar * d, ar * gap, h.pre_cluster * d, h.walk * d])

def replay(path, output):
    a = json.loads(path.read_text())
    for name, h in a['hashes'].items():
        assert ef.sha(ef.ROOT / name) == h, name
    tids = pd.read_parquet(ef.ROOT / 'clean/features_test.parquet', columns=[]).index
    X, _ = sources(a['kind'], 'test', tids)
    st = pd.read_parquet(ef.OUT / 'state_test.parquet').loc[tids]
    p = pd.read_csv(PARENT.parent / 'submission.csv', dtype={'id': str}).set_index('id').loc[tids, 'break_probability']
    z = logit(p.to_numpy())
    f = features(X, z, np.arange(len(tids)), st)
    sub = ef.write_submission(expit(z + f @ np.array(a['weights'])), tids, output)
    if _rebuild.fixed() and output.name == 'submission.csv':
        np.savez_compressed(path.parent / 'replay_inputs.npz', ids=tids.to_numpy(dtype=str),
                            order=tids.get_indexer(sub.id), parent_logits=z, features=f)
        from pipeline_helpers.replay_saved_head import package
        package(path.parent)

def run(kind, verify):
    data = ef.read_data()
    ids, tids, y, wt = data[1:5]
    X, files = sources(kind, 'train', ids)
    t = me.tables(data, ['state'])['state']
    miss, alarm = cohorts(ids, y)
    d = pd.read_parquet(ef.O / 'generator_segments.parquet', columns=['part', 'id', 'phi1_ar1', 'p_bic'])
    d = d.loc[d.part == 'pre'].set_index('id').loc[ids]
    phi, pb = (d.phi1_ar1, d.p_bic)
    fam = np.select([phi > 0.95, pb >= 5, pb == 0, phi < -0.2, phi > 0.2], ['RW', 'OSC', 'WN', 'negAR', 'posAR'], 'lowAR')
    folds, full = er.parent(data, (123, 2024, 11041) if verify else ())
    frozen = json.loads(PARENT.read_text())
    mods = er.modules(frozen['spec'])
    _, tables = er.tables(data, mods)
    old = np.load(ef.OUT / f'{STEM}_oof.npz')
    assert np.array_equal(ids, old['ids'])
    scores = {}
    archive = {}
    for seed, parts in folds.items():
        pred = np.full(len(y), np.nan)
        base = pred.copy()
        for tr, va, zr, zv in parts:
            fr, fv = (er.features(mods, tables, zr, tr), er.features(mods, tables, zv, va))
            co = fit(fr, zr, y[tr], frozen['selected'])
            zr, zv = (zr + fr @ co, zv + fv @ co)
            if seed in (0, 7, 29):
                assert np.max(abs(zv - old[f's{seed}_35.0'][va])) < 1e-07
            base[va] = zv
            v = fit(features(X, zr, tr, t), zr, y[tr], 100.0)
            pred[va] = zv + features(X, zv, va, t) @ v
        scores[seed] = {a: diagnostic(v, y, miss, alarm, fam, wt) for a, v in (('parent', base), ('expert', pred))}
        archive[f's{seed}_expert'] = pred
        print(seed, [(k, round(c['auc'], 7), round(c['missed_vs_negative'], 6)) for k, c in scores[seed].items()], flush=True)
    eligible = all((s['expert']['auc'] - s['parent']['auc'] >= 0.0002 and all((s['expert'][k] > s['parent'][k] for k in ('weighted_auc', 'missed_vs_negative'))) for s in scores.values()))
    tag = f'hard_{kind}' + ('_verified' if verify else '')
    report = _rebuild.annotate(dict(kind=kind, passes=eligible, scores=scores))
    (ef.OUT / f'{tag}.json').write_text(json.dumps(report, indent=2))
    np.savez_compressed(ef.OUT / f'{tag}_oof.npz', ids=ids.to_numpy(dtype=str), **archive)
    if _rebuild.export(eligible) and verify:
        _, testfiles = sources(kind, 'test', tids)
        zr = full + er.features(mods, tables, full, np.arange(len(full))) @ np.array(frozen['weights'])
        ix = np.arange(len(ids))
        w = fit(features(X, zr[ix], ix, t), zr[ix], y, 100.0)
        folder = ef.PACKAGE / f'hard_{kind}_candidate'
        folder.mkdir(exist_ok=True)
        files.extend([*testfiles, PARENT, PARENT.parent / 'submission.csv', Path(__file__).resolve(), ef.OUT / 'state_test.parquet'])
        rec = folder / 'weights.json'
        rec.write_text(json.dumps({**report, 'weights': w.tolist(), 'penalty': 100.0, 'hashes': {str(p.relative_to(ef.ROOT)): ef.sha(p) for p in files}}, indent=2))
        replay(rec, folder / 'submission.csv')
    print('passes', eligible, flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--kind', choices=['state_tree', 'claude_tree', 'harmonic_tree'], default='state_tree')
    p.add_argument('--verify', action='store_true')
    p.add_argument('--replay', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    with threadpool_limits(limits=2):
        if a.replay:
            replay(a.replay, a.output)
        else:
            run(a.kind, a.verify)
