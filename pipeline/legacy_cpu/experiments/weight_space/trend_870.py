# Fixed winning-recipe CPU adaptation v1
"""Add a local trend specialist to the frozen mixture/innovation readout."""
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
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
import expert_fusion as ef
import innovation_head as ih
import mechanism_edit as me
import seasonal_head as sh
from physical_870 import BASE, BASE_MODE, fit

def design(mode, t, z, ix):
    d = t['trend'].iloc[ix]
    gate = d.gate.to_numpy()[:, None]
    v = d[['level_bf', 'hinge_bf', 'free_level_bf', 'scale_bf', 'shape_bf']].to_numpy().clip(-10, 30)
    strength = d.intervention.to_numpy().clip(-6, 20)
    parts = [np.ones(len(z)), np.clip(z, -5, 10), np.tanh(z / 2), np.clip(strength - z, -7, 12), np.log1p(np.maximum(v, 0)), v / 6, np.arcsinh(d.coherence.clip(-12, 12)), np.log1p(d.common_slope.clip(0, 9)), np.log1p(d.slope_ratio.clip(0, 15)), d.phi, np.log(d.df), np.log(d.pre_scale.clip(0.05, 10)), np.log(d.post_scale.clip(0.05, 10))]
    if 'net' in mode:
        s = t['trend_seq'].iloc[ix].to_numpy()
        orig, c, sep = (q.mean(1) for q in np.split(s, 3, axis=1))
        diff = np.column_stack([c - z, sep - z, c - orig, sep - orig]).clip(-9, 9)
        parts.extend([diff, np.maximum(diff - 1, 0), diff * np.tanh(z[:, None] / 2)])
    return gate * np.column_stack(parts)

def names(mode):
    return ['trend'] + (['trend_seq'] if 'net' in mode else [])

def parent_folds(data, extra=()):
    _, ids, tids, y, wt, X, V, VT, labs, groups, E, D, nc = data
    backbone, zfull, g = ih.backbone(data)
    if extra:
        _, features, _, H = sh.read_tables(data, True)
        season = json.loads(ih.PARENT.read_text())
        A = features[season['cols']].to_numpy()
        for seed in extra:
            backbone[seed] = []
            for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
                w = ef.fit_views(V[tr], y[tr], groups, ef.logistic_fit(X[tr], y[tr]), 100.0)
                zr, zv = (V[i] @ w[:-1] + w[-1] for i in (tr, va))
                q = ef.distribution_maps(D[tr])
                fr, fv = (ef.design('regime', E[i], ef.ranks(D[i], q), z, nc) for i, z in ((tr, zr), (va, zv)))
                w = ef.fit_head(fr, zr, y[tr])
                zr, zv = (zr + fr @ w[:-1] + w[-1], zv + fv @ w[:-1] + w[-1])
                hr, hv = sh.harmonic_parent(H, tr, va, zr, zv, y)
                zr, zv = (sh.base(season['mode'], zr, hr, g[tr]), sh.base(season['mode'], zv, hv, g[va]))
                q = sh.lookup(A, tr, g)
                fr, fv = (sh.features(season['mode'], A[i], g[i], z, q) for i, z in ((tr, zr), (va, zv)))
                w = sh.fit(fr, zr, y[tr])
                backbone[seed].append((tr, va, zr + fr @ w, zv + fv @ w))
    t = me.tables(data, me.inputs(BASE_MODE))
    hist = np.load(ef.OUT / f'mechanism_{BASE_MODE}_oof.npz')
    assert np.array_equal(ids, hist['ids'])
    folds = {}
    for seed, parts in backbone.items():
        folds[seed] = []
        for tr, va, zr, zv in parts:
            fr, fv = (me.encode(BASE_MODE, t, z, g, i) for i, z in ((tr, zr), (va, zv)))
            w = sh.fit(fr, zr, y[tr], penalty=40.0)
            zr, zv = (zr + fr @ w, zv + fv @ w)
            if seed in (7, 29, 0):
                assert np.max(abs(zv - hist[f's{seed}_40.0'][va])) < 1e-07
            folds[seed].append((tr, va, zr, zv))
    w = np.array(json.loads(BASE.read_text())['weights'])
    zfull = zfull + me.encode(BASE_MODE, t, zfull, g, np.arange(len(g))) @ w
    official = pd.read_csv(BASE.parent / 'submission.csv', dtype={'id': str}).set_index('id').loc[tids, 'break_probability']
    assert np.max(abs(expit(zfull[len(y):]) - official)) < 1e-12
    return (folds, zfull)

def replay(recipe, path, portable=False):
    art = json.loads(recipe.read_text())
    for f, h in art['hashes'].items():
        assert ef.sha(ef.ROOT / f) == h, f
    data = ef.read_data()
    ids, tids = data[1:3]
    t = me.tables(data, names(art['mode']))
    ix = np.arange(len(ids), len(ids) + len(tids))
    p = pd.read_csv(BASE.parent / 'submission.csv', dtype={'id': str}).set_index('id').loc[tids, 'break_probability']
    z = logit(p.to_numpy())
    f = design(art['mode'], t, z, ix)
    sub = ef.write_submission(expit(z + f @ np.array(art['weights'])), tids, path)
    if portable:
        order = pd.Index(tids).get_indexer(sub.id)
        np.savez_compressed(recipe.parent / 'replay_inputs.npz', ids=tids.to_numpy(dtype=str), order=order, parent_logits=z, features=f)

def run(mode, check=None):
    data = ef.read_data()
    _, ids, tids, y, wt, *_ = data
    folds, zfull = parent_folds(data, (123, 2024, 11041) if check else ())
    t = me.tables(data, names(mode))
    penalties = _rebuild.select([60.0], [float(json.loads(check.read_text())['selected'])] if check else [12.0, 60.0])
    scores, archive = ({}, {})
    for seed, parts in folds.items():
        pred = {k: np.empty(len(y)) for k in ['parent', *penalties]}
        for tr, va, zr, zv in parts:
            pred['parent'][va] = zv
            a, b = (design(mode, t, zr, tr), design(mode, t, zv, va))
            for penalty in penalties:
                v = fit(a, zr, y[tr], penalty)
                pred[penalty][va] = zv + b @ v
        scores[seed] = {k: ef.score(y, v, wt) for k, v in pred.items()}
        archive.update({f's{seed}_{k}': v for k, v in pred.items()})
        print(seed, scores[seed], flush=True)
    if check:
        sha = ef.sha(check.parent / 'submission.csv')
        assert sha == ef.sha(check.parent / 'submission_replay.csv')
        (check.parent / 'validation.json').write_text(json.dumps({'scores': scores, 'recipe_sha256': ef.sha(check), 'submission_sha256': sha, 'rows': len(tids)}, indent=2))
        return
    eligible = [p for p in penalties if all((s[p][k] - s['parent'][k] >= 0.0003 for s in scores.values() for k in ('auc', 'weighted_auc')))]
    chosen = _rebuild.select(60.0, max(eligible, key=lambda p: np.mean([s[p]['auc'] for s in scores.values()])) if eligible else None)
    report = _rebuild.annotate({'mode': mode, 'scores': scores, 'selected': chosen})
    name = 'trend870_' + mode
    (ef.OUT / f'{name}_results.json').write_text(json.dumps(report, indent=2))
    np.savez(ef.OUT / f'{name}_oof.npz', ids=ids.to_numpy(dtype=str), **archive)
    if chosen:
        folder = ef.PACKAGE / f'{name}_candidate'
        folder.mkdir(exist_ok=True)
        tr = np.arange(len(y))
        v = fit(design(mode, t, zfull[tr], tr), zfull[tr], y, chosen)
        files = [Path(__file__).resolve(), BASE, BASE.parent / 'submission.csv']
        files.extend((ef.OUT / f'{tag}_{part}.parquet' for tag in names(mode) for part in ('train', 'test')))
        hashes = {str(f.relative_to(ef.ROOT)): ef.sha(f) for f in files}
        rec = folder / 'weights.json'
        rec.write_text(json.dumps({**report, 'weights': v.tolist(), 'hashes': hashes}, indent=2))
        replay(rec, folder / 'submission.csv', True)
    print('Selected', chosen, flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', default='law')
    p.add_argument('--replay', type=Path)
    p.add_argument('--check', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    with threadpool_limits(limits=2):
        if args.replay:
            replay(args.replay, args.output)
        else:
            run(args.mode, args.check)
