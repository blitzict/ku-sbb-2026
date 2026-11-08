# Fixed winning-recipe CPU adaptation v1
"""Fit and replay a penalized extension of the locked trend readout."""
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
import importlib
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits
import expert_fusion as ef
import mechanism_edit as me
import trend_870 as trend
from physical_870 import fit
BASE = ef.PACKAGE / 'trend870_net_candidate/weights.json'

def parent(data, extra):
    folds, z = trend.parent_folds(data, extra)
    trtable = me.tables(data, trend.names('net'))
    item = json.loads(BASE.read_text())
    penalty = item['selected']
    archive = np.load(ef.OUT / 'trend870_net_oof.npz')
    y = data[3]
    for seed, parts in folds.items():
        for j, (tr, va, zr, zv) in enumerate(parts):
            fr, fv = (trend.design('net', trtable, zr, tr), trend.design('net', trtable, zv, va))
            w = fit(fr, zr, y[tr], penalty)
            vr = zv + fv @ w
            if seed in (7, 29, 0):
                assert np.max(abs(vr - archive[f's{seed}_{penalty}'][va])) < 1e-07
            parts[j] = (tr, va, zr + fr @ w, vr)
    z = z + trend.design('net', trtable, z, np.arange(len(z))) @ np.array(item['weights'])
    return (folds, z)

def modules(spec):
    return [(importlib.import_module(item.split('.')[0] + '_channels'), item.split('.')[1]) for item in spec.split('+')]

def features(mods, t, z, ix):
    f = np.column_stack([m.design(mode, t, z, ix) for m, mode in mods])
    assert np.isfinite(f).all()
    return f

def tables(data, mods):
    tags = list(dict.fromkeys((tag for m, mode in mods for tag in m.tags(mode))))
    return (tags, me.tables(data, tags))

def replay(recipe, path, portable=False):
    art = json.loads(recipe.read_text())
    for name, h in art['hashes'].items():
        assert ef.sha(ef.ROOT / name) == h, name
    data = ef.read_data()
    ids, tids = data[1:3]
    mods = modules(art['spec'])
    _, t = tables(data, mods)
    p = pd.read_csv(BASE.parent / 'submission.csv', dtype={'id': str}).set_index('id').loc[tids, 'break_probability']
    z = logit(p.to_numpy())
    f = features(mods, t, z, np.arange(len(ids), len(ids) + len(tids)))
    sub = ef.write_submission(expit(z + f @ np.array(art['weights'])), tids, path)
    if portable:
        order = pd.Index(tids).get_indexer(sub.id)
        np.savez_compressed(recipe.parent / 'replay_inputs.npz', ids=tids.to_numpy(dtype=str), order=order, parent_logits=z, features=f)
        if _rebuild.fixed():
            from pipeline_helpers.replay_saved_head import package
            package(recipe.parent)

def run(spec, check):
    data = ef.read_data()
    ids, tids, y, wt = data[1:5]
    folds, z = parent(data, (123, 2024, 11041) if check else ())
    mods = modules(spec)
    tags, t = tables(data, mods)
    penalties = _rebuild.select([35.0], [json.loads(check.read_text())['selected']] if check else [35.0, 140.0])
    scores = {}
    archive = {}
    for seed, parts in folds.items():
        pred = {p: np.empty(len(y)) for p in ['parent', *penalties]}
        for tr, va, zr, zv in parts:
            pred['parent'][va] = zv
            fr, fv = (features(mods, t, zr, tr), features(mods, t, zv, va))
            for p in penalties:
                w = fit(fr, zr, y[tr], p)
                pred[p][va] = zv + fv @ w
        scores[seed] = {p: ef.score(y, v, wt) for p, v in pred.items()}
        archive.update({f's{seed}_{p}': v for p, v in pred.items()})
        print(seed, scores[seed], flush=True)
    if check:
        sha = ef.sha(check.parent / 'submission.csv')
        assert sha == ef.sha(check.parent / 'submission_replay.csv')
        (check.parent / 'validation.json').write_text(json.dumps(dict(scores=scores, recipe_sha256=ef.sha(check), submission_sha256=sha, rows=len(tids)), indent=2))
        shutil.copyfile(ef.PACKAGE / 'mechanism_phase_ar_candidate/replay.py', check.parent / 'replay.py')
        return
    eligible = [p for p in penalties if all((s[p][k] > s['parent'][k] for s in scores.values() for k in ('auc', 'weighted_auc')))]
    chosen = _rebuild.select(35.0, max(eligible, key=lambda p: np.mean([s[p]['auc'] for s in scores.values()])) if eligible else None)
    name = 'extended_' + spec.replace('.', '_').replace('+', '_')
    report = _rebuild.annotate(dict(spec=spec, scores=scores, selected=chosen))
    (ef.OUT / f'{name}_results.json').write_text(json.dumps(report, indent=2))
    np.savez(ef.OUT / f'{name}_oof.npz', ids=ids.to_numpy(dtype=str), **archive)
    if chosen:
        folder = ef.PACKAGE / (name + '_candidate')
        folder.mkdir(exist_ok=True)
        ix = np.arange(len(y))
        w = fit(features(mods, t, z[ix], ix), z[ix], y, chosen)
        paths = [BASE, BASE.parent / 'submission.csv', Path(__file__).resolve(), *[Path(m.__file__).resolve() for m, _ in mods]]
        paths.extend((ef.OUT / f'{tag}_{part}.parquet' for tag in tags for part in ('train', 'test')))
        hashes = {str(p.relative_to(ef.ROOT)): ef.sha(p) for p in paths}
        recipe = folder / 'weights.json'
        recipe.write_text(json.dumps({**report, 'weights': w.tolist(), 'hashes': hashes}, indent=2))
        replay(recipe, folder / 'submission.csv', True)
    print('Selected', chosen, flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--spec', default='curve.law')
    p.add_argument('--check', type=Path)
    p.add_argument('--replay', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    with threadpool_limits(limits=2):
        if args.replay:
            replay(args.replay, args.output)
        else:
            run(json.loads(args.check.read_text())['spec'] if args.check else args.spec, args.check)
