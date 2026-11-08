# Fixed winning-recipe CPU adaptation v1
"""Fit new physical evidence jointly with the locked bank of expert weights."""
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
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits
from white_checkpoint import SEEDS
from white_consult import CORE, compile_inputs
from white_nested import fit, audit
from white_outer_priors import load, fit_on
from partition_evidence import OUT, ROOT
import white_loop as wl

def new_inputs(names, ids, tids):
    paths = []
    fields = []

    def table(tag):
        pp = [OUT / f'{tag}_{part}.parquet' for part in ('train', 'test')]
        paths.extend(pp)
        return pd.concat([pd.read_parquet(p).loc[ix] for p, ix in zip(pp, (ids, tids))])
    vol = table('volatility_law')
    reliability = np.maximum(vol.pre_cluster, vol.pooled_cluster).to_numpy()
    white = table('innov_v2').pre_white.to_numpy()
    for name in names:
        tab = table(name)
        if name == 'hetero_jump':
            A = tab.filter(regex='^split_').to_numpy()
            B = tab.filter(regex='^common_').to_numpy()
            h = np.arcsinh(A.clip(-22, 60))
            difference = np.arcsinh((B - A).clip(-25, 40))
            fields.extend([0.85 * h, 0.7 * h * reliability[:, None], 0.55 * difference])
        elif name.startswith('native_law'):
            A = tab.filter(regex='^native_').to_numpy()
            B = tab.filter(regex='^standard_').to_numpy()
            x = np.arcsinh(A.clip(-22, 60))
            diff = np.arcsinh((A - B).clip(-22, 40))
            fields.extend([0.7 * x, 0.55 * diff, 0.45 * x * white[:, None]])
        else:
            h = np.arcsinh(tab.to_numpy().clip(-24, 60))
            fields.extend([0.65 * h, 0.4 * h * white[:, None]])
    return (np.column_stack(fields) if fields else np.zeros((len(white), 0)), paths)

def run(tag, tables, extra, seeds):
    q, ids, tids, y, wt, wn, miss = wl.load()
    n = len(y)
    models = CORE + ['claude_bag2', 'claude_bag3', *extra]
    base, paths = compile_inputs(ids, tids, models, ['volatility'])
    side, p = new_inputs(tables, ids, tids)
    paths += p

    def inputs(h, i, z):
        return np.column_stack([base(h, i, z, 'full'), side[i]])
    prev = OUT / 'wn_consult_full_bags_vol_oof.npz'
    with np.load(prev) as a:
        assert np.array_equal(a['ids'], ids)
        reference = {s: a[f's{s}'] for s in seeds}
    scores = {}
    audits = {}
    out = {}
    for s in seeds:
        parent = np.empty(n)
        z = np.empty(n)
        for f in range(5):
            k = f's{s}_f{f}_'
            tr, va, zr, zv = [q[k + c] for c in ('tr', 'va', 'zr', 'zv')]
            d = OUT / f'law_outer{s}'
            rows = np.load(d / f'fold{f}_rows.npz')
            assert np.array_equal(rows['train_ids'], ids[tr]) and np.array_equal(rows['validation_ids'], ids[va])
            h = pd.read_parquet(d / f'fold{f}_train.parquet').loc[ids].to_numpy()
            w = fit(inputs(h[tr], tr, zr), zr, y[tr], True)
            z[va] = zv + inputs(h[va], va, zv) @ w
            parent[va] = zv
        scores[s] = {k: wl.score(y, v, wt, wn, miss) for k, v in (('original', parent), ('incumbent', reference[s]), ('new', z))}
        audits[s] = audit(y, miss, parent, z)
        out[f's{s}'] = z
        print(tag, s, {k: round(scores[s]['new'][k], 7) for k in ('auc', 'remaining', 'target_vs_neg', 'weighted')}, flush=True)
    accepted = all((v['new']['auc'] > v['incumbent']['auc'] and v['new']['remaining'] >= v['original']['remaining'] and (v['new']['target_vs_neg'] > v['original']['target_vs_neg']) for v in scores.values()))
    name = 'stage_' + tag
    rep = _rebuild.annotate(dict(name=name, models=models, tables=tables, scores=scores, audits=audits, accepted=accepted))
    (OUT / f'wn_{name}.json').write_text(json.dumps(rep, indent=2))
    np.savez_compressed(OUT / f'wn_{name}_oof.npz', ids=ids.to_numpy(dtype=str), **out)
    if len(seeds) == 6 and _rebuild.export(accepted):
        fi, ft, M, mask, yy = load('physical')
        assert ids.equals(fi) and tids.equals(ft) and np.array_equal(y, yy)
        ix = np.arange(n)
        te = n + np.arange(len(tids))
        h, prior = fit_on(M, mask, y, ix)
        w = fit(inputs(h[ix], ix, q['zr_all']), q['zr_all'], y, True)
        F = inputs(h[te], te, q['zt'])
        dst = ROOT / f'new weights/wn_{name}_candidate'
        dst.mkdir(exist_ok=True)
        csv, recipe = (dst / 'submission.csv', dst / 'weights.json')
        sub = wl.ef.write_submission(expit(q['zt'] + F @ w), tids, csv)
        dependencies = [Path(__file__).resolve(), Path(__file__).with_name('white_consult.py').resolve(), wl.PARENT, *paths]
        recipe.write_text(json.dumps(dict(name=name, models=models, tables=tables, weights=w.tolist(), priors=prior, hashes={str(p.relative_to(ROOT)): wl.ef.sha(p) for p in dependencies}), indent=2))
        np.savez_compressed(dst / 'replay_inputs.npz', ids=tids.to_numpy(dtype=str), parent_logits=q['zt'], features=F, order=tids.get_indexer(sub.id))
        (dst / 'validation.json').write_text(json.dumps(dict(**rep, recipe_sha256=wl.ef.sha(recipe), submission_sha256=wl.ef.sha(csv)), indent=2))
        shutil.copyfile(wl.PARENT.parent / 'replay.py', dst / 'replay.py')
    print(name, accepted, audits.get(7), flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('tag')
    p.add_argument('--tables', nargs='*', default=[])
    p.add_argument('--extra', nargs='*', default=[])
    p.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    a = p.parse_args()
    with threadpool_limits(limits=1):
        run(a.tag, a.tables, a.extra, a.seeds)
