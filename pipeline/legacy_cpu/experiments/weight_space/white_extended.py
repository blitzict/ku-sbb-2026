# Fixed winning-recipe CPU adaptation v1
"""A single weight edit for jump likelihoods, walk likelihoods and held-out experts."""
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
import white_loop as wl
from white_nested import law_design, gate_fields, fit, audit
from white_outer_priors import load, fit_on
from white_joint import read_member
from partition_evidence import OUT, ROOT

def evaluate(names, seeds, weighted):
    q, ids, tids, y, wt, wn, miss = wl.load()
    n = len(y)
    fi, ft, M, mask, yy = load('physical')
    assert ids.equals(fi) and tids.equals(ft) and np.array_equal(y, yy)
    g, t, white = gate_fields(ids, tids)

    def frames(tag):
        p = [OUT / f'{tag}_{v}.parquet' for v in ('train', 'test')]
        df = pd.concat([pd.read_parquet(file).loc[ix] for file, ix in zip(p, (ids, tids))])
        return (df, p)
    process, _ = frames('innov_v2')
    walk = process.pre_walk.to_numpy()
    static = []
    features = []
    paths = []
    for name in names:
        if name in ('pulse', 'state'):
            a, p = frames('jump_law_' + name)
            paths += p
            h = np.arcsinh(a.clip(-24, 60).to_numpy())
            target = 0.2 + 0.8 * white if name == 'pulse' else 1 - 0.35 * white
            static.append(h * target[:, None])
            joint = a[['level2', 'scale2', 'shape0', 'shape1']].clip(-14, 35).to_numpy()
            joint = np.logaddexp.reduce(joint, axis=1) - np.log(joint.shape[1])
            features.append((joint[:, None], target))
        elif name == 'walk':
            a, p = frames('walk_jump')
            paths += p
            static.append(walk[:, None] * np.arcsinh(a.clip(-25, 60).to_numpy()))
            features.append((a.total.to_numpy()[:, None], walk))
        else:
            h, p = read_member(name, ids, tids)
            paths += p
            if name in ('shape', 'deconv'):
                static.append((0.2 + 0.8 * white[:, None]) * np.arcsinh(h.clip(-15, 30)))
            else:
                features.append((h, np.ones(len(white))))
    constants = np.column_stack(static) if static else np.zeros((len(white), 0))

    def design(h, i, z):
        blocks = [law_design(h, z, g[i], t[i]), constants[i]]
        for a, weight in features:
            d = weight[i, None] * (a[i] - z[:, None]).clip(-7, 8)
            blocks.extend([d, d * np.tanh(z[:, None] / 2), d * white[i, None], d * t[i, None], d * walk[i, None]])
        return np.column_stack(blocks)
    name = 'physical_' + '_'.join(names) + ('_rank' if weighted else '')
    scores = {}
    audits = {}
    pred = {}
    for seed in seeds:
        base = np.empty(n)
        new = np.empty(n)
        for fold in range(5):
            key = f's{seed}_f{fold}_'
            folder = OUT / f'law_outer{seed}'
            tr, va, zr, zv = [q[key + k] for k in ('tr', 'va', 'zr', 'zv')]
            with np.load(folder / f'fold{fold}_rows.npz') as a:
                assert np.array_equal(ids[tr], a['train_ids']) and np.array_equal(ids[va], a['validation_ids'])
            h = pd.read_parquet(folder / f'fold{fold}_train.parquet').loc[ids].to_numpy()
            w = fit(design(h[tr], tr, zr), zr, y[tr], weighted)
            base[va] = zv
            new[va] = zv + design(h[va], va, zv) @ w
        scores[seed] = {k: wl.score(y, z, wt, wn, miss) for k, z in (('base', base), ('new', new))}
        audits[seed] = audit(y, miss, base, new)
        pred[f's{seed}'] = new
        print(seed, {k: round(scores[seed]['new'][k], 7) for k in ('auc', 'remaining', 'target_vs_neg', 'weighted')}, flush=True)
    accepted = all((s['new']['auc'] > s['base']['auc'] and s['new']['remaining'] >= s['base']['remaining'] and (s['new']['target_vs_neg'] > s['base']['target_vs_neg']) for s in scores.values()))
    report = _rebuild.annotate(dict(name=name, scores=scores, audits=audits, accepted=accepted))
    (OUT / f'wn_{name}.json').write_text(json.dumps(report, indent=2))
    np.savez_compressed(OUT / f'wn_{name}_oof.npz', ids=ids.to_numpy(dtype=str), **pred)
    if _rebuild.export(accepted) and len(seeds) == 6:
        ix = np.arange(n)
        te = n + np.arange(len(tids))
        h, params = fit_on(M, mask, y, ix)
        w = fit(design(h[ix], ix, q['zr_all']), q['zr_all'], y, weighted)
        f = design(h[te], te, q['zt'])
        folder = ROOT / f'new weights/wn_{name}_candidate'
        folder.mkdir(exist_ok=True)
        file = folder / 'submission.csv'
        recipe = folder / 'weights.json'
        sub = wl.ef.write_submission(expit(q['zt'] + f @ w), tids, file)
        code = [Path(__file__).resolve(), *[Path(__file__).with_name(v).resolve() for v in ('white_nested.py', 'white_outer_priors.py', 'white_jump_priors.py', 'white_joint.py')]]
        inputs = [*code, wl.PARENT, *paths]
        recipe.write_text(json.dumps(dict(name=name, weights=w.tolist(), priors=params, hashes={str(p.relative_to(ROOT)): wl.ef.sha(p) for p in inputs}), indent=2))
        np.savez_compressed(folder / 'replay_inputs.npz', parent_logits=q['zt'], features=f, ids=tids.to_numpy(dtype=str), order=tids.get_indexer(sub.id))
        (folder / 'validation.json').write_text(json.dumps(dict(**report, recipe_sha256=wl.ef.sha(recipe), submission_sha256=wl.ef.sha(file)), indent=2))
        shutil.copyfile(wl.PARENT.parent / 'replay.py', folder / 'replay.py')
    print(name, 'accepted', accepted, 'primary audit', audits.get(7), flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('names', nargs='+')
    p.add_argument('--seeds', nargs='+', type=int, default=[7, 29, 0, 123, 2024, 11041])
    p.add_argument('--rank', action='store_true')
    a = p.parse_args()
    with threadpool_limits(limits=1):
        evaluate(a.names, a.seeds, a.rank)
