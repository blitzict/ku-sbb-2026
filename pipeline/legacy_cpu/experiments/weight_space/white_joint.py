# Fixed winning-recipe CPU adaptation v1
"""Combine fold-local law evidence with complementary out-of-fold detectors."""
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
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits
import white_loop as wl
from white_nested import law_design, gate_fields, fit, audit
from white_outer_priors import load, fit_on
from partition_evidence import OUT, ROOT

def read_member(name, ids, tids):
    if name == 'net':
        train = np.empty(len(ids))
        test = []
        paths = []
        for f in range(5):
            path = OUT / f'physical_ft_8/fold{f}.npz'
            paths.append(path)
            with np.load(path) as b:
                train[b['va']] = b['pv']
                test.append(b['pt'])
        return (np.r_[train, np.mean(test, axis=0)][:, None], paths)
    if name == 'shape':
        path = OUT / 'shape_transition_scores.npz'
        with np.load(path) as a:
            assert np.array_equal(a['ids'], ids) and np.array_equal(a['tids'], tids)
            F = a['features'][:, [0, 1, 6, 7, -6, -5]]
        return (F, [path])
    if name == 'deconv':
        path = OUT / 'deconv_pulse_scores.npz'
        with np.load(path) as a:
            assert np.array_equal(a['ids'], ids) and np.array_equal(a['tids'], tids)
            F = a['features'][:, [0, 10]]
        return (F, [path])
    path = OUT / f'{name}.npz'
    with np.load(path) as a:
        assert np.array_equal(a['ids'], ids) and np.array_equal(a['tids'], tids)
        z = logit(np.r_[a['oof'], a['test']].clip(1e-06, 1 - 1e-06))
    return (z[:, None], [path])

def combined(names, seeds):
    q, ids, tids, y, wt, wn, miss = wl.load()
    n = len(y)
    fi, ft, M, mask, yy = load('physical')
    assert ids.equals(fi) and tids.equals(ft) and np.array_equal(y, yy)
    g, t, white = gate_fields(ids, tids)
    members = []
    paths = []
    for name in names:
        f, p = read_member(name, ids, tids)
        members.append(f)
        paths.extend(p)

    def design(h, i, z):
        blocks = [law_design(h, z, g[i], t[i])]
        for name, F in zip(names, members):
            a = F[i]
            if name == 'deconv':
                d = (a - z[:, None]).clip(-7, 8) * np.column_stack([white[i], 1 - white[i]])
            elif name == 'shape':
                d = np.arcsinh(a.clip(-20, 30)) * (0.1 + 0.9 * white[i, None])
            else:
                d = (a - z[:, None]).clip(-7, 8)
            blocks.extend([d, d * np.tanh(z[:, None] / 2), d * white[i, None], d * t[i, None]])
        return np.column_stack(blocks)
    name = 'joint_' + '_'.join(names)
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
            w = fit(design(h[tr], tr, zr), zr, y[tr], True)
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
        w = fit(design(h[ix], ix, q['zr_all']), q['zr_all'], y, True)
        f = design(h[te], te, q['zt'])
        folder = ROOT / f'new weights/wn_{name}_candidate'
        folder.mkdir(exist_ok=True)
        file = folder / 'submission.csv'
        recipe = folder / 'weights.json'
        sub = wl.ef.write_submission(expit(q['zt'] + f @ w), tids, file)
        inputs = [Path(__file__).resolve(), Path(__file__).with_name('white_nested.py').resolve(), wl.PARENT, *paths]
        recipe.write_text(json.dumps(dict(name=name, weights=w.tolist(), priors=params, hashes={str(p.relative_to(ROOT)): wl.ef.sha(p) for p in inputs}), indent=2))
        np.savez_compressed(folder / 'replay_inputs.npz', parent_logits=q['zt'], features=f, ids=tids.to_numpy(dtype=str), order=tids.get_indexer(sub.id))
        (folder / 'validation.json').write_text(json.dumps(dict(**report, recipe_sha256=wl.ef.sha(recipe), submission_sha256=wl.ef.sha(file)), indent=2))
        shutil.copyfile(wl.PARENT.parent / 'replay.py', folder / 'replay.py')
    print(name, 'accepted', accepted, 'primary audit', audits.get(7), flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('names', nargs='+')
    p.add_argument('--seeds', nargs='+', type=int, default=[7, 29, 0, 123, 2024, 11041])
    a = p.parse_args()
    with threadpool_limits(limits=1):
        combined(a.names, a.seeds)
