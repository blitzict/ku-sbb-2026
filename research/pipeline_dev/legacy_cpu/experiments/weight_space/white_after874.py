# CPU port; original experiments/weight_space/white_after874.py, SHA256 d93f45f5fc3b2d47ddc5ac77a215ce1a2e4c99e752a1e512dedfb24b4d41d77c
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Measure new edits against the jointly verified 0.87405 checkpoint."""
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
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits
import white_loop as wl
from white_checkpoint import CACHE, FOLDER, SEEDS
from white_nested import fit, audit
from white_joint import read_member
from partition_evidence import OUT, ROOT

def pair_fit(x, z, y):
    p = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    neg = neg[np.argsort(z[neg])]
    left = np.searchsorted(z[neg], z[p] - 1.65)
    right = np.searchsorted(z[neg], z[p] + 1.65)
    valid = right > left
    window = (right - left)[valid]
    p = p[valid]
    left = left[valid]
    count = 48
    rng = np.random.default_rng(925)
    grid = (np.arange(count)[None, :] + rng.random((len(p), 1))) / count
    j = neg[(left[:, None] + (window[:, None] * grid).astype(int)).ravel()]
    weight = np.repeat(window / (count * len(neg) * len(p)) * len(y), count)
    i = np.repeat(p, count)
    tau = 0.34

    def obj(w):
        s = z + x @ w
        v = expit((s[i] - s[j]) / tau)
        g = weight * v * (1 - v) / tau
        grad = np.bincount(i, weights=g, minlength=len(z)) - np.bincount(j, weights=g, minlength=len(z))
        return (-weight @ v + 26 * (w @ w), 52 * w - x.T @ grad)
    r = minimize(obj, np.zeros(x.shape[1]), jac=True, method='L-BFGS-B', options={'maxiter': 160, 'ftol': 1e-10})
    assert r.success, r.message
    return r.x

def run(names, method, seeds):
    q, ids, tids, y, wt, wn, miss = wl.load()
    n = len(y)
    cp = np.load(CACHE)
    assert np.array_equal(cp['ids'], ids) and str(cp['parent_hash']) == wl.ef.sha(FOLDER / 'weights.json')
    paths = []
    static = []
    members = []

    def frame(tag):
        pp = [OUT / f'{tag}_{p}.parquet' for p in ('train', 'test')]
        paths.extend(pp)
        return pd.concat([pd.read_parquet(p).loc[ix] for p, ix in zip(pp, (ids, tids))])
    s = frame('state')
    proc = frame('innov_v2')
    white, walk, cluster = (proc.pre_white.to_numpy(), proc.pre_walk.to_numpy(), s.pre_cluster.to_numpy())
    trend = frame('curve').gate.to_numpy()
    base = frame('jump_law_state')
    cols = ['level' + str(j) for j in range(6)] + ['scale' + str(j) for j in range(6)] + ['shape0', 'shape1', 'relax']
    for name in names:
        if name in ('nullstate', 'smoothvol', 'prestate'):
            a = frame('jump_law_' + name)
            d = (a[cols] - base[cols]).to_numpy().clip(-36, 36)
            v = np.arcsinh(d)
            static.extend([v, v * white[:, None], v * cluster[:, None], v * walk[:, None]])
        elif name == 'memory':
            for j in (1, 2):
                h = frame(f'memory_law{j}').clip(-20, 55)
                v = np.arcsinh(h.to_numpy())
                static.extend([v, v * white[:, None]])
                members.append((h.mixed.to_numpy()[:, None], np.ones(len(white))))
        elif name == 'clock':
            a = frame('trend_population')
            static.append(np.arcsinh(a.clip(-18, 45).to_numpy()) * (0.12 + 0.88 * trend[:, None]))
        elif name == 'influence':
            a = frame('conditional_scores')
            v = np.arcsinh(a.clip(-18, 45).to_numpy())
            static.extend([v, v[:, :27] * white[:, None]])
        elif name == 'latent':
            path = OUT / 'joint_latent_state_scores.npz'
            paths.append(path)
            with np.load(path) as a:
                assert np.array_equal(a['ids'], ids) and np.array_equal(a['tids'], tids)
                h = a['features'].reshape(len(white), 5, 2, -1)
            static.append(np.arcsinh((white[:, None, None] * h[:, :, 0] + (1 - white[:, None, None]) * h[:, :, 1]).reshape(len(h), -1)))
        else:
            h, p = read_member(name, ids, tids)
            paths += p
            members.append((h, np.ones(len(white))))
    const = np.column_stack(static) if static else np.zeros((len(white), 0))

    def design(ix, z):
        blocks = [const[ix], np.ones(len(z)), np.tanh(z / 2), white[ix] * np.tanh(z / 2)]
        for a, g in members:
            d = (a[ix] - z[:, None]).clip(-7, 8) * g[ix, None]
            blocks.extend([d, d * np.tanh(z[:, None] / 2), d * white[ix, None], d * walk[ix, None], d * trend[ix, None], d * cluster[ix, None]])
        return np.column_stack(blocks)

    def train(f, z, yy):
        return pair_fit(f, z, yy) if method == 'pairs' else fit(f, z, yy, method == 'rank')
    name = 'after874_' + method + '_' + '_'.join(names)
    scores = {}
    audits = {}
    pred = {}
    for seed in seeds:
        original = np.empty(n)
        basez = np.empty(n)
        new = np.empty(n)
        for fold in range(5):
            key = f's{seed}_f{fold}_'
            tr, va = (q[key + 'tr'], q[key + 'va'])
            zr, zv = (cp[key + 'zr'], cp[key + 'zv'])
            w = train(design(tr, zr), zr, y[tr])
            new[va] = zv + design(va, zv) @ w
            basez[va] = zv
            original[va] = q[key + 'zv']
        scores[seed] = {k: wl.score(y, z, wt, wn, miss) for k, z in (('original', original), ('base', basez), ('new', new))}
        audits[seed] = audit(y, miss, original, new)
        pred[f's{seed}'] = new
        print(seed, {k: round(scores[seed]['new'][k], 7) for k in ('auc', 'remaining', 'target_vs_neg', 'weighted')}, flush=True)
    accepted = all((s['new']['auc'] > s['base']['auc'] and s['new']['remaining'] >= s['original']['remaining'] and (s['new']['target_vs_neg'] > s['original']['target_vs_neg']) for s in scores.values()))
    report = dict(name=name, scores=scores, audits=audits, accepted=accepted)
    (OUT / f'wn_{name}.json').write_text(json.dumps(report, indent=2))
    np.savez_compressed(OUT / f'wn_{name}_oof.npz', ids=ids.to_numpy(dtype=str), **pred)
    if accepted and len(seeds) == 6:
        ix = np.arange(n)
        te = n + np.arange(len(tids))
        w = train(design(ix, cp['zr_all']), cp['zr_all'], y)
        f = design(te, cp['zt'])
        folder = ROOT / f'new weights/wn_{name}_candidate'
        folder.mkdir(exist_ok=True)
        file = folder / 'submission.csv'
        recipe = folder / 'weights.json'
        sub = wl.ef.write_submission(expit(cp['zt'] + f @ w), tids, file)
        paths = [Path(__file__).resolve(), FOLDER / 'weights.json', FOLDER / 'submission.csv', *paths]
        recipe.write_text(json.dumps(dict(name=name, weights=w.tolist(), hashes={str(p.relative_to(ROOT)): wl.ef.sha(p) for p in paths}), indent=2))
        np.savez_compressed(folder / 'replay_inputs.npz', parent_logits=cp['zt'], features=f, ids=tids.to_numpy(dtype=str), order=tids.get_indexer(sub.id))
        (folder / 'validation.json').write_text(json.dumps(dict(**report, recipe_sha256=wl.ef.sha(recipe), submission_sha256=wl.ef.sha(file)), indent=2))
        shutil.copyfile(FOLDER / 'replay.py', folder / 'replay.py')
    print(name, 'accepted', accepted, 'primary audit', audits.get(7), flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('names', nargs='+')
    p.add_argument('--method', choices=('rank', 'plain', 'pairs'), default='rank')
    p.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    a = p.parse_args()
    with threadpool_limits(limits=1):
        run(a.names, a.method, a.seeds)
