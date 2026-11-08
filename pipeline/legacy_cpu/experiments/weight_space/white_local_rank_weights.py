# Fixed winning-recipe CPU adaptation v1
"""Write a compact weight correction using comparable held-out score bands."""
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
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits
from white_checkpoint import SEEDS
from white_joint import read_member
from white_nested import audit
from partition_evidence import OUT, ROOT
import white_loop as wl

def inputs(ids, tids, models, physics):
    paths = []

    def frame(tag):
        p = [OUT / f'{tag}_{part}.parquet' for part in ('train', 'test')]
        paths.extend(p)
        return pd.concat([pd.read_parquet(v).loc[i] for v, i in zip(p, (ids, tids))])
    regime = frame('innov_v2')
    vol = frame('volatility_law')
    tr = frame('curve')
    wh = regime.pre_white.to_numpy()
    walk = regime.pre_walk.to_numpy()
    trend = tr.gate.to_numpy()
    gate = np.column_stack([wh, walk, trend, vol.pre_cluster, frame('periodic').gate])
    members = []
    for name in models:
        x, p = read_member(name, ids, tids)
        paths += p
        members.append(x[:, 0])
    E = np.column_stack(members)
    physical = []
    if physics:
        h = frame('latent_level_pulse')
        for cname in ('observed_step1', 'kick1', 'driver1', 'slope1', 'pulse1', 'restart_state', 'restart_new_scale', 'restart_new_process'):
            v = np.arcsinh(h[cname].clip(-20, 32).to_numpy())
            physical.extend([v, walk * v, h.slow_hidden_pre.to_numpy() * v])
        c = frame('paired_view_v1')
        physical.extend(np.arcsinh(c.to_numpy().clip(-7, 7)).T)
        h = frame('transport_state')
        physical.extend(h.iloc[:, -2:].to_numpy().T)
    static = np.column_stack([gate, *physical]) if physical else gate

    def make(ix, z):
        e = E[ix]
        d = (e - z[:, None]).clip(-6, 6)
        med = np.median(d, 1)
        v = [0.55 * static[ix], d, 0.6 * np.tanh(d), 0.4 * abs(d), 0.55 * med[:, None] * gate[ix]]
        for c, width in ((-2.1, 0.95), (-0.8, 0.65), (0.15, 0.8), (1.45, 1.15)):
            local = np.exp(-0.5 * ((z - c) / width) ** 2)
            v.extend([0.55 * local[:, None] * d, 0.48 * local[:, None] * static[ix]])
        v.append(0.38 * ((d - med[:, None])[:, :, None] * gate[ix, None, :]).reshape(len(ix), -1))
        return np.ascontiguousarray(np.column_stack(v), dtype=float)
    return (make, paths)
GRID = np.linspace(-9, 13, 134)

def centered(x, z, ref=None):
    if ref is None:
        ind = np.rint((z - GRID[0]) / (GRID[1] - GRID[0])).astype(int).clip(0, len(GRID) - 1)
        num = np.zeros((len(GRID), x.shape[1]))
        np.add.at(num, ind, x)
        count = np.bincount(ind, minlength=len(GRID)).astype(float)
        prior = x.mean(0)
        ref = (gaussian_filter1d(num, 2.9, axis=0) + 4 * prior) / (gaussian_filter1d(count, 2.9)[:, None] + 4)
    pos = ((z - GRID[0]) / (GRID[1] - GRID[0])).clip(0, len(GRID) - 1)
    i = pos.astype(int).clip(0, len(GRID) - 2)
    frac = (pos - i)[:, None]
    return (x - ((1 - frac) * ref[i] + frac * ref[i + 1]), ref)

def weights(x, z, y):
    positives = np.flatnonzero(y)
    neg = np.flatnonzero(y == 0)
    neg = neg[np.argsort(z[neg])]
    left = np.searchsorted(z[neg], z[positives] - 1.35)
    end = np.searchsorted(z[neg], z[positives] + 1.35)
    valid = end > left
    pos = positives[valid]
    length = (end - left)[valid]
    left = left[valid]
    m = 32
    rng = np.random.default_rng(1729)
    grid = (np.arange(m) + rng.random((len(pos), 1))) / m
    i = np.repeat(pos, m)
    j = neg[(left[:, None] + (length[:, None] * grid).astype(int)).ravel()]
    pw = np.repeat(length, m) * np.exp(-0.5 * ((z[i] - z[j]) / 0.62) ** 2)
    pw *= len(y) * 0.53 / pw.sum()

    def loss(w):
        s = z + x @ w
        gap = (s[j] - s[i]) / 0.55
        p = pw * expit(gap)
        g = 0.47 * (expit(s) - y) + np.bincount(j, weights=p, minlength=len(y)) - np.bincount(i, weights=p, minlength=len(y))
        return (0.47 * np.sum(np.logaddexp(0, s) - y * s) + 0.55 * pw @ np.logaddexp(0, gap) + 84 * (w @ w), x.T @ g + 168 * w)
    r = minimize(loss, np.zeros(x.shape[1]), jac=True, method='L-BFGS-B', options={'maxiter': 300, 'ftol': 1e-11})
    assert r.success, r.message
    return r.x

def run(tag, name, models, physics):
    q, ids, tids, y, wt, wn, miss = wl.load()
    N = len(y)
    folder = ROOT / f'new weights/wn_stage_{tag}_candidate'
    cache = OUT / ('frozen_stage_' + tag)
    parent_sha = wl.ef.sha(folder / 'weights.json')
    make, paths = inputs(ids, tids, models, physics)
    preds = {}
    audits = {}
    scores = {}
    for s in SEEDS:
        first = np.empty(N)
        candidate = np.empty(N)
        orig = np.empty(N)
        for f in range(5):
            k = f's{s}_f{f}_'
            tr, va = [q[k + c] for c in ('tr', 'va')]
            with np.load(cache / f's{s}_f{f}.npz') as a:
                assert str(a['parent_sha']) == parent_sha and np.array_equal(a['train_ids'], ids[tr]) and np.array_equal(a['validation_ids'], ids[va])
                zr, zv = (a['oof_train'], a['valid'])
            x, mean = centered(make(tr, zr), zr)
            v, _ = centered(make(va, zv), zv, mean)
            w = weights(x, zr, y[tr])
            candidate[va] = zv + v @ w
            first[va] = zv
            orig[va] = q[k + 'zv']
        scores[s] = {k: wl.score(y, v, wt, wn, miss) for k, v in (('incumbent', first), ('new', candidate), ('original', orig))}
        audits[s] = audit(y, miss, orig, candidate)
        preds[f's{s}'] = candidate
        print(name, s, {k: round(scores[s]['new'][k], 7) for k in ('auc', 'remaining', 'target_vs_neg', 'weighted')}, flush=True)
    accepted = all((z['new']['auc'] > z['incumbent']['auc'] and z['new']['remaining'] > z['original']['remaining'] and (z['new']['target_vs_neg'] > z['original']['target_vs_neg']) for z in scores.values()))
    name = 'local_rank_' + name
    report = _rebuild.annotate(dict(scores=scores, audits=audits, accepted=accepted, parent=tag, models=models, physics=physics))
    (OUT / (name + '.json')).write_text(json.dumps(report, indent=2))
    np.savez_compressed(OUT / (name + '_oof.npz'), ids=ids.to_numpy(dtype=str), **preds)
    if _rebuild.export(accepted):
        with np.load(cache / 'full.npz') as a:
            assert str(a['parent_sha']) == parent_sha and np.array_equal(ids, a['train_ids']) and np.array_equal(tids, a['test_ids'])
            zr, zt = (a['oof_train'], a['test'])
        ix = np.arange(N)
        te = N + np.arange(len(tids))
        x, mean = centered(make(ix, zr), zr)
        v, _ = centered(make(te, zt), zt, mean)
        w = weights(x, zr, y)
        dst = ROOT / f'new weights/wn_{name}_candidate'
        dst.mkdir(exist_ok=True)
        csv = dst / 'submission.csv'
        sub = wl.ef.write_submission(expit(zt + v @ w), tids, csv)
        recipe = dst / 'weights.json'
        recipe.write_text(json.dumps(dict(weights=w.tolist(), parent=tag, models=models, physics=physics, centers=mean.tolist(), hashes={str(p.relative_to(ROOT)): wl.ef.sha(p) for p in (Path(__file__).resolve(), folder / 'weights.json', *paths)})))
        np.savez_compressed(dst / 'replay_inputs.npz', ids=tids.to_numpy(dtype=str), order=tids.get_indexer(sub.id), parent_logits=zt, features=v)
        (dst / 'validation.json').write_text(json.dumps(dict(**report, recipe_sha256=wl.ef.sha(recipe), submission_sha256=wl.ef.sha(csv)), indent=2))
        shutil.copyfile(folder / 'replay.py', dst / 'replay.py')
    print(name, accepted, audits[7], flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('parent')
    p.add_argument('name')
    p.add_argument('models', nargs='+')
    p.add_argument('--physics', action='store_true')
    a = p.parse_args()
    with threadpool_limits(limits=1):
        run(a.parent, a.name, a.models, a.physics)
