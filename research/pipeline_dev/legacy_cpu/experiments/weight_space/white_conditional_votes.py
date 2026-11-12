# CPU port; original experiments/weight_space/white_conditional_votes.py, SHA256 d5ffe72a25b274febab0dc4c5b1fd4467c62191b205ae0a36cfaca4663f60ab8
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Estimate how unusual each set of detector votes is among comparable controls."""
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
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import csr_matrix
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from white_checkpoint import SEEDS
from white_joint import read_member
from white_local_rank_weights import centered, weights
from white_nested import audit
from partition_evidence import OUT, ROOT
import white_loop as wl
GRID = np.linspace(-6, 8, 33)
FAMILIES = (('claude_cropvolbag', 'claude_revlow', 'claude_revorig', 'claude_three_crop_vol'), ('lawrank_original', 'lawrank_crops', 'lawtree_both', 'lawgrowth_c1_c2_c3_control'), ('lawpredictive', 'lawrankdensity', 'lawforecast', 'lawpatchforecast', 'lawrestart'), ('net', 'viewnet_c1', 'lawconditional', 'lawshared', 'lawblock'))

def charts(ids, tids):
    dep = []
    groups = []
    for family in FAMILIES:
        scores = []
        for tag in family:
            u, p = read_member(tag, ids, tids)
            dep.extend(p)
            scores.append(u[:, 0])
        groups.append(np.column_stack(scores))

    def table(name, cols):
        paths = [OUT / f'{name}_{p}.parquet' for p in ('train', 'test')]
        dep.extend(paths)
        return pd.concat([pd.read_parquet(p, columns=cols).loc[ix] for p, ix in zip(paths, (ids, tids))]).to_numpy()
    wh, walk = table('innov_v2', ['pre_white', 'pre_walk']).T
    osc = table('periodic', ['gate'])[:, 0]
    cluster = table('volatility_law', ['pre_cluster'])[:, 0]
    trend = table('curve', ['gate'])[:, 0]
    g = np.column_stack([np.ones(len(wh)), wh, walk, trend, osc, cluster]).clip(0, 1)
    groups.append(np.column_stack([np.mean(h, 1) for h in groups]))
    return (groups, g, list(dict.fromkeys(dep)))

def histogram(z, g, y):
    bins = np.rint((z - GRID[0]) / (GRID[1] - GRID[0])).astype(int).clip(0, len(GRID) - 1)
    r = len(y)
    K = g.shape[1]
    group = bins[:, None] + len(GRID) * (np.arange(K)[None, :] + K * y[:, None])
    return csr_matrix((g.ravel(), (group.ravel(), np.repeat(np.arange(r), K))), shape=(2 * K * len(GRID), r))

def table(h, dims, groups, prior=40.0):
    n, d = h.shape
    i, j = np.triu_indices(d)
    values = np.column_stack([np.ones(n), h, h[:, i] * h[:, j]])
    stat = (groups @ values).reshape(2, dims, len(GRID), -1)
    stat = gaussian_filter1d(stat, 1.55, axis=2)
    count = stat[:, :, :, 0]
    population = stat[:, 0].sum(0)
    global_mean = values.mean(0)
    common = (population + 18 * global_mean) / (population[:, [0]] + 18)
    overall = (stat[:, 0] + prior * common[None, :, :]) / (count[:, 0, :, None] + prior)
    moments = (stat + prior * overall[:, None, :, :]) / (count[:, :, :, None] + prior)
    mu = moments[:, :, :, 1:1 + d]
    c = np.zeros((*mu.shape[:-1], d, d))
    e = moments[:, :, :, 1 + d:] - mu[:, :, :, i] * mu[:, :, :, j]
    c[:, :, :, i, j] = e
    c[:, :, :, j, i] = e
    diag = np.arange(d)
    c = 0.94 * c + 0.06 * c.mean(axis=0, keepdims=True)
    c[:, :, :, diag, diag] += 0.024
    shared = (c[0] + c[1]) * 0.5
    return (mu, np.linalg.inv(c), np.linalg.slogdet(c)[1], np.linalg.inv(shared))

def apply(h, z, g, tables):
    mu, inv, logdet, pool = tables
    pos = ((z - GRID[0]) / (GRID[1] - GRID[0])).clip(0, len(GRID) - 1)
    low = pos.astype(int).clip(0, len(GRID) - 2)
    w = pos - low
    output = []
    for regime in range(g.shape[1]):
        mixture = []
        for ix in (low, low + 1):
            r0 = h - mu[0, regime, ix]
            r1 = h - mu[1, regime, ix]
            quad = lambda r, M: np.einsum('ni,nij,nj->n', r, M, r)
            linear = 0.5 * (quad(r0, pool[regime, ix]) - quad(r1, pool[regime, ix]))
            qda = 0.5 * (quad(r0, inv[0, regime, ix]) - quad(r1, inv[1, regime, ix]) + logdet[0, regime, ix] - logdet[1, regime, ix])
            mixture.append(np.column_stack([linear, qda - linear]))
        x = (1 - w[:, None]) * mixture[0] + w[:, None] * mixture[1]
        output.append(np.tanh(x / 2) * g[:, [regime]])
    return np.column_stack(output)

def conditional(groups, g, tr, va, zr, zv, y):
    mat = histogram(zr, g[tr], y[tr])
    features = []
    for group in groups:
        h = np.tanh((group[tr] - zr[:, None]) / 1.7)
        v = np.tanh((group[va] - zv[:, None]) / 1.7)
        params = table(h, g.shape[1], mat)
        features.append(apply(v, zv, g[va], params))
    return np.column_stack(features)

def main():
    name = 'local_rank_conditional_votes'
    parent = 'local_rank_family_agreement'
    q, ids, tids, y, wt, wn, miss = wl.load()
    groups, g, deps = charts(ids, tids)
    pack = ROOT / f'new weights/wn_{parent}_candidate'
    digest = wl.ef.sha(pack / 'weights.json')
    cache = OUT / ('frozen_' + parent)
    arrays = {}
    scores = {}
    audits = {}

    def one(tr, va, zr, zv):
        p = conditional(groups, g, tr, va, zr, zv, y)
        x = np.empty((len(tr), p.shape[1]))
        for a, b in StratifiedKFold(3, shuffle=True, random_state=729).split(tr, y[tr]):
            x[b] = conditional(groups, g, tr[a], tr[b], zr[a], zr[b], y)
        x, mean = centered(x, zr)
        p, _ = centered(p, zv, mean)
        w = weights(x, zr, y[tr])
        return (zv + p @ w, p, w)
    for s in SEEDS:
        old = np.empty(len(y))
        new = np.empty(len(y))
        base = np.empty(len(y))
        for f in range(5):
            key = f's{s}_f{f}_'
            tr, va = (q[key + k] for k in ('tr', 'va'))
            with np.load(cache / f's{s}_f{f}.npz') as a:
                assert str(a['parent_sha']) == digest and np.array_equal(ids[va], a['validation_ids']) and np.array_equal(ids[tr], a['train_ids'])
                zr, zv = (a['oof_train'], a['valid'])
            new[va] = one(tr, va, zr, zv)[0]
            old[va] = zv
            base[va] = q[key + 'zv']
        scores[s] = {k: wl.score(y, a, wt, wn, miss) for k, a in (('original', base), ('incumbent', old), ('new', new))}
        audits[s] = audit(y, miss, base, new)
        arrays[f's{s}'] = new
        print(name, s, {k: round(scores[s]['new'][k], 7) for k in ('auc', 'target_vs_neg', 'remaining', 'weighted')}, flush=True)
    accepted = all((v['new']['auc'] > v['incumbent']['auc'] and v['new']['remaining'] > v['original']['remaining'] and (v['new']['target_vs_neg'] > v['original']['target_vs_neg']) for v in scores.values()))
    report = dict(parent=parent, accepted=accepted, scores=scores, audits=audits)
    (OUT / (name + '.json')).write_text(json.dumps(report, indent=2))
    np.savez_compressed(OUT / (name + '_oof.npz'), ids=ids.to_numpy(dtype=str), **arrays)
    if accepted:
        dst = ROOT / f'new weights/wn_{name}_candidate'
        dst.mkdir(exist_ok=True)
        with np.load(cache / 'full.npz') as a:
            assert str(a['parent_sha']) == digest and np.array_equal(a['train_ids'], ids) and np.array_equal(a['test_ids'], tids)
            zr, zv = (a['oof_train'], a['test'])
        pred, ft, w = one(np.arange(len(y)), len(y) + np.arange(len(tids)), zr, zv)
        csv = dst / 'submission.csv'
        sub = wl.ef.write_submission(expit(pred), tids, csv)
        recipe = dst / 'weights.json'
        recipe.write_text(json.dumps(dict(parent=parent, families=FAMILIES, weights=w.tolist(), hashes={str(p.relative_to(ROOT)): wl.ef.sha(p) for p in [Path(__file__).resolve(), pack / 'weights.json', *deps]})))
        np.savez_compressed(dst / 'replay_inputs.npz', ids=tids.to_numpy(dtype=str), order=tids.get_indexer(sub.id), parent_logits=zv, features=ft)
        (dst / 'validation.json').write_text(json.dumps(dict(**report, recipe_sha256=wl.ef.sha(recipe), submission_sha256=wl.ef.sha(csv)), indent=2))
        shutil.copyfile(pack / 'replay.py', dst / 'replay.py')
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
