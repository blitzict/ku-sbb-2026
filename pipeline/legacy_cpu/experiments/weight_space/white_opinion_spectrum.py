# CPU port; original experiments/weight_space/white_opinion_spectrum.py, SHA256 36f831abca6419d88b8ab562986eeaf9a8dd9760dfb2b6d53a300e0f54b143ea
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Learn from shared and contrasting directions across the verified OOF weight families."""
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
from scipy.linalg import eigh
from scipy.special import expit
from threadpoolctl import threadpool_limits
from white_checkpoint import SEEDS
from white_joint import read_member
from white_freeze_first_layer import NAME
from white_local_rank_weights import centered
from white_nested import fit, audit
from partition_evidence import ROOT, OUT
import white_loop as wl
FAMILIES = [('claude_crop1', 'claude_cropbag', 'claude_cropvolbag', 'claude_revorig', 'claude_revlow', 'claude_three_crop_vol', 'claude_four_crop_vol', 'claude_two_crop_k7', 'claude_two_crop_k29', 'claude_two_crop_10x43', 'claude_source_loss', 'claude_source_loss_k7', 'claude_source_loss_k29'), ('lawtree_both', 'lawvoltree', 'lawvoltree_res', 'lawrank_crops', 'lawrank_original', 'law_linked_rank_s29', 'law_linked_rank_s10x43', 'law_independent_rank_s29', 'law_independent_rank_s7', 'law_symmetric_pairs', 'law_symmetric_binary'), ('lawshared', 'lawconditional', 'lawpredictive', 'lawfourier', 'lawvectors', 'lawrankdensity', 'lawforecast', 'lawblock', 'lawretrieval', 'lawpatchforecast', 'lawrestart', 'law_weight_ensemble_s29', 'law_weight_ensemble_s29_res', 'law_rank_dependence_s29', 'net')]

def inputs(ids, tids):
    state = wl.ef.read_data()
    assert state[1].equals(ids) and state[2].equals(tids)
    old = np.vstack([state[6], state[7]])
    refs = [old[:, a:b].mean(1) for a, b, _ in state[9]]
    oldmean = np.column_stack(refs)
    packs = [oldmean, old]
    evidence = []
    files = [wl.ef.BASE]
    for group in FAMILIES:
        values = []
        for name in group:
            a, p = read_member(name, ids, tids)
            values.append(a[:, 0])
            files.extend(p)
        packs.append(np.column_stack(values))
    for part, ix in (('train', ids), ('test', tids)):
        data = []
        for tag, col in (('innov_v2', ['pre_white', 'pre_walk']), ('curve', ['gate']), ('periodic', ['gate']), ('volatility_law', ['pre_cluster']), ('mode_residual', ['gate'])):
            p = OUT / f'{tag}_{part}.parquet'
            files.append(p)
            data.append(pd.read_parquet(p, columns=col).loc[ix].to_numpy())
        evidence.append(np.column_stack(data))
    return (packs, np.vstack(evidence), files)

def design(packs, reg, tr, va, z, v):
    rows = np.r_[tr, va]
    n = len(tr)
    score = np.r_[z, v]
    coords = []
    for i, bank in enumerate(packs):
        q = np.quantile(bank[tr], [0.1, 0.5, 0.9], axis=0)
        a, b, c = np.quantile(z, [0.1, 0.5, 0.9])
        h = (bank[rows] - q[1]) * (c - a) / (q[2] - q[0]).clip(0.3) + b - score[:, None]
        h = np.arcsinh(h.clip(-9, 9))
        h0, ref = centered(h[:n], z)
        h1, _ = centered(h[n:], v, ref)
        h = np.vstack([h0, h1])
        val, vec = eigh(h0.T @ h0 / n)
        keep = np.flatnonzero(val > 0.0015)[-(10 if i != 1 else 14):][::-1]
        pcs = h @ vec[:, keep] / (val[keep] + 0.065) ** 0.5
        coords.append(pcs.clip(-4, 4))
    h = np.column_stack(coords)
    summary = np.column_stack([p[:, :min(3, p.shape[1])] for p in coords])
    g = reg[rows]
    base = np.column_stack([h, np.tanh(score[:, None] / 1.3) * summary, 0.62 * (summary[:, :, None] * (g[:, None, :] - 0.45)).reshape(len(rows), -1)])
    return (base, summary)

def terms(base, coords, quadratic):
    if not quadratic:
        return base
    i, j = np.triu_indices(coords.shape[1])
    cross = (coords[:, i] * coords[:, j]).clip(-6, 6)
    return np.column_stack([base, 0.55 * cross])

def main():
    q, ids, tids, y, wt, wn, miss = wl.load()
    n = len(y)
    packs, reg, files = inputs(ids, tids)
    parent = ROOT / f'new weights/wn_{NAME}_candidate'
    sha = wl.ef.sha(parent / 'weights.json')
    cache = OUT / ('frozen_' + NAME)
    archive = np.load(OUT / (NAME + '_oof.npz'))
    assert np.array_equal(archive['ids'], ids)
    scores = {k: {} for k in ('linear', 'covariance')}
    details = {k: {} for k in scores}
    arr = {k: {} for k in scores}
    for s in SEEDS:
        old = archive[f's{s}']
        new = {k: np.empty(n) for k in scores}
        locked = np.empty(n)
        for f in range(5):
            key = f's{s}_f{f}_'
            tr, va = (q[key + 'tr'], q[key + 'va'])
            with np.load(cache / f's{s}_f{f}.npz') as a:
                assert str(a['parent_sha']) == sha and np.array_equal(a['train_ids'], ids[tr]) and np.array_equal(a['validation_ids'], ids[va])
                z, v = (a['oof_train'], a['valid'])
            assert np.max(abs(old[va] - v)) < 1e-08
            h, p = design(packs, reg, tr, va, z, v)
            for k in scores:
                H = terms(h, p, k == 'covariance')
                w = fit(H[:len(tr)], z, y[tr], True)
                new[k][va] = v + 0.5 * (H[len(tr):] @ w)
            locked[va] = q[key + 'zv']
        for k in scores:
            arr[k][f's{s}'] = new[k]
            scores[k][s] = {a: wl.score(y, b, wt, wn, miss) for a, b in (('locked', locked), ('previous', old), ('new', new[k]))}
            details[k][s] = audit(y, miss, old, new[k])
            print(k, s, [round(scores[k][s]['new'][i], 7) for i in ('auc', 'target_vs_neg', 'remaining')], flush=True)
    for k, val in scores.items():
        passed = all((v['new']['auc'] > v['previous']['auc'] and all((v['new'][m] > v['locked'][m] for m in ('remaining', 'target_vs_neg'))) for v in val.values()))
        tag = 'local_rank_opinion_' + k
        result = dict(accepted=bool(passed), scores=val, audits=details[k])
        (OUT / (tag + '.json')).write_text(json.dumps(result, indent=2))
        np.savez_compressed(OUT / (tag + '_oof.npz'), ids=ids.to_numpy(dtype=str), **arr[k])
        if not passed:
            continue
        with np.load(cache / 'full.npz') as a:
            assert np.array_equal(a['test_ids'], tids)
            z, v = (a['oof_train'], a['test'])
        tr = np.arange(n)
        te = n + np.arange(len(tids))
        h, p = design(packs, reg, tr, te, z, v)
        H = terms(h, p, k == 'covariance')
        w = 0.5 * fit(H[:n], z, y, True)
        F = H[n:]
        dst = ROOT / f'new weights/wn_{tag}_candidate'
        dst.mkdir(exist_ok=True)
        rec, csv = (dst / 'weights.json', dst / 'submission.csv')
        sub = wl.ef.write_submission(expit(v + F @ w), tids, csv)
        deps = [parent / 'weights.json', Path(__file__).resolve(), *files]
        rec.write_text(json.dumps(dict(weights=w.tolist(), groups=FAMILIES, hashes={str(p.relative_to(ROOT)): wl.ef.sha(p) for p in dict.fromkeys(deps)}), indent=2))
        np.savez_compressed(dst / 'replay_inputs.npz', ids=tids.to_numpy(dtype=str), order=tids.get_indexer(sub.id), parent_logits=v, features=F)
        (dst / 'validation.json').write_text(json.dumps(dict(**result, recipe_sha256=wl.ef.sha(rec), submission_sha256=wl.ef.sha(csv)), indent=2))
        shutil.copyfile(parent / 'replay.py', dst / 'replay.py')
if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
