# CPU port; original experiments/weight_space/white_local_covariance.py, SHA256 f6b9b92027d76db0211f4543b430eea50666c0d4de2dadfb4bcfc5ae595cd78b
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Measure how class-conditional model disagreement varies among similarly scored sources."""
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
from scipy.linalg import eigh, solve_triangular
from scipy.special import expit
from threadpoolctl import threadpool_limits
from white_checkpoint import SEEDS
from white_opinion_spectrum import NAME, design, inputs
from white_score_scale import REFERENCE
from white_paired_branches import test_scores
from white_nested import audit
from partition_evidence import ROOT, OUT
import white_loop as wl

def log_density(x, mean, cov):
    L = np.linalg.cholesky(cov)
    d = solve_triangular(L, (x - mean).T, lower=True, check_finite=False)
    return -np.sum(np.log(np.diag(L))) - 0.5 * np.sum(d * d, axis=0)

def evidence(packs, reg, tr, va, z, v, labels):
    _, votes = design(packs, reg, tr, va, z, v)
    n = len(tr)
    mu = votes[:n].mean(0)
    eig, vec = eigh(np.cov(votes[:n].T))
    keep = np.flatnonzero(eig > 0.04)[-10:]
    x = np.tanh((votes - mu) @ vec[:, keep] / (3 * np.sqrt(eig[keep]))) * 3
    y = labels[tr]
    left = x[:n]
    right = x[n:]
    global_cov = np.cov(left.T) + 0.075 * np.eye(len(keep))
    centers = np.quantile(z, np.linspace(0.035, 0.965, 13))
    radii = np.maximum(0.44, 0.9 * np.gradient(centers))
    interpolation = np.exp(-0.5 * ((np.clip(v, centers[0], centers[-1])[:, None] - centers) / radii) ** 2)
    interpolation /= interpolation.sum(1, keepdims=True)
    linear = np.zeros(len(va))
    curved = np.zeros(len(va))
    for j, (anchor, width) in enumerate(zip(centers, radii)):
        proximity = np.exp(-0.5 * ((z - anchor) / width) ** 2)
        means = []
        covs = []
        masses = []
        for cls in (0, 1):
            weight = proximity * (y == cls)
            mass = weight.sum()
            masses.append(mass)
            mean = weight @ left / mass
            d = left - mean
            means.append(mean)
            covs.append(d.T @ (weight[:, None] * d) / mass)
        pool = (masses[0] * covs[0] + masses[1] * covs[1] + 250 * global_cov) / (sum(masses) + 250)
        pool += 0.045 * np.eye(len(keep))
        ll = []
        lq = []
        for m, c, k in zip(means, covs, masses):
            w = k / (k + 800.0)
            ll.append(log_density(right, m, pool))
            lq.append(log_density(right, m, (1 - w) * pool + w * c + 0.015 * np.eye(len(keep))))
        weight = interpolation[:, j]
        linear += weight * (ll[1] - ll[0])
        curved += weight * (lq[1] - lq[0])
    return np.column_stack([linear, curved - linear]).clip(-5.0, 5.0)

def main():
    q, ids, tids, y, wt, wn, miss = wl.load()
    n = len(ids)
    packs, reg, paths = inputs(ids, tids)
    cache = OUT / ('frozen_' + NAME)
    reference = ROOT / f'new weights/wn_{REFERENCE}_candidate'
    scores = {}
    audits = {}
    arrays = {}
    rules = {'scatter': np.array([0.0, 0.2, 0.0]), 'local_law': np.array([0.22, 0.22, -0.22])}
    with np.load(OUT / (REFERENCE + '_oof.npz')) as archive:
        assert np.array_equal(archive['ids'], ids)
        for s in SEEDS:
            old = archive[f's{s}']
            h = np.empty((n, 3))
            locked = np.empty(n)
            for f in range(5):
                k = f's{s}_f{f}_'
                tr, va = (q[k + 'tr'], q[k + 'va'])
                with np.load(cache / f's{s}_f{f}.npz') as a:
                    assert np.array_equal(ids[tr], a['train_ids']) and np.array_equal(ids[va], a['validation_ids'])
                    z, v = (a['oof_train'], a['valid'])
                h[va, :2] = evidence(packs, reg, tr, va, z, v, y)
                h[va, 2] = old[va] - v
                locked[va] = q[k + 'zv']
            for name, w in rules.items():
                p = old + h @ w
                metric = {k: wl.score(y, o, wt, wn, miss) for k, o in (('locked', locked), ('previous', old), ('new', p))}
                scores.setdefault(name, {})[s] = metric
                audits.setdefault(name, {})[s] = audit(y, miss, old, p)
                arrays.setdefault(name, {})[f's{s}'] = p
                print(name, s, [round(metric['new'][k], 8) for k in ('auc', 'target_vs_neg', 'remaining')], flush=True)
    for name, report in scores.items():
        ok = all((r['new']['auc'] > r['previous']['auc'] and all((r['new'][k] > r['locked'][k] for k in ('target_vs_neg', 'remaining'))) for r in report.values()))
        tag = 'local_rank_disagreement_' + name
        result = dict(accepted=bool(ok), scores=report, audits=audits[name])
        (OUT / (tag + '.json')).write_text(json.dumps(result, indent=2))
        np.savez_compressed(OUT / (tag + '_oof.npz'), ids=ids.to_numpy(dtype=str), **arrays[name])
        if not ok:
            continue
        with np.load(cache / 'full.npz') as a:
            assert np.array_equal(a['test_ids'], tids)
            z, v = (a['oof_train'], a['test'])
        old = test_scores(REFERENCE, tids)
        h = np.column_stack([evidence(packs, reg, np.arange(n), n + np.arange(len(tids)), z, v, y), old - v])
        w = rules[name]
        dst = ROOT / f'new weights/wn_{tag}_candidate'
        dst.mkdir(exist_ok=True)
        csv = dst / 'submission.csv'
        rec = dst / 'weights.json'
        sub = wl.ef.write_submission(expit(old + h @ w), tids, csv)
        deps = [Path(__file__).resolve(), reference / 'weights.json', *paths]
        rec.write_text(json.dumps(dict(weights=w.tolist(), hashes={str(p.relative_to(ROOT)): wl.ef.sha(p) for p in dict.fromkeys(deps)}), indent=2))
        np.savez_compressed(dst / 'replay_inputs.npz', ids=tids.to_numpy(dtype=str), order=tids.get_indexer(sub.id), parent_logits=old, features=h)
        (dst / 'validation.json').write_text(json.dumps(dict(**result, recipe_sha256=wl.ef.sha(rec), submission_sha256=wl.ef.sha(csv)), indent=2))
        shutil.copyfile(reference / 'replay.py', dst / 'replay.py')
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
