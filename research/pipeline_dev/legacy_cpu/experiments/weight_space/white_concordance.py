# Fixed winning-recipe CPU adaptation v1
"""Edit physical expert weights by optimizing smoothed ROC concordance on nested scores."""
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
from scipy.special import expit, ndtr
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
from white_checkpoint import SEEDS
from white_joint import read_member
from white_local_rank_weights import centered
from white_crop_replication import BASE, PREVIOUS
from white_nested import audit
from partition_evidence import ROOT, OUT
import white_loop as wl

def train(x, z, y, white, focus):
    grid = np.arange(-22.0, 32.01, 0.075)
    bandwidth = 0.42
    cdf = ndtr((grid[:, None] - grid[None, :]) / bandwidth)
    stem = white * np.exp(-0.5 * ((z + 1.15) / 1.15) ** 2)
    neg = (1 - y) / np.sum(1 - y)
    pu = y / y.sum()
    hard = y * stem / (y @ stem)
    positive = (1 - focus) * pu + focus * hard

    def objective(w):
        diff = x @ w
        score = z + diff
        r = np.clip((score - grid[0]) / 0.075, 0, len(grid) - 1.001)
        k = r.astype(int)
        t = r - k

        def counts(m):
            return np.bincount(k, weights=m * (1 - t), minlength=len(grid)) + np.bincount(k + 1, weights=m * t, minlength=len(grid))
        a, b = (counts(positive), counts(neg))
        left = cdf @ b
        right = a @ cdf
        deriv = (positive * (left[k + 1] - left[k]) + neg * (right[k + 1] - right[k])) / 0.075
        N = len(z)
        bce = np.mean(np.logaddexp(0, score) - y * score)
        g = -deriv + 0.05 * (expit(score) - y) / N + 0.009 * diff / N
        val = -(a @ left) + 0.05 * bce + 0.0045 * np.mean(diff ** 2) + 0.005 * (w @ w)
        return (val, x.T @ g + 0.01 * w)
    opt = minimize(objective, np.zeros(x.shape[1]), method='L-BFGS-B', jac=True, options={'ftol': 2e-10, 'maxiter': 140, 'maxls': 50})
    if not opt.success:
        raise RuntimeError(opt.message)
    return opt.x

def run(channels, focus):
    q, ids, tids, y, wt, wn, miss = wl.load()
    n = len(y)
    base = ROOT / f'new weights/wn_{BASE}_candidate'
    rec = json.loads((base / 'weights.json').read_text())
    deps = [base / 'weights.json', Path(__file__).resolve()]

    def frame(tag, cols):
        p = [OUT / f'{tag}_{s}.parquet' for s in ('train', 'test')]
        deps.extend(p)
        return pd.concat([pd.read_parquet(a, columns=cols).loc[i] for a, i in zip(p, (ids, tids))])
    gates = frame('innov_v2', ['pre_white', 'pre_walk']).to_numpy()
    white = gates[:, 0]
    g = np.column_stack([gates, frame('volatility_law', ['pre_cluster']), frame('periodic', ['gate']), frame('curve', ['gate'])])
    experts = []
    for name in rec['members']:
        ar, p = read_member(name, ids, tids)
        experts.append(ar[:, 0])
        deps.extend(p)
    experts = np.column_stack(experts)
    physical = frame('reliable_signals', ['mode_within_mode', 'mode_occupancy', 'mode_agree']).to_numpy()
    alt = frame('quantile_alignment', ['state_shoulders_after_location_scale', 'pulse_tails_after_location_scale', 'state_width_replicate', 'pulse_location_replicate']).to_numpy()
    physical = np.column_stack([physical, np.arcsinh(alt.clip(-24, 48))])
    if channels:
        tab = frame(channels, None)
        h = tab.filter(regex='^(full|late)_').to_numpy()
        physical = np.column_stack([physical, 1.2 * np.tanh(h / 1.5)])
    baseline = np.load(OUT / (BASE + '_oof.npz'))
    secondary = np.load(OUT / 'local_rank_upstream_joint_oof.npz')
    assert np.array_equal(ids, baseline['ids']) and np.array_equal(ids, secondary['ids'])
    deps.extend([OUT / 'local_rank_upstream_joint_oof.npz', OUT / 'wn_stage_reversible_rank_oof.npz'])
    alt_cache = OUT / 'frozen_stage_reversible_rank'
    ref_cache = OUT / ('frozen_' + PREVIOUS)

    def reference(tr, z, i, v):
        a, b, c = np.quantile(z, [0.25, 0.5, 0.75])
        lo, med, hi = np.quantile(experts[tr], [0.25, 0.5, 0.75], axis=0)
        h = (experts[i] - med) * ((c - a) / (hi - lo)) + b
        gate = 1 - 0.85 * white[i] * np.exp(-0.5 * ((v + 1.1) / 1.25) ** 2)
        return v + (h - v[:, None]) * gate[:, None] @ rec['weights']

    def basis(i, z, p, u, early):
        d = np.column_stack([u - z, early - z, p - z]).clip(-5, 5)
        h = np.tanh(d / 1.3)
        gate = g[i]
        ambiguous = white[i] * np.exp(-0.5 * ((z + 1.15) / 1.15) ** 2)
        return np.column_stack([d, h, (h[:, :, None] * gate[:, None, :]).reshape(len(i), -1), h * ambiguous[:, None], 0.64 * physical[i], 0.64 * ambiguous[:, None] * physical[i]])

    def edit(tr, z, p, b, e, va, v, pp, bb, ee):
        tx, mean = centered(basis(tr, z, p, b, e), z)
        test, _ = centered(basis(va, v, pp, bb, ee), v, mean)
        w = train(tx, z, y[tr], white[tr], focus)
        return (test, w)
    name = 'local_rank_concordance_' + (channels if channels else 'experts') + ('' if focus == 0.07 else '_overall')
    scores, details, allp = ({}, {}, {})
    for s in SEEDS:
        prev = baseline[f's{s}']
        new = np.empty(n)
        for f in range(5):
            k = f's{s}_f{f}_'
            tr, va = (q[k + 'tr'], q[k + 'va'])
            with np.load(ref_cache / f's{s}_f{f}.npz') as a, np.load(alt_cache / f's{s}_f{f}.npz') as ar:
                for data in (a, ar):
                    assert np.array_equal(data['train_ids'], ids[tr]) and np.array_equal(data['validation_ids'], ids[va])
                p, v = (a['oof_train'], a['valid'])
                e, ev = (ar['oof_train'], ar['valid'])
            z, baseval = (reference(tr, p, tr, p), reference(tr, p, va, v))
            assert max(abs(baseval - prev[va])) < 1e-08
            with np.load(OUT / f'upstream_inner/s{s}_f{f}.npz') as a:
                assert np.array_equal(a['ids'], ids[tr])
                b = a['branch']
            F, w = edit(tr, z, p, b, e, va, baseval, v, secondary[f'branch_{s}'][va], ev)
            new[va] = baseval + F @ w
        allp[f's{s}'] = new
        scores[s] = {k: wl.score(y, v, wt, wn, miss) for k, v in (('previous', prev), ('new', new))}
        details[s] = audit(y, miss, prev, new)
        print(name, s, scores[s]['new'], details[s]['candidate'], flush=True)
    passed = all((v['new']['auc'] > v['previous']['auc'] for v in scores.values())) and all((scores[7]['new'][k] >= scores[7]['previous'][k] for k in ('remaining', 'target_vs_neg')))
    specialist = details[7]['candidate']['fpr_20'] > details[7]['parent']['fpr_20'] and scores[7]['new']['target_vs_neg'] > scores[7]['previous']['target_vs_neg']
    for s in SEEDS:
        old = np.empty(n)
        for f in range(5):
            k = f's{s}_f{f}_'
            old[q[k + 'va']] = q[k + 'zv']
        anchor = wl.score(y, old, wt, wn, miss)
        specialist = specialist and all((scores[s]['new'][k] > anchor[k] for k in ('auc', 'target_vs_neg', 'remaining')))
    result = _rebuild.annotate(dict(parent=BASE, focus=focus, accepted=bool(passed), recovery_specialist=bool(specialist), scores=scores, audits=details))
    (OUT / (name + '.json')).write_text(json.dumps(result, indent=2))
    np.savez_compressed(OUT / (name + '_oof.npz'), ids=ids.to_numpy(dtype=str), **allp)
    if _rebuild.export(passed) or _rebuild.export(specialist):
        tr, te = (np.arange(n), n + np.arange(len(tids)))
        with np.load(ref_cache / 'full.npz') as a, np.load(alt_cache / 'full.npz') as e:
            for data in (a, e):
                assert np.array_equal(data['train_ids'], ids) and np.array_equal(data['test_ids'], tids)
            p, v = (a['oof_train'], a['test'])
            zt = reference(tr, p, te, v)
            F, w = edit(tr, baseline['s7'], p, secondary['branch_7'], e['oof_train'], te, zt, v, secondary['branch_test'], e['test'])
        dst = ROOT / f'new weights/wn_{name}_candidate'
        dst.mkdir(exist_ok=True)
        recipe, csv = (dst / 'weights.json', dst / 'submission.csv')
        sub = wl.ef.write_submission(expit(zt + F @ w), tids, csv)
        recipe.write_text(json.dumps(dict(weights=w.tolist(), hashes={str(k.relative_to(ROOT)): wl.ef.sha(k) for k in dict.fromkeys(deps)}), indent=2))
        np.savez_compressed(dst / 'replay_inputs.npz', ids=tids.to_numpy(dtype=str), order=tids.get_indexer(sub.id), parent_logits=zt, features=F)
        (dst / 'validation.json').write_text(json.dumps(dict(**result, recipe_sha256=wl.ef.sha(recipe), submission_sha256=wl.ef.sha(csv)), indent=2))
        shutil.copyfile(base / 'replay.py', dst / 'replay.py')
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--channels')
    p.add_argument('--focus', choices=[0.0, 0.07], type=float, default=0.07)
    a = p.parse_args()
    with threadpool_limits(limits=1):
        run(a.channels, a.focus)
