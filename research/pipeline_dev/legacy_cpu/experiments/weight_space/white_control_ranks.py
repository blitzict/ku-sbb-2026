# CPU port; original experiments/weight_space/white_control_ranks.py, SHA256 6267542e936ceb5b726d4e2db4ef63f64156b2c922a9fd58955ccd9ca8f4b2ab
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Change expert weights while anchoring no-break score percentiles within each regime."""
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
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from white_checkpoint import SEEDS
from white_joint import read_member
from white_nested import audit
from white_local_rank_weights import weights
from white_upstream_stack import collect, train_basis
from white_crop_replication import BASE, PREVIOUS
from partition_evidence import ROOT, OUT
import white_loop as wl

def calibrated(z, b, i, white, walk, tr, y, zt, bt, it):

    def combine(p, v, wh):
        return p + 0.38 * (1 - wh * np.exp(-0.5 * ((p + 1.1) / 1.35) ** 2)) * (v - p)
    mixed = combine(z, b, white[i])
    query = combine(zt, bt, white[it])
    result = query.copy()
    grid = np.linspace(0.005, 0.995, 71)

    def group(ix):
        return np.where(walk[ix] > 0.5, 3, np.digitize(white[ix], [0.6, 0.84]))
    ti, te = (group(i), group(it))
    for g in np.unique(te):
        mask = (y[tr] == 0) & (ti == g)
        assert mask.sum() > 50
        src = np.quantile(mixed[mask], grid)
        dest = np.quantile(z[mask], grid)
        correction = gaussian_filter1d(dest - src, 1.7) * mask.sum() / (mask.sum() + 120)
        pick = te == g
        result[pick] += np.interp(query[pick], src, correction)
    return result

def main():
    q, ids, tids, y, wt, wn, miss = wl.load()
    n = len(y)
    raw = collect(ids, tids)
    guide = raw[0]
    white, walk = raw[3][:, :2].T
    basefolder = ROOT / f'new weights/wn_{BASE}_candidate'
    spec = json.loads((basefolder / 'weights.json').read_text())
    experts = np.column_stack([read_member(name, ids, tids)[0][:, 0] for name in spec['members']])

    def baseline(tr, z, i, v):
        a, b, c = np.quantile(z, [0.25, 0.5, 0.75])
        lo, med, hi = np.quantile(experts[tr], [0.25, 0.5, 0.75], axis=0)
        cal = (experts[i] - med) * ((c - a) / (hi - lo)) + b
        gate = 1 - 0.85 * white[i] * np.exp(-0.5 * ((v + 1.1) / 1.25) ** 2)
        return v + gate[:, None] * (cal - v[:, None]) @ spec['weights']
    ref = np.load(OUT / (BASE + '_oof.npz'))
    branch = np.load(OUT / 'local_rank_upstream_joint_oof.npz')
    assert np.array_equal(ids, ref['ids']) and np.array_equal(ids, branch['ids'])
    saved = OUT / 'upstream_inner'
    saved.mkdir(exist_ok=True)
    cache = OUT / ('frozen_' + PREVIOUS)
    scores, details, pred = ({}, {}, {})
    tag = 'local_rank_control_ranks'
    for s in SEEDS:
        prev = ref[f's{s}']
        side = branch[f'branch_{s}']
        new = np.empty(n)
        for f in range(5):
            k = f's{s}_f{f}_'
            tr, va = (q[k + 'tr'], q[k + 'va'])
            with np.load(cache / f's{s}_f{f}.npz') as a:
                assert np.array_equal(a['train_ids'], ids[tr]) and np.array_equal(a['validation_ids'], ids[va])
                inner_oof = a['oof_train']
                assert np.max(abs(baseline(tr, inner_oof, va, a['valid']) - prev[va])) < 1e-08
            z = baseline(tr, inner_oof, tr, inner_oof)
            inner = saved / f's{s}_f{f}.npz'
            branch_sha = wl.ef.sha(OUT / 'local_rank_upstream_joint_oof.npz')
            if inner.exists():
                with np.load(inner) as a:
                    assert np.array_equal(a['ids'], ids[tr]) and str(a['branch_sha']) == branch_sha
                    b = a['branch']
            else:
                b = np.empty(len(tr))
                X = train_basis(raw, tr)[tr]
                for fit, hold in StratifiedKFold(3, shuffle=True, random_state=7811 + f).split(tr, y[tr]):
                    w = weights(X[fit], guide[tr[fit]], y[tr[fit]])
                    b[hold] = guide[tr[hold]] + X[hold] @ w
                np.savez_compressed(inner, ids=ids[tr].to_numpy(dtype=str), branch=b, branch_sha=branch_sha)
            new[va] = calibrated(z, b, tr, white, walk, tr, y, prev[va], side[va], va)
        pred[f's{s}'] = new
        scores[s] = {k: wl.score(y, v, wt, wn, miss) for k, v in (('previous', prev), ('new', new))}
        details[s] = audit(y, miss, prev, new)
        print(s, scores[s]['new'], details[s]['candidate'], flush=True)
    passed = all((v['new']['auc'] > v['previous']['auc'] for v in scores.values())) and all((scores[7]['new'][k] >= scores[7]['previous'][k] for k in ('remaining', 'target_vs_neg')))
    result = dict(parent=BASE, accepted=bool(passed), scores=scores, audits=details)
    (OUT / (tag + '.json')).write_text(json.dumps(result, indent=2))
    np.savez_compressed(OUT / (tag + '_oof.npz'), ids=ids.to_numpy(dtype=str), **pred)
    if passed:
        tr, te = (np.arange(n), n + np.arange(len(tids)))
        with np.load(basefolder / 'replay_inputs.npz') as a:
            assert np.array_equal(a['ids'], tids)
            pv = a['parent_logits'] + a['features'] @ spec['weights']
        dest = calibrated(ref['s7'], branch['branch_7'], tr, white, walk, tr, y, pv, branch['branch_test'], te)
        dst = ROOT / f'new weights/wn_{tag}_candidate'
        dst.mkdir(exist_ok=True)
        recipe, csv = (dst / 'weights.json', dst / 'submission.csv')
        sha = wl.ef.sha
        delta = (dest - pv)[:, None]
        sub = wl.ef.write_submission(expit(pv + delta[:, 0]), tids, csv)
        sources = [basefolder / 'weights.json', OUT / 'local_rank_upstream_joint_oof.npz', Path(__file__).resolve()]
        recipe.write_text(json.dumps(dict(weights=[1.0], upstream_weight=0.38, control_groups=['other', 'intermediate_white', 'white', 'walk'], hashes={str(p.relative_to(ROOT)): sha(p) for p in sources}), indent=2))
        np.savez_compressed(dst / 'replay_inputs.npz', ids=tids.to_numpy(dtype=str), order=tids.get_indexer(sub.id), parent_logits=pv, features=delta)
        (dst / 'validation.json').write_text(json.dumps(dict(**result, recipe_sha256=sha(recipe), submission_sha256=sha(csv)), indent=2))
        shutil.copyfile(basefolder / 'replay.py', dst / 'replay.py')
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
