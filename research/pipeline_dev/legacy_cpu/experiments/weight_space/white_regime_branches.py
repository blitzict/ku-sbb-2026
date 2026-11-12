# Fixed winning-recipe CPU adaptation v1
"""Replay the first-layer fit; route ambiguous prewhite cases to the recovery weights."""
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
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits
from white_crop_replication import BASE
from white_paired_branches import RECOVERY, load_scores, test_scores
from white_upstream_stack import collect, train_basis
from white_local_rank_weights import weights
from white_checkpoint import SEEDS
from white_nested import audit
from partition_evidence import ROOT, OUT
import white_loop as wl

def route(white, base):
    return white * np.exp(-0.5 * ((base + 1.15) / 1.15) ** 2)

def main():
    q, ids, tids, y, wt, wn, miss = wl.load()
    old, oldpath = load_scores(BASE, ids)
    upstream, uppath = load_scores('local_rank_upstream_joint', ids)
    spec, specpath = load_scores(RECOVERY, ids)
    raw = collect(ids, tids)
    guide = raw[0]
    n = len(y)
    tr, te = (np.arange(n), n + np.arange(len(tids)))
    X = train_basis(raw, tr)
    coeff = weights(X[tr], guide[tr], y)
    branch = guide[te] + X[te] @ coeff
    assert np.array_equal(upstream['tids'], tids) and np.max(abs(branch - upstream['branch_test'])) < 1e-07
    base_test = test_scores(BASE, tids)
    spec_test = test_scores(RECOVERY, tids)
    paths = [oldpath, uppath, specpath, ROOT / f'new weights/wn_{BASE}_candidate/weights.json', ROOT / f'new weights/wn_{RECOVERY}_candidate/weights.json', *raw[-1], Path(__file__).resolve()]
    white = raw[3][:, 0]
    for tag, conditional in (('first_layer_overall', False), ('regime_branches', True)):
        scores, checks, pred = ({}, {}, {})
        for s in SEEDS:
            k = f's{s}'
            ref = old[k]
            z = upstream[k]
            assert np.allclose(0.75 * ref + 0.25 * upstream[f'branch_{s}'], z, atol=1e-10, rtol=0)
            if conditional:
                z = z + route(white[:n], ref) * (spec[k] - z)
            locked = np.empty(n)
            for f in range(5):
                key = f's{s}_f{f}_'
                locked[q[key + 'va']] = q[key + 'zv']
            scores[s] = {name: wl.score(y, v, wt, wn, miss) for name, v in (('locked', locked), ('previous', ref), ('new', z))}
            checks[s] = audit(y, miss, locked, z)
            pred[k] = z
            print(tag, s, round(scores[s]['new']['auc'], 8), round(scores[s]['new']['target_vs_neg'], 6), round(scores[s]['new']['remaining'], 6), checks[s]['candidate']['fpr_20'], flush=True)
        pass_locked = all((t['new'][key] > t['locked'][key] for t in scores.values() for key in ('auc', 'remaining', 'target_vs_neg')))
        pass_overall = all((t['new']['auc'] > t['previous']['auc'] for t in scores.values()))
        result = _rebuild.annotate(dict(passes_locked=pass_locked, improves_parent_overall=pass_overall, scores=scores, audits=checks))
        name = 'local_rank_' + tag
        (OUT / (name + '.json')).write_text(json.dumps(result, indent=2))
        np.savez_compressed(OUT / (name + '_oof.npz'), ids=ids.to_numpy(dtype=str), **pred)
        if not (pass_locked and pass_overall):
            continue
        dst = ROOT / f'new weights/wn_{name}_candidate'
        dst.mkdir(exist_ok=True)
        F = np.column_stack([guide[te] - base_test, X[te]])
        w = 0.25 * np.r_[1.0, coeff]
        if conditional:
            g = route(white[te], base_test)
            F = np.column_stack([(1 - g[:, None]) * F, g * (spec_test - base_test)])
            w = np.r_[w, 1.0]
        rec, csv = (dst / 'weights.json', dst / 'submission.csv')
        sub = wl.ef.write_submission(expit(base_test + F @ w), tids, csv)
        rec.write_text(json.dumps(dict(weights=w.tolist(), conditional=conditional, hashes={str(p.relative_to(ROOT)): wl.ef.sha(p) for p in dict.fromkeys(paths)}), indent=2))
        np.savez_compressed(dst / 'replay_inputs.npz', ids=tids.to_numpy(dtype=str), order=tids.get_indexer(sub.id), parent_logits=base_test, features=F)
        (dst / 'validation.json').write_text(json.dumps(dict(**result, recipe_sha256=wl.ef.sha(rec), submission_sha256=wl.ef.sha(csv)), indent=2))
        shutil.copyfile(ROOT / f'new weights/wn_{BASE}_candidate/replay.py', dst / 'replay.py')
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
