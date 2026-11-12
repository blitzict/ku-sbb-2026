# CPU port; original experiments/weight_space/white_first_physics.py, SHA256 3d2a9f53fcf8f47057caaa0013b4086986bbb4e4bc7f47ec3f4acbbbdd5f3c6d
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Place fold-local physical likelihoods and forest contributions in the initial ensemble fit."""
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
from white_checkpoint import SEEDS
from white_joint import read_member
from white_stage import new_inputs
from white_consult import compile_inputs
from white_outer_priors import load, fit_on
from white_nested import fit, audit
from white_upstream_stack import collect, train_basis
from white_crop_replication import BASE
from partition_evidence import ROOT, OUT
import white_loop as wl

def run():
    q, ids, tids, y, wt, wn, miss = wl.load()
    n = len(y)
    raw = collect(ids, tids)
    seed_guide = raw[0]
    names = ['claude_crop_three_partitions', 'claude_joint_three_splits', 'law_linked_plus_rank']
    experts = []
    for name in names:
        v, _ = read_member(name, ids, tids)
        experts.append(v[:, 0])
    guide = 0.35 * seed_guide + 0.65 * np.mean(experts, axis=0)
    anchor = ROOT / 'new weights/wn_stage_reversible_rank_candidate/weights.json'
    stage = json.loads(anchor.read_text())
    physical, paths = compile_inputs(ids, tids, stage['models'] + ['claude_joint_three_splits', 'law_linked_plus_rank'], ['volatility'])
    additional, p = new_inputs(stage['tables'], ids, tids)
    paths += p
    sh = []
    for part, ix in (('train', ids), ('test', tids)):
        path = OUT / f'tree_channels_named_{part}.parquet'
        paths.append(path)
        sh.append(pd.read_parquet(path).loc[ix].filter(regex='^(full|late)_').to_numpy())
    sh = np.tanh(np.vstack(sh) / 0.45)
    ref = np.load(OUT / (BASE + '_oof.npz'))
    assert np.array_equal(ids, ref['ids'])
    scores, details, arrays = ({}, {}, {})

    def design(prior, ix, basis):
        z = guide[ix]
        gate = raw[3][ix, 0, None]
        channels = sh[ix]
        return np.column_stack([basis[ix], 0.57 * physical(prior, ix, z, 'full'), 0.54 * additional[ix], 0.75 * channels, 0.58 * gate * channels])
    for s in SEEDS:
        fresh = np.empty(n)
        for f in range(5):
            k = f's{s}_f{f}_'
            tr, va = (q[k + 'tr'], q[k + 'va'])
            prior = OUT / f'law_outer{s}'
            with np.load(prior / f'fold{f}_rows.npz') as a:
                assert np.array_equal(a['train_ids'], ids[tr]) and np.array_equal(a['validation_ids'], ids[va])
            h = pd.read_parquet(prior / f'fold{f}_train.parquet').loc[ids].to_numpy()
            basis = train_basis(raw, tr)
            x = design(h[tr], tr, basis)
            v = design(h[va], va, basis)
            w = fit(x, guide[tr], y[tr], True)
            fresh[va] = guide[va] + v @ w
        old = ref[f's{s}']
        pred = old + 0.3 * (fresh - old)
        arrays[f's{s}'] = pred
        arrays[f'branch_{s}'] = fresh
        scores[s] = {k: wl.score(y, z, wt, wn, miss) for k, z in (('previous', old), ('branch', fresh), ('new', pred))}
        details[s] = audit(y, miss, old, pred)
        print(s, [(k, round(t['auc'], 7), round(t['target_vs_neg'], 7), round(t['remaining'], 7)) for k, t in scores[s].items()], flush=True)
    tag = 'local_rank_first_physics'
    passed = all((v['new']['auc'] > v['previous']['auc'] for v in scores.values()))
    for s in SEEDS:
        locked = np.empty(n)
        for f in range(5):
            k = f's{s}_f{f}_'
            locked[q[k + 'va']] = q[k + 'zv']
        a = wl.score(y, locked, wt, wn, miss)
        passed = passed and all((scores[s]['new'][k] > a[k] for k in ('target_vs_neg', 'remaining')))
    rep = dict(parent=BASE, accepted=bool(passed), scores=scores, audits=details)
    (OUT / (tag + '.json')).write_text(json.dumps(rep, indent=2))
    fi, ft, M, mask, labels = load('physical')
    assert ids.equals(fi) and tids.equals(ft) and np.array_equal(y, labels)
    tr = np.arange(n)
    te = n + np.arange(len(tids))
    h, params = fit_on(M, mask, y, tr)
    basis = train_basis(raw, tr)
    w = fit(design(h[tr], tr, basis), guide[tr], y, True)
    test = design(h[te], te, basis)
    new = guide[te] + test @ w
    np.savez_compressed(OUT / (tag + '_oof.npz'), ids=ids.to_numpy(dtype=str), tids=tids.to_numpy(dtype=str), branch_test=new, **arrays)
    if passed:
        parent = ROOT / f'new weights/wn_{BASE}_candidate'
        rec = json.loads((parent / 'weights.json').read_text())
        with np.load(parent / 'replay_inputs.npz') as a:
            assert np.array_equal(a['ids'], tids)
            b = a['parent_logits'] + a['features'] @ rec['weights']
        dst = ROOT / f'new weights/wn_{tag}_candidate'
        dst.mkdir(exist_ok=True)
        recipe, csv = (dst / 'weights.json', dst / 'submission.csv')
        W = np.r_[0.3, 0.3 * w]
        F = np.column_stack([guide[te] - b, test])
        sub = wl.ef.write_submission(expit(b + F @ W), tids, csv)
        deps = [parent / 'weights.json', anchor, *paths, *raw[-1], Path(__file__).resolve()]
        recipe.write_text(json.dumps(dict(weights=W.tolist(), priors=params, guide=names, hashes={str(p.relative_to(ROOT)): wl.ef.sha(p) for p in dict.fromkeys(deps)}), indent=2))
        np.savez_compressed(dst / 'replay_inputs.npz', ids=tids.to_numpy(dtype=str), order=tids.get_indexer(sub.id), parent_logits=b, features=F)
        (dst / 'validation.json').write_text(json.dumps(dict(**rep, recipe_sha256=wl.ef.sha(recipe), submission_sha256=wl.ef.sha(csv)), indent=2))
        shutil.copyfile(parent / 'replay.py', dst / 'replay.py')
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
