# CPU port; original experiments/weight_space/white_first_family_weights.py, SHA256 ae72ffde0473a6de16df2915a25144db72938aaecff9150b196b793d4dc1698a
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Learn all family disagreements in the first fit and replace the earlier branch weights."""
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
from scipy.special import expit
from threadpoolctl import threadpool_limits
from white_opinion_spectrum import inputs, design
from white_upstream_stack import collect, train_basis
from white_paired_branches import test_scores
from white_local_rank_weights import weights
from white_checkpoint import SEEDS
from white_nested import audit
from partition_evidence import ROOT, OUT
import white_loop as wl
PARENT = 'local_rank_opinion_linear'

def branch(raw, reg, packs, tr, va, y):
    guide = raw[0]
    h = train_basis(raw, tr)
    a, _ = design(packs, reg, tr, va, guide[tr], guide[va])
    ix = np.r_[tr, va]
    H = np.column_stack([h[ix], 0.8 * a])
    w = weights(H[:len(tr)], guide[tr], y[tr])
    return (guide[va] + H[len(tr):] @ w, H[len(tr):], w)

def main():
    q, ids, tids, y, wt, wn, miss = wl.load()
    raw = collect(ids, tids)
    packs, reg, paths = inputs(ids, tids)
    original = np.load(OUT / 'local_rank_upstream_joint_oof.npz')
    current = np.load(OUT / (PARENT + '_oof.npz'))
    assert np.array_equal(ids, current['ids']) and np.array_equal(ids, original['ids'])
    n = len(y)
    arr = {}
    scores = {}
    details = {}
    for s in SEEDS:
        new = np.empty(n)
        locked = np.empty(n)
        ref = current[f's{s}']
        for fold in range(5):
            key = f's{s}_f{fold}_'
            tr, va = (q[key + 'tr'], q[key + 'va'])
            fresh, _, _ = branch(raw, reg, packs, tr, va, y)
            new[va] = ref[va] + 0.25 * (fresh - original[f'branch_{s}'][va])
            locked[va] = q[key + 'zv']
        scores[s] = {k: wl.score(y, v, wt, wn, miss) for k, v in (('previous', ref), ('new', new), ('locked', locked))}
        details[s] = audit(y, miss, ref, new)
        arr[f's{s}'] = new
        print(s, [round(scores[s]['new'][k], 8) for k in ('auc', 'target_vs_neg', 'remaining')], details[s]['candidate'], flush=True)
    passed = all((v['new']['auc'] > v['previous']['auc'] and all((v['new'][k] > v['locked'][k] for k in ('target_vs_neg', 'remaining'))) for v in scores.values()))
    tag = 'local_rank_first_family_weights'
    result = dict(accepted=bool(passed), scores=scores, audits=details)
    (OUT / (tag + '.json')).write_text(json.dumps(result, indent=2))
    tr = np.arange(n)
    te = n + np.arange(len(tids))
    fresh, F, w = branch(raw, reg, packs, tr, te, y)
    np.savez_compressed(OUT / (tag + '_oof.npz'), ids=ids.to_numpy(dtype=str), tids=tids.to_numpy(dtype=str), branch_test=fresh, **arr)
    if passed:
        base = test_scores(PARENT, tids)
        assert np.array_equal(tids, original['tids'])
        h = np.column_stack([raw[0][te] - original['branch_test'], F])
        w = 0.25 * np.r_[1.0, w]
        dst = ROOT / f'new weights/wn_{tag}_candidate'
        dst.mkdir(exist_ok=True)
        old = ROOT / f'new weights/wn_{PARENT}_candidate'
        rec, csv = (dst / 'weights.json', dst / 'submission.csv')
        sub = wl.ef.write_submission(expit(base + h @ w), tids, csv)
        deps = [old / 'weights.json', Path(__file__).resolve(), OUT / 'local_rank_upstream_joint_oof.npz', *paths, *raw[-1]]
        rec.write_text(json.dumps(dict(weights=w.tolist(), hashes={str(p.relative_to(ROOT)): wl.ef.sha(p) for p in dict.fromkeys(deps)}), indent=2))
        np.savez_compressed(dst / 'replay_inputs.npz', ids=tids.to_numpy(dtype=str), order=tids.get_indexer(sub.id), parent_logits=base, features=h)
        (dst / 'validation.json').write_text(json.dumps(dict(**result, recipe_sha256=wl.ef.sha(rec), submission_sha256=wl.ef.sha(csv)), indent=2))
        shutil.copyfile(old / 'replay.py', dst / 'replay.py')
if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
