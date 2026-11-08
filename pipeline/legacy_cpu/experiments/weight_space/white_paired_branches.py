# CPU port; original experiments/weight_space/white_paired_branches.py, SHA256 b78f5d451bb62a00a23b2f1b1cbc8d516992f0c924f808206a9971337a0b7d21
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Combine a high-specificity branch with the independently fitted recovery head."""
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
from white_checkpoint import SEEDS
from white_nested import audit
from white_crop_replication import BASE
from partition_evidence import OUT, ROOT
import white_loop as wl
RECOVERY = 'local_rank_concordance_tree_channels_shared'

def load_scores(name, ids):
    path = OUT / (name + '_oof.npz')
    a = np.load(path)
    assert np.array_equal(a['ids'], ids)
    return (a, path)

def test_scores(name, tids):
    d = ROOT / f'new weights/wn_{name}_candidate'
    weights = json.loads((d / 'weights.json').read_text())['weights']
    with np.load(d / 'replay_inputs.npz') as a:
        assert np.array_equal(a['ids'], tids)
        return a['parent_logits'] + a['features'] @ weights

def main():
    q, ids, tids, y, wt, wn, miss = wl.load()
    baseline, p0 = load_scores(BASE, ids)
    general, p1 = load_scores('local_rank_upstream_joint', ids)
    specialized, p2 = load_scores(RECOVERY, ids)
    tag = 'local_rank_paired_branches'
    scores, detail, pred = ({}, {}, {})
    for s in SEEDS:
        z = baseline[f's{s}']
        full = general[f's{s}']
        new = 0.5 * (full + specialized[f's{s}'])
        scores[s] = {k: wl.score(y, v, wt, wn, miss) for k, v in (('previous', z), ('new', new))}
        pred[f's{s}'] = new
        detail[s] = audit(y, miss, z, new)
        print(s, scores[s]['new'], detail[s]['candidate'], flush=True)
    passed = all((v['new']['auc'] > v['previous']['auc'] for v in scores.values())) and all((scores[7]['new'][k] >= scores[7]['previous'][k] for k in ('remaining', 'target_vs_neg')))
    report = dict(accepted=bool(passed), scores=scores, audits=detail)
    (OUT / (tag + '.json')).write_text(json.dumps(report, indent=2))
    np.savez_compressed(OUT / (tag + '_oof.npz'), ids=ids.to_numpy(dtype=str), **pred)
    if passed:
        prev = test_scores(BASE, tids)
        spec = test_scores(RECOVERY, tids)
        assert np.array_equal(general['tids'], tids)
        dst = ROOT / f'new weights/wn_{tag}_candidate'
        dst.mkdir(exist_ok=True)
        w = np.array([0.125, 0.5])
        feat = np.column_stack([general['branch_test'] - prev, spec - prev])
        recipe, csv = (dst / 'weights.json', dst / 'submission.csv')
        sub = wl.ef.write_submission(expit(prev + feat @ w), tids, csv)
        deps = [p0, p1, p2, Path(__file__).resolve(), ROOT / f'new weights/wn_{BASE}_candidate/weights.json', ROOT / f'new weights/wn_{RECOVERY}_candidate/weights.json']
        recipe.write_text(json.dumps(dict(weights=w.tolist(), hashes={str(p.relative_to(ROOT)): wl.ef.sha(p) for p in deps}), indent=2))
        np.savez_compressed(dst / 'replay_inputs.npz', ids=tids.to_numpy(dtype=str), order=tids.get_indexer(sub.id), parent_logits=prev, features=feat)
        (dst / 'validation.json').write_text(json.dumps(dict(**report, recipe_sha256=wl.ef.sha(recipe), submission_sha256=wl.ef.sha(csv)), indent=2))
        shutil.copyfile(ROOT / f'new weights/wn_{BASE}_candidate/replay.py', dst / 'replay.py')
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
