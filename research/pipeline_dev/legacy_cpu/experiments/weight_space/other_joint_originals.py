# CPU port; original experiments/weight_space/other_joint_originals.py, SHA256 f3b824b8a9e6f368e83395795a7ea6d86f79ce8586af757e0da5d4b9ad19b570
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Fit the original feature trees jointly with the newer physical detector families."""
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
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from hard_pair_budget import cohorts
from other_archived_votes import load
from other_missed_breaks import CURRENT, rankings
from partition_evidence import OUT, ROOT
from white_checkpoint import SEEDS
from white_joint import read_member
from white_local_rank_weights import weights
from white_nested import audit
from white_opinion_spectrum import inputs, NAME
from white_shared_residual import chart, MEMBERS
import white_loop as wl

def main():
    q, ids, tids, y, wt, wn, white = wl.load()
    n = len(y)
    other = cohorts(ids, y)[0] & ~white
    bank, reg, files = inputs(ids, tids)
    classic, paths = load(ids, tids)
    bank = [*bank, classic]
    files += paths
    additional = []
    for name in MEMBERS:
        val, pp = read_member(name, ids, tids)
        additional.append(val[:, 0])
        files += pp
    diff = np.column_stack(additional)

    def step(tr, va, z, v):
        h = chart(bank, reg, diff, tr, va, z, v)['response']
        w = 0.5 * weights(h[:len(tr)], z, y[tr])
        return (v + h[len(tr):] @ w, h[len(tr):], w)
    records, preds, audits = ({}, {}, {})
    with np.load(OUT / (CURRENT + '_oof.npz')) as old:
        assert np.array_equal(old['ids'], ids)
        for s in [*SEEDS, 61817]:
            before = old[f's{s}']
            new, locked = (np.empty(n), np.empty(n))
            for f, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=s).split(ids, y)):
                with np.load(OUT / f'frozen_{NAME}/s{s}_f{f}.npz') as a:
                    assert np.array_equal(a['train_ids'], ids[tr]) and np.array_equal(a['validation_ids'], ids[va])
                    locked[va] = q[f's{s}_f{f}_zv'] if s in SEEDS else a['locked']
                    new[va] = step(tr, va, a['oof_train'], a['valid'])[0]
            row = {}
            for label, p in (('locked', locked), ('previous', before), ('new', new)):
                r = wl.score(y, p, wt, wn, white)
                rank = rankings(y, p)
                r['other_historical_auc'] = float(rank[other].mean() / sum(y == 0))
                r['other_historical_caught_20'] = int(sum(rank[other] >= 0.8 * sum(y == 0)))
                row[label] = r
            records[s] = row
            preds[f's{s}'] = new
            audits[s] = audit(y, white, before, new)
            print(s, [round(row['new'][k], 8) for k in ('auc', 'target_vs_neg', 'other_historical_auc', 'remaining')], flush=True)
    ok = all((r['new']['auc'] > r['previous']['auc'] and all((r['new'][k] >= r['locked'][k] for k in ('target_vs_neg', 'remaining'))) for r in records.values()))
    tag = 'local_rank_joint_originals'
    result = dict(accepted=ok, scores=records, audits=audits)
    (OUT / (tag + '.json')).write_text(json.dumps(result, indent=2))
    np.savez_compressed(OUT / (tag + '_oof.npz'), ids=ids.to_numpy(dtype=str), **preds)
    if not ok:
        return
    parent = ROOT / f'new weights/wn_{NAME}_candidate'
    pack = ROOT / f'new weights/wn_{tag}_candidate'
    pack.mkdir(exist_ok=True)
    with np.load(OUT / f'frozen_{NAME}/full.npz') as a:
        assert np.array_equal(a['train_ids'], ids) and np.array_equal(a['test_ids'], tids)
        v = a['test']
        new, h, w = step(np.arange(n), n + np.arange(len(tids)), a['oof_train'], v)
    sub = wl.ef.write_submission(expit(new), tids, pack / 'submission.csv')
    deps = [parent / 'weights.json', Path(__file__).resolve(), Path(__file__).with_name('white_shared_residual.py'), Path(__file__).with_name('other_archived_votes.py'), *files]
    rec = pack / 'weights.json'
    rec.write_text(json.dumps(dict(weights=w.tolist(), hashes={str(p.relative_to(ROOT)): wl.ef.sha(p) for p in deps}), indent=2))
    np.savez_compressed(pack / 'replay_inputs.npz', ids=tids.to_numpy(dtype=str), order=tids.get_indexer(sub.id), parent_logits=v, features=h)
    (pack / 'validation.json').write_text(json.dumps(dict(**result, recipe_sha256=wl.ef.sha(rec), submission_sha256=wl.ef.sha(pack / 'submission.csv')), indent=2))
    shutil.copyfile(parent / 'replay.py', pack / 'replay.py')
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
