# CPU port; original experiments/weight_space/other_dynamics_weights.py, SHA256 767f273617e7e48b4854bbd2d01da2a35acfe701979f76c9917b2867b2e06cd9
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Test fixed changes to the weight edits on serially correlated processes."""
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
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
import white_loop as wl
from partition_evidence import OUT, ROOT
from white_checkpoint import SEEDS
from white_nested import audit
from white_shared_residual import NAME
from white_paired_branches import test_scores
from other_missed_breaks import CURRENT, rankings
from hard_pair_budget import cohorts

def adjustments(z, prior, locked, g):
    return {'full_final': (z - prior)[:, None], 'serial_final': (g * (z - prior))[:, None], 'serial_physics': (0.25 * g * (z - locked))[:, None]}

def main():
    q, ids, tids, y, wt, wn, miss = wl.load()
    other = cohorts(ids, y)[0] & ~miss
    source = ROOT / 'tune_out/generator_segments.parquet'
    gen = pd.read_parquet(source, columns=['id', 'part', 'p_bic', 'phi1_ar1'])
    pre = gen[gen.part == 'pre'].set_index('id').loc[ids.append(tids)]
    gate = ((pre.p_bic != 0) | (pre.phi1_ar1 > 0.95)).to_numpy().astype(float)
    assert np.array_equal(wn, gate[:len(y)] == 0)
    parent = ROOT / f'new weights/wn_{CURRENT}_candidate'
    scores, audits, values = ({}, {}, {})
    with np.load(OUT / (CURRENT + '_oof.npz')) as data:
        assert np.array_equal(data['ids'], ids)
        for s in [*SEEDS, 61817]:
            latest = data[f's{s}']
            previous, locked = (np.empty(len(y)), np.empty(len(y)))
            for f, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=s).split(ids, y)):
                with np.load(OUT / f'frozen_{NAME}/s{s}_f{f}.npz') as a:
                    assert np.array_equal(ids[va], a['validation_ids'])
                    previous[va] = a['valid']
                    locked[va] = q[f's{s}_f{f}_zv'] if s in SEEDS else a['locked']
            for rule, h in adjustments(latest, previous, locked, gate[:len(y)]).items():
                new = latest + h[:, 0]
                record = {}
                for k, p in (('locked', locked), ('previous', latest), ('new', new)):
                    r = wl.score(y, p, wt, wn, miss)
                    rank = rankings(y, p)
                    r.update(other_historical_auc=float(rank[other].mean() / sum(y == 0)), other_historical_caught_20=int(sum(rank[other] >= 0.8 * sum(y == 0))))
                    record[k] = r
                scores.setdefault(rule, {})[s] = record
                audits.setdefault(rule, {})[s] = audit(y, miss, latest, new)
                values.setdefault(rule, {})[f's{s}'] = new
                print(rule, s, [round(record['new'][k], 8) for k in ('auc', 'target_vs_neg', 'other_historical_auc', 'remaining')], flush=True)
    for rule, records in scores.items():
        ok = all((r['new']['auc'] > r['previous']['auc'] and all((r['new'][k] >= r['locked'][k] for k in ('target_vs_neg', 'remaining'))) for r in records.values()))
        tag = 'local_rank_dynamics_' + rule
        result = dict(accepted=ok, parent=CURRENT, rule=rule, scores=records, audits=audits[rule])
        (OUT / (tag + '.json')).write_text(json.dumps(result, indent=2))
        np.savez_compressed(OUT / (tag + '_oof.npz'), ids=ids.to_numpy(dtype=str), **values[rule])
        if not ok:
            continue
        dst = ROOT / f'new weights/wn_{tag}_candidate'
        dst.mkdir(exist_ok=True)
        z = test_scores(CURRENT, tids)
        h = adjustments(z, test_scores(NAME, tids), q['zt'], gate[len(y):])[rule]
        sub = wl.ef.write_submission(expit(z + h[:, 0]), tids, dst / 'submission.csv')
        deps = [parent / 'weights.json', wl.PARENT, ROOT / f'new weights/wn_{NAME}_candidate/weights.json', source, Path(__file__).resolve()]
        rec = dst / 'weights.json'
        rec.write_text(json.dumps(dict(weights=[1.0], rule=rule, hashes={str(p.relative_to(ROOT)): wl.ef.sha(p) for p in deps}), indent=2))
        np.savez_compressed(dst / 'replay_inputs.npz', ids=tids.to_numpy(dtype=str), order=tids.get_indexer(sub.id), parent_logits=z, features=h)
        (dst / 'validation.json').write_text(json.dumps(dict(**result, recipe_sha256=wl.ef.sha(rec), submission_sha256=wl.ef.sha(dst / 'submission.csv')), indent=2))
        shutil.copyfile(parent / 'replay.py', dst / 'replay.py')
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
