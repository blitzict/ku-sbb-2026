# CPU port; original experiments/weight_space/white_family_objectives.py, SHA256 d13580b7852c0ab0dbd8b7a6d1bec28d83dc120993f54f42e46d9482d2256ded
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Compare probability fitting to pairwise ranking on the same family-disagreement coordinates."""
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
from white_concordance import train as concordance
from white_local_rank_weights import weights
from white_freeze_opinion import TAG
from white_opinion_spectrum import NAME, inputs, design
from white_nested import audit
from partition_evidence import ROOT, OUT
import white_loop as wl

def run():
    q, ids, tids, y, wt, wn, miss = wl.load()
    packs, reg, files = inputs(ids, tids)
    n = len(ids)
    cache = OUT / ('frozen_' + NAME)
    modes = ('near_pairs', 'all_pairs')
    scores = {k: {} for k in modes}
    details = {k: {} for k in modes}
    values = {k: {} for k in modes}

    def trial(k, h, z, tr):
        if k == 'near_pairs':
            return 0.5 * weights(h, z, y[tr])
        return 0.5 * concordance(h, z, y[tr], reg[tr, 0], 0.0)
    prior = ROOT / f'new weights/wn_{NAME}_candidate/weights.json'
    sha = wl.ef.sha(prior)
    with np.load(OUT / (TAG + '_oof.npz')) as incumbent:
        assert np.array_equal(incumbent['ids'], ids)
        for s in SEEDS:
            ref = incumbent[f's{s}']
            old = np.empty(n)
            pred = {k: np.empty(n) for k in modes}
            for f in range(5):
                key = f's{s}_f{f}_'
                tr, va = (q[key + 'tr'], q[key + 'va'])
                with np.load(cache / f's{s}_f{f}.npz') as a:
                    assert str(a['parent_sha']) == sha and np.array_equal(ids[va], a['validation_ids']) and np.array_equal(ids[tr], a['train_ids'])
                    z, v = (a['oof_train'], a['valid'])
                h, _ = design(packs, reg, tr, va, z, v)
                for k in modes:
                    pred[k][va] = v + h[len(tr):] @ trial(k, h[:len(tr)], z, tr)
                old[va] = q[key + 'zv']
            for k in modes:
                scores[k][s] = {name: wl.score(y, p, wt, wn, miss) for name, p in (('locked', old), ('previous', ref), ('new', pred[k]))}
                details[k][s] = audit(y, miss, ref, pred[k])
                values[k][f's{s}'] = pred[k]
                print(s, k, [round(scores[k][s]['new'][i], 8) for i in ('auc', 'target_vs_neg', 'remaining')], flush=True)
    for k, report in scores.items():
        tag = 'local_rank_family_' + k
        ok = all((r['new']['auc'] > r['previous']['auc'] and all((r['new'][c] > r['locked'][c] for c in ('remaining', 'target_vs_neg'))) for r in report.values()))
        result = dict(accepted=bool(ok), scores=report, audits=details[k])
        (OUT / (tag + '.json')).write_text(json.dumps(result, indent=2))
        np.savez_compressed(OUT / (tag + '_oof.npz'), ids=ids.to_numpy(dtype=str), **values[k])
        if not ok:
            continue
        tr = np.arange(n)
        te = n + np.arange(len(tids))
        with np.load(cache / 'full.npz') as a:
            assert np.array_equal(tids, a['test_ids']) and str(a['parent_sha']) == sha
            z, v = (a['oof_train'], a['test'])
        h, _ = design(packs, reg, tr, te, z, v)
        w = trial(k, h[:n], z, tr)
        F = h[n:]
        dst = ROOT / f'new weights/wn_{tag}_candidate'
        dst.mkdir(exist_ok=True)
        csv = dst / 'submission.csv'
        rec = dst / 'weights.json'
        sub = wl.ef.write_submission(expit(v + F @ w), tids, csv)
        paths = [Path(__file__).resolve(), prior, *files]
        rec.write_text(json.dumps(dict(weights=w.tolist(), objective=k, hashes={str(p.relative_to(ROOT)): wl.ef.sha(p) for p in dict.fromkeys(paths)}), indent=2))
        np.savez_compressed(dst / 'replay_inputs.npz', ids=tids.to_numpy(dtype=str), order=tids.get_indexer(sub.id), parent_logits=v, features=F)
        (dst / 'validation.json').write_text(json.dumps(dict(**result, recipe_sha256=wl.ef.sha(rec), submission_sha256=wl.ef.sha(csv)), indent=2))
        shutil.copyfile(prior.parent / 'replay.py', dst / 'replay.py')
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
