# CPU port; original experiments/weight_space/other_response_gate.py, SHA256 3953f3e24f22afb0c680018d4e788b8c739a0d9dadd1a29e19b71aa0336b69a2
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Learn when a change in the physical model's vote should move a series farther."""
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
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from scipy.special import expit
import white_loop as wl
from partition_evidence import OUT, ROOT
from white_checkpoint import SEEDS
from white_joint import read_member
from white_local_rank_weights import weights
from white_paired_branches import test_scores
from white_shared_residual import NAME
from other_missed_breaks import CURRENT, rankings
from hard_pair_budget import cohorts

def context(ids, tids):
    paths = []
    groups = (('claude_crop_three_partitions', 'law_independent_rank_bag_29_7'), ('lawforecast', 'lawblock', 'lawretrieval', 'lawpredictive'), ('law_ordered_paths', 'law_shock_quantiles'))
    banks = []
    for names in groups:
        arr = []
        for m in names:
            s, pp = read_member(m, ids, tids)
            arr.append(s[:, 0])
            paths += pp
        banks.append(np.mean(arr, axis=0))
    p = ROOT / 'tune_out/generator_segments.parquet'
    pre = pd.read_parquet(p, columns=['id', 'part', 'phi1_ar1', 'p_bic'])
    pre = pre[pre.part == 'pre'].set_index('id').loc[ids.append(tids)]
    phi = pre.phi1_ar1.to_numpy()
    gate = ((pre.p_bic != 0) | (phi > 0.95)).to_numpy()
    phase = []
    for part, ix in (('train', ids), ('test', tids)):
        p0 = OUT / f'periodic_{part}.parquet'
        paths.append(p0)
        phase.extend(pd.read_parquet(p0, columns=['pre_fraction']).loc[ix].pre_fraction)
    return (np.column_stack(banks), gate, phi, np.array(phase), [*paths, p])

def design(ix, base, prior, anchors, gate, phi, phase, scale):
    median, spread, target_mid, target_span = scale
    aligned = (anchors[ix] - median) / spread * target_span + target_mid
    delta = np.column_stack([base - prior, base[:, None] - aligned])
    delta = np.tanh(delta / np.array([0.45, 1.0, 1.6, 1.6]))
    risk = np.tanh((base + 0.5) / 1.4)
    cond = np.column_stack([np.ones(len(ix)), risk, 0.75 * risk ** 2, 0.65 * phase[ix], 0.65 * np.tanh(2.4 * phi[ix])])
    return (delta[:, :, None] * cond[:, None, :] * gate[ix, None, None]).reshape(len(ix), -1)

def run(alternative=None, tag='local_rank_serial_response'):
    q, ids, tids, y, wt, wn, white = wl.load()
    other = cohorts(ids, y)[0] & ~white
    n = len(y)
    anchors, g, phi, phase, paths = context(ids, tids)

    def fit(tr, va, z, v, pr, pv):
        lo, med, hi = np.quantile(anchors[tr], [0.2, 0.5, 0.8], axis=0)
        a, b, c = np.quantile(z, [0.2, 0.5, 0.8])
        scale = (med, (hi - lo).clip(0.1), b, c - a)
        X = design(tr, z, pr, anchors, g, phi, phase, scale)
        V = design(va, v, pv, anchors, g, phi, phase, scale)
        w = 0.65 * weights(X, z, y[tr])
        return (v + V @ w, V, w)
    if alternative is not None:
        fit, paths = alternative(ids, tids, y)
    parent = ROOT / f'new weights/wn_{CURRENT}_candidate'
    digest = wl.ef.sha(parent / 'weights.json')
    scores, preds = ({}, {})
    for s in [*SEEDS, 61817]:
        old, new, locked = [np.empty(n) for _ in range(3)]
        for f, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=s).split(ids, y)):
            file = f's{s}_f{f}.npz'
            with np.load(OUT / ('frozen_' + CURRENT) / file) as a, np.load(OUT / ('frozen_' + NAME) / file) as b:
                assert np.array_equal(a['train_ids'], ids[tr]) and np.array_equal(a['validation_ids'], ids[va])
                assert str(a['parent_sha']) == digest and np.array_equal(a['train_ids'], b['train_ids'])
                assert np.array_equal(a['validation_ids'], b['validation_ids'])
                old[va], locked[va] = (a['valid'], a['locked'])
                new[va] = fit(tr, va, a['oof_train'], old[va], b['oof_train'], b['valid'])[0]
        row = {}
        for key, p in (('locked', locked), ('previous', old), ('new', new)):
            r = wl.score(y, p, wt, wn, white)
            rank = rankings(y, p)
            for name, mask in (('other_historical', other), ('historical_white', white)):
                r[name + '_auc'] = float(rank[mask].mean() / sum(y == 0))
                r[name + '_caught_20'] = int(sum(rank[mask] >= 0.8 * sum(y == 0)))
            row[key] = r
        scores[s] = row
        preds[f's{s}'] = new
        print(s, [round(row['new'][k], 8) for k in ('auc', 'target_vs_neg', 'other_historical_auc', 'remaining')], flush=True)
    ok = all((r['new']['auc'] > r['previous']['auc'] and all((r['new'][k] >= r['locked'][k] for k in ('target_vs_neg', 'remaining'))) for r in scores.values()))
    result = dict(accepted=ok, scores=scores)
    (OUT / (tag + '.json')).write_text(json.dumps(result, indent=2))
    np.savez_compressed(OUT / (tag + '_oof.npz'), ids=ids.to_numpy(dtype=str), **preds)
    if not ok:
        return
    pack = ROOT / f'new weights/wn_{tag}_candidate'
    pack.mkdir(exist_ok=True)
    with np.load(OUT / ('frozen_' + CURRENT) / 'full.npz') as a, np.load(OUT / ('frozen_' + NAME) / 'full.npz') as b:
        assert np.array_equal(ids, a['train_ids']) and np.array_equal(tids, a['test_ids'])
        v = a['test']
        new, V, w = fit(np.arange(n), n + np.arange(len(tids)), a['oof_train'], v, b['oof_train'], b['test'])
    sub = wl.ef.write_submission(expit(new), tids, pack / 'submission.csv')
    deps = [parent / 'weights.json', ROOT / f'new weights/wn_{NAME}_candidate/weights.json', Path(__file__).resolve(), *paths]
    rec = pack / 'weights.json'
    rec.write_text(json.dumps(dict(weights=w.tolist(), hashes={str(p.relative_to(ROOT)): wl.ef.sha(p) for p in deps}), indent=2))
    np.savez_compressed(pack / 'replay_inputs.npz', ids=tids.to_numpy(dtype=str), order=tids.get_indexer(sub.id), parent_logits=v, features=V)
    (pack / 'validation.json').write_text(json.dumps(dict(**result, recipe_sha256=wl.ef.sha(rec), submission_sha256=wl.ef.sha(pack / 'submission.csv')), indent=2))
    shutil.copyfile(parent / 'replay.py', pack / 'replay.py')
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
