# CPU port; original experiments/weight_space/white_score_scale.py, SHA256 eaf7f2ef6c7d41de74a8dc15546d63b72a2d9dcac3bced35c3742208ed396917
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Test whether detector rankings agree after aligning their full score distributions."""
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
from white_local_rank_weights import weights
from white_nested import audit
from white_opinion_spectrum import NAME, design, inputs
from partition_evidence import ROOT, OUT
import white_loop as wl
REFERENCE = 'local_rank_family_near_pairs'
GRID = np.r_[0.001, 0.004, 0.01, 0.025, np.arange(0.05, 0.951, 0.05), 0.975, 0.99, 0.996, 0.999]

def align(packs, tr, z):
    target = np.quantile(z, GRID)
    result = []
    for panel in packs:
        knots = np.quantile(panel[tr], GRID, axis=0).T
        calibrated = []
        for v, axis in zip(panel.T, knots):
            axis, group = np.unique(axis, return_inverse=True)
            pred = np.bincount(group, weights=target) / np.bincount(group)
            smooth = np.interp(v, axis, pred)
            a, b, c = np.quantile(v[tr], [0.1, 0.5, 0.9])
            l, m, r = np.quantile(z, [0.1, 0.5, 0.9])
            calibrated.append(0.55 * smooth + 0.45 * ((v - b) * (r - l) / max(c - a, 0.3) + m))
        result.append(np.column_stack(calibrated))
    return result

def features(packs, reg, tr, va, z, v):
    h, p = design(packs, reg, tr, va, z, v)
    n = len(tr)
    xcal, _ = design(align(packs, tr, z), reg, tr, va, z, v)
    score = np.r_[z, v]
    windows = np.exp(-0.5 * ((score[:, None] - np.array([-1.65, -0.25, 1.25])) / 1.1) ** 2)
    windows -= windows[:n].mean(0)
    regional = (p[:, :, None] * windows[:, None, :]).reshape(len(h), -1)
    return {'quantile': xcal, 'risk_windows': np.column_stack([h, 0.6 * regional])}

def main(make=features, prefix='local_rank_score_', source=None):
    q, ids, tids, y, wt, wn, miss = wl.load()
    n = len(y)
    packs, reg, files = inputs(ids, tids)
    parent = ROOT / f'new weights/wn_{NAME}_candidate'
    sha = wl.ef.sha(parent / 'weights.json')
    cache = OUT / ('frozen_' + NAME)
    scores = {}
    detail = {}
    arrays = {}
    with np.load(OUT / (REFERENCE + '_oof.npz')) as prev:
        assert np.array_equal(prev['ids'], ids)
        for s in SEEDS:
            preds = {}
            locked = np.empty(n)
            reference = prev[f's{s}']
            for f in range(5):
                k = f's{s}_f{f}_'
                tr, va = (q[k + 'tr'], q[k + 'va'])
                with np.load(cache / f's{s}_f{f}.npz') as a:
                    assert str(a['parent_sha']) == sha and np.array_equal(ids[tr], a['train_ids']) and np.array_equal(ids[va], a['validation_ids'])
                    z, v = (a['oof_train'], a['valid'])
                for tag, h in make(packs, reg, tr, va, z, v).items():
                    if tag not in preds:
                        preds[tag] = np.empty(n)
                    w = 0.5 * weights(h[:len(tr)], z, y[tr])
                    preds[tag][va] = v + h[len(tr):] @ w
                locked[va] = q[k + 'zv']
            for tag, p in preds.items():
                record = {k: wl.score(y, value, wt, wn, miss) for k, value in (('locked', locked), ('previous', reference), ('new', p))}
                scores.setdefault(tag, {})[s] = record
                detail.setdefault(tag, {})[s] = audit(y, miss, reference, p)
                arrays.setdefault(tag, {})[f's{s}'] = p
                print(tag, s, [round(record['new'][c], 8) for c in ('auc', 'target_vs_neg', 'remaining')], flush=True)
    for method, val in scores.items():
        ok = all((r['new']['auc'] > r['previous']['auc'] and all((r['new'][k] > r['locked'][k] for k in ('target_vs_neg', 'remaining'))) for r in val.values()))
        tag = prefix + method
        result = dict(accepted=bool(ok), scores=val, audits=detail[method])
        (OUT / (tag + '.json')).write_text(json.dumps(result, indent=2))
        np.savez_compressed(OUT / (tag + '_oof.npz'), ids=ids.to_numpy(dtype=str), **arrays[method])
        if not ok:
            continue
        tr = np.arange(n)
        te = n + np.arange(len(tids))
        with np.load(cache / 'full.npz') as a:
            assert np.array_equal(tids, a['test_ids']) and str(a['parent_sha']) == sha
            z, v = (a['oof_train'], a['test'])
        h = make(packs, reg, tr, te, z, v)[method]
        w = 0.5 * weights(h[:n], z, y)
        F = h[n:]
        dst = ROOT / f'new weights/wn_{tag}_candidate'
        dst.mkdir(exist_ok=True)
        csv = dst / 'submission.csv'
        rec = dst / 'weights.json'
        sub = wl.ef.write_submission(expit(v + F @ w), tids, csv)
        deps = [Path(__file__).resolve(), parent / 'weights.json', *files]
        if source is not None:
            deps.append(source)
        rec.write_text(json.dumps(dict(weights=w.tolist(), method=method, hashes={str(p.relative_to(ROOT)): wl.ef.sha(p) for p in dict.fromkeys(deps)}), indent=2))
        np.savez_compressed(dst / 'replay_inputs.npz', ids=tids.to_numpy(dtype=str), order=tids.get_indexer(sub.id), parent_logits=v, features=F)
        (dst / 'validation.json').write_text(json.dumps(dict(**result, recipe_sha256=wl.ef.sha(rec), submission_sha256=wl.ef.sha(csv)), indent=2))
        shutil.copyfile(parent / 'replay.py', dst / 'replay.py')
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
