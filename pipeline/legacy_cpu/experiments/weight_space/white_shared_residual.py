# Fixed winning-recipe CPU adaptation v1
"""Separate consensus from deviations of the physical and original detector families."""
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
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
import white_loop as wl
from partition_evidence import OUT, ROOT
from white_checkpoint import SEEDS
from white_joint import read_member
from white_local_rank_weights import weights
from white_nested import audit
from white_opinion_spectrum import NAME, design, inputs
REFERENCE = 'local_rank_without_multiview_trees'
MEMBERS = ('law_ordered_paths', 'law_path_control', 'law_shock_quantiles', 'law_shock_quantile_control', 'law_rank_dependence_s7', 'law_rank_dependence_s29', 'claude_masked_physics', 'claude_source_loss')

def chart(banks, reg, extra, tr, va, z, v):
    bank = list(banks)
    bank[2] = bank[2][:, :0]
    h, p = design(bank, reg, tr, va, z, v)
    n = len(tr)
    common = p[:, ::3]
    centered = common - np.mean(common[:n], axis=0)
    params = np.linalg.solve(centered[:n].T @ centered[:n] + np.eye(common.shape[1]) * n * 0.08, centered[:n].T @ h[:n])
    residual = h - centered @ params
    base = np.column_stack([common, residual])
    ix = np.r_[tr, va]
    d = np.arcsinh((extra[ix, ::2] - extra[ix, 1::2]).clip(-12, 12))
    d -= d[:n].mean(0)
    d /= np.sqrt(np.mean(d[:n] ** 2, axis=0) + 0.14)
    shared = np.column_stack([np.ones(len(h)), centered])
    neutral = d - shared @ np.linalg.solve(shared[:n].T @ shared[:n] + np.eye(shared.shape[1]) * n * 0.04, shared[:n].T @ d[:n])
    g = reg[ix][:, [0, 3, 4, 5]]
    effect = np.column_stack([neutral, 0.55 * (neutral[:, :, None] * (g[:, None, :] - 0.45)).reshape(len(h), -1)])
    return {'agreement': base, 'response': np.column_stack([base, effect])}

def run():
    q, ids, tids, y, wt, wn, miss = wl.load()
    n = len(y)
    bank, reg, files = inputs(ids, tids)
    cache = OUT / ('frozen_' + NAME)
    members = []
    for name in MEMBERS:
        a, paths = read_member(name, ids, tids)
        members.append(a[:, 0])
        files.extend(paths)
    extra = np.column_stack(members)
    scores, audits, preds = ({}, {}, {})
    for seed in [*SEEDS, 61817]:
        old, locked = (np.empty(n), np.empty(n))
        result = {}
        for f, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(ids, y)):
            with np.load(cache / f's{seed}_f{f}.npz') as a:
                assert np.array_equal(a['validation_ids'], ids[va]) and np.array_equal(a['train_ids'], ids[tr])
                z, v = (a['oof_train'], a['valid'])
                locked[va] = q[f's{seed}_f{f}_zv'] if seed in SEEDS else a['locked']
            active = list(bank)
            active[2] = active[2][:, :0]
            baseline = design(active, reg, tr, va, z, v)[0]
            old[va] = v + baseline[len(tr):] @ (0.5 * weights(baseline[:len(tr)], z, y[tr]))
            for mode, h in chart(bank, reg, extra, tr, va, z, v).items():
                result.setdefault(mode, np.empty(n))[va] = v + h[len(tr):] @ (0.5 * weights(h[:len(tr)], z, y[tr]))
        if seed in SEEDS:
            with np.load(OUT / (REFERENCE + '_oof.npz')) as a:
                np.testing.assert_allclose(old, a[f's{seed}'], atol=2e-07)
        for mode, new in result.items():
            r = {key: wl.score(y, value, wt, wn, miss) for key, value in (('locked', locked), ('previous', old), ('new', new))}
            scores.setdefault(mode, {})[seed] = r
            audits.setdefault(mode, {})[seed] = audit(y, miss, old, new)
            preds.setdefault(mode, {})[f's{seed}'] = new
            print(mode, seed, [round(r['new'][k], 8) for k in ('auc', 'target_vs_neg', 'remaining')], flush=True)
    for mode, r in scores.items():
        ok = all((x['new']['auc'] > x['previous']['auc'] and all((x['new'][k] >= x['locked'][k] for k in ('remaining', 'target_vs_neg'))) for x in r.values()))
        tag = 'local_rank_shared_residual_' + mode
        report = _rebuild.annotate(dict(accepted=ok, reference=REFERENCE, scores=r, audits=audits[mode]))
        (OUT / (tag + '.json')).write_text(json.dumps(report, indent=2))
        np.savez_compressed(OUT / (tag + '_oof.npz'), ids=ids.to_numpy(dtype=str), **preds[mode])
        if not _rebuild.export(ok):
            continue
        parent = ROOT / f'new weights/wn_{NAME}_candidate'
        with np.load(cache / 'full.npz') as a:
            assert str(a['parent_sha']) == wl.ef.sha(parent / 'weights.json') and np.array_equal(tids, a['test_ids'])
            z, v = (a['oof_train'], a['test'])
        h = chart(bank, reg, extra, np.arange(n), n + np.arange(len(tids)), z, v)[mode]
        w = 0.5 * weights(h[:n], z, y)
        dst = ROOT / f'new weights/wn_{tag}_candidate'
        dst.mkdir(exist_ok=True)
        sub = wl.ef.write_submission(expit(v + h[n:] @ w), tids, dst / 'submission.csv')
        deps = [Path(__file__).resolve(), parent / 'weights.json', *files]
        rec = dst / 'weights.json'
        rec.write_text(json.dumps(dict(weights=w.tolist(), mode=mode, hashes={str(p.relative_to(ROOT)): wl.ef.sha(p) for p in dict.fromkeys(deps)}), indent=2))
        np.savez_compressed(dst / 'replay_inputs.npz', ids=tids.to_numpy(dtype=str), order=tids.get_indexer(sub.id), parent_logits=v, features=h[n:])
        (dst / 'validation.json').write_text(json.dumps(dict(**report, recipe_sha256=wl.ef.sha(rec), submission_sha256=wl.ef.sha(dst / 'submission.csv')), indent=2))
        shutil.copyfile(parent / 'replay.py', dst / 'replay.py')
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
