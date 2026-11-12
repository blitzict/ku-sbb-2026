# CPU port; original experiments/weight_space/white_view_spectrum.py, SHA256 c58a88c6f0a2c128d45cc3aca79ffc6200769656a1838fc8357cdb26b9abdb41
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Fit the individual genuine crop and residual-view votes jointly with the source models."""
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
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits
from white_opinion_spectrum import inputs, design
from white_freeze_first_layer import NAME
from white_checkpoint import SEEDS
from white_nested import fit, audit
from partition_evidence import ROOT, OUT
import white_loop as wl
TAGS = ('MECH3C12V_s0', 'MECH3C12V_s1', 'MECH3C12V_s2', 'MECH3C12V_k29_s0', 'MECH3C12V_k7_s0', 'MECH3C12V_10x43_s0', 'MECH3C12V_srank_s0', 'MECH3C12V_srank_k29_s0', 'MECH3C12V_srank_k7_s0', 'MECH3C12RV_s0', 'MECH3C123V_s0', 'MECH3C1234V_s0', 'MECH3C12V_srank_skew_s0')
STREAMS = ('periodic', 'clock', 'curve', 'trend', 'innov_v2', 'pulse', 'mode_residual', 'tail_channels', 'volatility_channels')
PREVIOUS = 'local_rank_opinion_linear'

def extra(ids, tids):
    archive = ROOT / 'tune_out'
    controls, views = ([], [])
    paths = []
    with np.load(archive / 'var/mat_ns3_plain.npz', allow_pickle=True) as a:
        assert np.array_equal(a['ids'], ids) and np.array_equal(a['tid'], tids)
    for tag in TAGS:
        p = archive / f'ver_{tag}.npz'
        with np.load(p) as a:
            types = a['kinds'].tolist()
            assert types[:3] == ['orig', 'c1', 'c2']
            s = np.hstack([a['oof'], a['test']]).T
        assert s.shape == (len(ids) + len(tids), len(types)) and np.isfinite(s).all()
        logits = logit(s.astype(float).clip(1e-06, 1 - 1e-06))
        controls.append(logits[:, 0])
        views.extend(logits[:, 1:].T)
        paths.append(p)
    clean = []
    for stream in STREAMS:
        parts = []
        for part, ix in (('train', ids), ('test', tids)):
            p = OUT / f'{stream}_seq_{part}.parquet'
            tab = pd.read_parquet(p).loc[ix]
            assert tab.index.is_unique and np.isfinite(tab.to_numpy()).all()
            parts.append(tab)
            paths.append(p)
        h = pd.concat(parts)
        clean.append(h.filter(regex='^(common|separate)_').to_numpy())
    return ([np.column_stack(controls), np.column_stack(views), np.column_stack(clean)], paths)

def main():
    q, ids, tids, y, wt, wn, miss = wl.load()
    n = len(y)
    packs, reg, files = inputs(ids, tids)
    more, paths = extra(ids, tids)
    cache = OUT / ('frozen_' + NAME)
    prev = np.load(OUT / (PREVIOUS + '_oof.npz'))
    parent = ROOT / f'new weights/wn_{NAME}_candidate'
    sha = wl.ef.sha(parent / 'weights.json')
    scores, ops, pred = ({}, {}, {})
    for s in SEEDS:
        new, locked = (np.empty(n), np.empty(n))
        ref = prev[f's{s}']
        for fold in range(5):
            key = f's{s}_f{fold}_'
            tr, va = (q[key + 'tr'], q[key + 'va'])
            with np.load(cache / f's{s}_f{fold}.npz') as a:
                assert str(a['parent_sha']) == sha and np.array_equal(a['validation_ids'], ids[va]) and np.array_equal(a['train_ids'], ids[tr])
                z, v = (a['oof_train'], a['valid'])
            h, _ = design(packs, reg, tr, va, z, v)
            h2, _ = design(more, reg, tr, va, z, v)
            H = np.column_stack([h, 0.65 * h2])
            w = fit(H[:len(tr)], z, y[tr], True)
            new[va] = v + 0.5 * H[len(tr):] @ w
            locked[va] = q[key + 'zv']
        scores[s] = {k: wl.score(y, val, wt, wn, miss) for k, val in (('previous', ref), ('new', new), ('locked', locked))}
        ops[s] = audit(y, miss, ref, new)
        pred[f's{s}'] = new
        print(s, [round(scores[s]['new'][k], 8) for k in ('auc', 'target_vs_neg', 'remaining')], ops[s]['candidate'], flush=True)
    passed = all((v['new']['auc'] > v['previous']['auc'] and all((v['new'][k] > v['locked'][k] for k in ('remaining', 'target_vs_neg'))) for v in scores.values()))
    tag = 'local_rank_view_spectrum'
    result = dict(accepted=bool(passed), scores=scores, audits=ops)
    (OUT / (tag + '.json')).write_text(json.dumps(result, indent=2))
    np.savez_compressed(OUT / (tag + '_oof.npz'), ids=ids.to_numpy(dtype=str), **pred)
    if passed:
        tr = np.arange(n)
        te = n + np.arange(len(tids))
        with np.load(cache / 'full.npz') as a:
            assert np.array_equal(a['test_ids'], tids)
            z, v = (a['oof_train'], a['test'])
        H = np.column_stack([design(packs, reg, tr, te, z, v)[0], 0.65 * design(more, reg, tr, te, z, v)[0]])
        w = 0.5 * fit(H[:n], z, y, True)
        F = H[n:]
        dst = ROOT / f'new weights/wn_{tag}_candidate'
        dst.mkdir(exist_ok=True)
        rec, csv = (dst / 'weights.json', dst / 'submission.csv')
        sub = wl.ef.write_submission(expit(v + F @ w), tids, csv)
        deps = [parent / 'weights.json', Path(__file__).resolve(), *files, *paths]
        rec.write_text(json.dumps(dict(weights=w.tolist(), hashes={str(p.relative_to(ROOT)): wl.ef.sha(p) for p in dict.fromkeys(deps)}), indent=2))
        np.savez_compressed(dst / 'replay_inputs.npz', ids=tids.to_numpy(dtype=str), order=tids.get_indexer(sub.id), parent_logits=v, features=F)
        (dst / 'validation.json').write_text(json.dumps(dict(**result, recipe_sha256=wl.ef.sha(rec), submission_sha256=wl.ef.sha(csv)), indent=2))
        shutil.copyfile(parent / 'replay.py', dst / 'replay.py')
if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
