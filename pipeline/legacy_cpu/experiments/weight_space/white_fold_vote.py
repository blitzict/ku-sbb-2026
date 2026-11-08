# Fixed winning-recipe CPU adaptation v1
"""Let separately trained crop and rank experts vote on the final score."""
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
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from white_checkpoint import SEEDS
from white_joint import read_member
from white_local_rank_weights import centered, weights
from white_nested import fit, audit
from partition_evidence import ROOT, OUT
import white_loop as wl
PARENT = 'local_rank_early_mode_peaks'
EARLIER = 'local_rank_early_opinions'

def freeze(q, ids, tids, y):
    dst = OUT / ('frozen_' + PARENT)
    dst.mkdir(exist_ok=True)
    pack = ROOT / f'new weights/wn_{PARENT}_candidate'
    earlier = OUT / ('frozen_' + EARLIER)
    digest = wl.ef.sha(pack / 'weights.json')
    spec = json.loads((pack / 'weights.json').read_text())
    earlier_sha = wl.ef.sha(ROOT / f'new weights/wn_{EARLIER}_candidate/weights.json')
    tab = pd.read_parquet(OUT / 'reliable_signals_train.parquet', columns=spec['columns']).loc[ids]
    g = pd.read_parquet(OUT / 'innov_v2_train.parquet').loc[ids]
    h = np.arcsinh(tab.to_numpy().clip(-24, 55))
    features = np.column_stack((h, 0.6 * h * g.pre_white.to_numpy()[:, None], 0.6 * h * g.pre_walk.to_numpy()[:, None]))
    with np.load(OUT / (PARENT + '_oof.npz')) as r:
        assert np.array_equal(r['ids'], ids)
        oof = {s: r[f's{s}'] for s in SEEDS}
    for s in SEEDS:
        for f in range(5):
            target = dst / f's{s}_f{f}.npz'
            if target.exists():
                with np.load(target) as r:
                    assert str(r['parent_sha']) == digest
                continue
            k = f's{s}_f{f}_'
            tr, va = (q[k + 'tr'], q[k + 'va'])
            with np.load(earlier / target.name) as r:
                assert str(r['parent_sha']) == earlier_sha
                assert np.array_equal(r['train_ids'], ids[tr]) and np.array_equal(r['validation_ids'], ids[va])
                z, v = (r['oof_train'], r['valid'])

            def edit(a, b, za, zb):
                aa, mean = centered(features[a], za)
                bb, _ = centered(features[b], zb, mean)
                return zb + bb @ fit(aa, za, y[a], True)
            vv = edit(tr, va, z, v)
            assert np.max(abs(vv - oof[s][va])) < 1e-07
            train = np.empty(len(tr))
            for a, b in StratifiedKFold(3, shuffle=True, random_state=4343 + f).split(tr, y[tr]):
                train[b] = edit(tr[a], tr[b], z[a], z[b])
            np.savez_compressed(target, parent_sha=digest, train_ids=ids[tr].to_numpy(dtype=str), validation_ids=ids[va].to_numpy(dtype=str), oof_train=train, valid=vv)
    with np.load(pack / 'replay_inputs.npz') as r:
        assert np.array_equal(r['ids'], tids)
        v = r['parent_logits'] + r['features'] @ spec['weights']
    np.savez_compressed(dst / 'full.npz', parent_sha=digest, train_ids=ids.to_numpy(dtype=str), test_ids=tids.to_numpy(dtype=str), oof_train=oof[7], test=v)
    return (dst, digest)

def run(tag, members, method):
    q, ids, tids, y, wt, wn, miss = wl.load()
    cache, digest = freeze(q, ids, tids, y)
    pp = [read_member(m, ids, tids) for m in members]
    x = np.column_stack([a[:, 0] for a, _ in pp])
    fields, paths = ([], [p for _, ps in pp for p in ps])
    for part, ix in (('train', ids), ('test', tids)):
        path = OUT / f'innov_v2_{part}.parquet'
        reg = pd.read_parquet(path, columns=['pre_white', 'pre_walk']).loc[ix]
        fields.append(reg.to_numpy())
        paths.append(path)
    reg = np.vstack(fields)

    def design(tr, z, va, v):
        a, b, c = np.quantile(z, [0.25, 0.5, 0.75])
        lo, med, hi = np.quantile(x[tr], [0.25, 0.5, 0.75], axis=0)

        def basis(ix, baseline):
            calibrated = (x[ix] - med) * ((c - a) / (hi - lo)) + b
            d = calibrated - baseline[:, None]
            if method == 'physical':
                white = reg[ix, 0] * np.exp(-0.5 * ((baseline + 1.1) / 1.25) ** 2)
                return d * (1.0 - 0.85 * white[:, None])
            if method == 'fit':
                return np.column_stack((d, d * reg[ix, :1], d * reg[ix, 1:2], d * np.tanh(baseline[:, None] / 2)))
            return d
        train = basis(tr, z)
        valid = basis(va, v)
        if method == 'fit':
            w = weights(train, z, y[tr])
        else:
            w = np.full(x.shape[1], 0.2 / x.shape[1])
        return (valid, w, dict(parent_quantiles=[a, b, c], member_quantiles=[lo.tolist(), med.tolist(), hi.tolist()]))
    scores, audits, preds = ({}, {}, {})
    for s in SEEDS:
        old, new, original = [np.empty(len(y)) for _ in range(3)]
        for f in range(5):
            k = f's{s}_f{f}_'
            tr, va = (q[k + 'tr'], q[k + 'va'])
            with np.load(cache / f's{s}_f{f}.npz') as a:
                assert str(a['parent_sha']) == digest and np.array_equal(ids[tr], a['train_ids'])
                assert np.array_equal(ids[va], a['validation_ids'])
                z, v = (a['oof_train'], a['valid'])
            F, w, _ = design(tr, z, va, v)
            new[va], old[va], original[va] = (v + F @ w, v, q[k + 'zv'])
        scores[s] = {k: wl.score(y, z, wt, wn, miss) for k, z in (('incumbent', old), ('new', new), ('original', original))}
        audits[s] = audit(y, miss, original, new)
        preds[f's{s}'] = new
    name = 'local_rank_' + tag
    passed = all((r['new']['auc'] > r['incumbent']['auc'] and r['new']['remaining'] >= r['original']['remaining'] and (r['new']['target_vs_neg'] > r['original']['target_vs_neg']) for r in scores.values()))
    rep = _rebuild.annotate(dict(parent=PARENT, members=members, method=method, accepted=passed, scores=scores, audits=audits))
    (OUT / (name + '.json')).write_text(json.dumps(rep, indent=2))
    print(name, passed, scores[7]['new'], audits[7]['candidate'], flush=True)
    np.savez_compressed(OUT / (name + '_oof.npz'), ids=ids.to_numpy(dtype=str), **preds)
    if not _rebuild.export(passed):
        return
    with np.load(cache / 'full.npz') as a:
        assert str(a['parent_sha']) == digest and np.array_equal(a['test_ids'], tids)
        z, v = (a['oof_train'], a['test'])
    F, w, center = design(np.arange(len(y)), z, len(y) + np.arange(len(tids)), v)
    dst = ROOT / f'new weights/wn_{name}_candidate'
    dst.mkdir(exist_ok=True)
    spec, output = (dst / 'weights.json', dst / 'submission.csv')
    sub = wl.ef.write_submission(expit(v + F @ w), tids, output)
    parent_pack = ROOT / f'new weights/wn_{PARENT}_candidate'
    dependencies = [parent_pack / 'weights.json', Path(__file__).resolve(), *paths]
    spec.write_text(json.dumps(dict(weights=w.tolist(), method=method, members=members, **center, hashes={str(p.relative_to(ROOT)): wl.ef.sha(p) for p in dependencies}), indent=2))
    np.savez_compressed(dst / 'replay_inputs.npz', ids=tids.to_numpy(dtype=str), order=tids.get_indexer(sub.id), parent_logits=v, features=F)
    (dst / 'validation.json').write_text(json.dumps(dict(**rep, recipe_sha256=wl.ef.sha(spec), submission_sha256=wl.ef.sha(output)), indent=2))
    shutil.copyfile(parent_pack / 'replay.py', dst / 'replay.py')
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('tag')
    p.add_argument('method', choices=('fixed', 'physical', 'fit'))
    p.add_argument('members', nargs='+')
    a = p.parse_args()
    with threadpool_limits(limits=1):
        run(a.tag, a.members, a.method)
