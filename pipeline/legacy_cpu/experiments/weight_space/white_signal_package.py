# Fixed winning-recipe CPU adaptation v1
"""Package a previously measured physical increment after verifying all six folds."""
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
from threadpoolctl import threadpool_limits
from white_local_rank_weights import centered
from white_nested import fit, audit
from white_checkpoint import SEEDS
from white_signal_audit import terms
from partition_evidence import OUT, ROOT
import white_loop as wl

def run(parent, name, signal):
    q, ids, tids, y, wt, wn, miss = wl.load()
    stage = ROOT / f'new weights/wn_{parent}_candidate'
    sha = wl.ef.sha(stage / 'weights.json')
    cache = OUT / ('frozen_' + parent)
    fields = []
    for part, ix in (('train', ids), ('test', tids)):
        table, cols = terms(signal)
        a = pd.read_parquet(OUT / f'{table}_{part}.parquet', columns=cols).loc[ix]
        h = np.arcsinh(a.to_numpy().clip(-24, 55))
        g = pd.read_parquet(OUT / f'innov_v2_{part}.parquet', columns=['pre_white', 'pre_walk']).loc[ix]
        fields.append(np.column_stack([h, 0.6 * h * g.pre_white.to_numpy()[:, None], 0.6 * h * g.pre_walk.to_numpy()[:, None]]))
    features = np.vstack(fields)
    scores = {}
    audits = {}
    arrays = {}

    def step(tr, va, z, v):
        x, mu = centered(features[tr], z)
        val, _ = centered(features[va], v, mu)
        w = fit(x, z, y[tr], True)
        return (v + val @ w, val, w, mu)
    reference = json.loads((OUT / f'signal_ablation_{parent}_{signal}.json').read_text())[signal]['delta']
    for s in SEEDS:
        old, base, new = [np.empty(len(y)) for _ in range(3)]
        for f in range(5):
            k = f's{s}_f{f}_'
            tr, va = (q[k + 'tr'], q[k + 'va'])
            with np.load(cache / f's{s}_f{f}.npz') as a:
                assert str(a['parent_sha']) == sha and np.array_equal(a['train_ids'], ids[tr]) and np.array_equal(a['validation_ids'], ids[va])
                z, v = (a['oof_train'], a['valid'])
            new[va] = step(tr, va, z, v)[0]
            old[va] = v
            base[va] = q[k + 'zv']
        scores[s] = {k: wl.score(y, v, wt, wn, miss) for k, v in (('incumbent', old), ('original', base), ('new', new))}
        assert abs(scores[s]['new']['auc'] - scores[s]['incumbent']['auc'] - reference[str(s)]['auc']) < 1e-10
        audits[s] = audit(y, miss, base, new)
        arrays[f's{s}'] = new
    accepted = all((a['new']['auc'] > a['incumbent']['auc'] and a['new']['remaining'] > a['original']['remaining'] and (a['new']['target_vs_neg'] > a['original']['target_vs_neg']) for a in scores.values()))
    tag = 'local_rank_' + name
    report = _rebuild.annotate(dict(parent=parent, signal=signal, scores=scores, audits=audits, accepted=accepted))
    print(tag, accepted, scores[7]['new'], audits[7]['candidate'], flush=True)
    (OUT / (tag + '.json')).write_text(json.dumps(report, indent=2))
    np.savez_compressed(OUT / (tag + '_oof.npz'), ids=ids.to_numpy(dtype=str), **arrays)
    if not _rebuild.export(accepted):
        return
    dst = ROOT / f'new weights/wn_{tag}_candidate'
    dst.mkdir(exist_ok=True)
    with np.load(cache / 'full.npz') as a:
        assert str(a['parent_sha']) == sha and np.array_equal(ids, a['train_ids']) and np.array_equal(tids, a['test_ids'])
        z, v = (a['oof_train'], a['test'])
    pred, F, w, mu = step(np.arange(len(y)), len(y) + np.arange(len(tids)), z, v)
    csv = dst / 'submission.csv'
    sub = wl.ef.write_submission(expit(pred), tids, csv)
    deps = [Path(__file__).resolve(), stage / 'weights.json', *[OUT / f'{t}_{p}.parquet' for t in (table, 'innov_v2') for p in ('train', 'test')]]
    recipe = dst / 'weights.json'
    recipe.write_text(json.dumps(dict(parent=parent, signal=signal, columns=cols, weights=w.tolist(), centers=mu.tolist(), hashes={str(p.relative_to(ROOT)): wl.ef.sha(p) for p in deps})))
    np.savez_compressed(dst / 'replay_inputs.npz', ids=tids.to_numpy(dtype=str), order=tids.get_indexer(sub.id), parent_logits=v, features=F)
    (dst / 'validation.json').write_text(json.dumps(dict(**report, recipe_sha256=wl.ef.sha(recipe), submission_sha256=wl.ef.sha(csv)), indent=2))
    shutil.copyfile(stage / 'replay.py', dst / 'replay.py')
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('parent')
    p.add_argument('name')
    p.add_argument('signal')
    a = p.parse_args()
    with threadpool_limits(limits=1):
        run(a.parent, a.name, a.signal)
