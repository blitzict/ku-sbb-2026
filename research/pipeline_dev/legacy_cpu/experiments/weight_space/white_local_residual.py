# CPU port; original experiments/weight_space/white_local_residual.py, SHA256 9e2f025fdc4964a3526d33ad7a2abd5f627e67625076f5ebf14fea39e59e686c
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Test shallow residual trees on honest training-side scores from the accepted readout."""
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
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.special import expit
from threadpoolctl import threadpool_limits
from white_checkpoint import SEEDS
from white_expert_agreement import GROUPS
from white_joint import read_member
from white_nested import audit
from white_crop_growth import LAWS
from partition_evidence import ROOT, OUT
import white_loop as wl
PARENT = 'local_rank_family_agreement'

def load(ixs, new):
    cols = []
    deps = []
    for law, fields in LAWS.items():
        files = [OUT / f'{law}_{p}.parquet' for p in ('train', 'test')]
        deps += files
        vals = pd.concat([pd.read_parquet(p, columns=fields.split()).loc[ix] for p, ix in zip(files, ixs)])
        cols.append(np.arcsinh(vals.to_numpy().clip(-25, 65)))
    for law, fields in (('innov_v2', 'pre_white pre_walk post_white post_walk pre_phi post_phi'), ('periodic', 'gate period fraction'), ('volatility_law', 'pre_cluster post_cluster'), ('mixture_law', 'pre_bim post_bim balance_pre balance_post sep_pre sep_post'), ('curve', 'gate'), ('latent_level_pulse', 'slow_hidden_pre hidden_post walk_pre')):
        pp = [OUT / f'{law}_{p}.parquet' for p in ('train', 'test')]
        deps += pp
        cols.append(pd.concat([pd.read_parquet(p, columns=fields.split()).loc[ix] for p, ix in zip(pp, ixs)]).to_numpy())
    static = np.column_stack(cols).astype(np.float32)
    families = []
    members = []
    for names in (*GROUPS.values(), new):
        if not names:
            continue
        values = []
        for name in names:
            s, p = read_member(name, *ixs)
            deps += p
            values.append(s[:, 0])
        u = np.column_stack(values).astype(np.float32)
        members.append(u)
        families.append(u.mean(1))
    group = np.column_stack(families)
    e = np.column_stack(members)

    def features(ix, z):
        d = (e[ix] - z[:, None]).clip(-7, 7)
        g = (group[ix] - z[:, None]).clip(-7, 7)
        return np.column_stack([z, static[ix], d, g, d.std(1), np.median(d, 1)]).astype(np.float32)
    return (features, list(dict.fromkeys(deps)))

def model(x, z, y, seed):
    sampling = 0.55 + 0.75 * np.exp(-0.5 * ((z + 0.35) / 1.25) ** 2)
    data = lgb.Dataset(x, label=y, init_score=z, weight=sampling)
    params = dict(objective='binary', learning_rate=0.029, num_leaves=7, max_depth=3, min_data_in_leaf=230, min_gain_to_split=0.7, lambda_l2=36.0, feature_fraction=0.59, max_bin=39, num_threads=2, deterministic=True, force_col_wise=True, boost_from_average=False, seed=seed, verbosity=-1)
    return lgb.train(params, data, num_boost_round=160)

def run(tag, extra):
    q, ids, tids, y, wt, wn, miss = wl.load()
    N = len(y)
    make, deps = load((ids, tids), extra)
    cache = OUT / ('frozen_' + PARENT)
    folder = ROOT / f'new weights/wn_{PARENT}_candidate'
    sha = wl.ef.sha(folder / 'weights.json')
    reports = {}
    scores = {}
    arrays = {}
    for s in SEEDS:
        v = np.empty(N)
        p = np.empty(N)
        ref = np.empty(N)
        for f in range(5):
            with np.load(cache / f's{s}_f{f}.npz') as a:
                tr, va = (q[f's{s}_f{f}_{k}'] for k in ('tr', 'va'))
                assert str(a['parent_sha']) == sha and np.array_equal(a['train_ids'], ids[tr]) and np.array_equal(a['validation_ids'], ids[va])
                z, b = (a['oof_train'], a['valid'])
            m = model(make(tr, z), z, y[tr], 1501 + f)
            p[va] = b + m.predict(make(va, b), raw_score=True, num_threads=2)
            v[va] = b
            ref[va] = q[f's{s}_f{f}_zv']
        scores[s] = {k: wl.score(y, a, wt, wn, miss) for k, a in (('original', ref), ('incumbent', v), ('new', p))}
        reports[s] = audit(y, miss, ref, p)
        arrays[f's{s}'] = p
        print(tag, s, {k: round(scores[s]['new'][k], 7) for k in ('auc', 'remaining', 'target_vs_neg', 'weighted')}, flush=True)
    accepted = all((a['new']['auc'] > a['incumbent']['auc'] and a['new']['remaining'] > a['original']['remaining'] and (a['new']['target_vs_neg'] > a['original']['target_vs_neg']) for a in scores.values()))
    name = 'local_rank_' + tag
    rep = dict(scores=scores, audits=reports, accepted=accepted, parent=PARENT, extra=extra)
    (OUT / (name + '.json')).write_text(json.dumps(rep, indent=2))
    np.savez_compressed(OUT / (name + '_oof.npz'), ids=ids.to_numpy(dtype=str), **arrays)
    if accepted:
        dst = ROOT / f'new weights/wn_{name}_candidate'
        dst.mkdir(exist_ok=True)
        with np.load(cache / 'full.npz') as a:
            assert str(a['parent_sha']) == sha and np.array_equal(a['train_ids'], ids) and np.array_equal(a['test_ids'], tids)
            z, b = (a['oof_train'], a['test'])
        m = model(make(np.arange(N), z), z, y, 1501)
        ck = dst / 'residual_tree.txt'
        m.save_model(str(ck))
        d = m.predict(make(N + np.arange(len(tids)), b), raw_score=True, num_threads=2)
        csv = dst / 'submission.csv'
        sub = wl.ef.write_submission(expit(b + d), tids, csv)
        recipe = dst / 'weights.json'
        recipe.write_text(json.dumps(dict(parent=PARENT, extra=extra, weights=[1.0], hashes={str(p.relative_to(ROOT)): wl.ef.sha(p) for p in (Path(__file__).resolve(), folder / 'weights.json', ck, *deps)})))
        np.savez_compressed(dst / 'replay_inputs.npz', ids=tids.to_numpy(dtype=str), parent_logits=b, features=d[:, None], order=tids.get_indexer(sub.id))
        (dst / 'validation.json').write_text(json.dumps(dict(**rep, recipe_sha256=wl.ef.sha(recipe), submission_sha256=wl.ef.sha(csv)), indent=2))
        shutil.copyfile(folder / 'replay.py', dst / 'replay.py')
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('tag')
    p.add_argument('extra', nargs='*')
    a = p.parse_args()
    with threadpool_limits(limits=2):
        run(a.tag, a.extra)
