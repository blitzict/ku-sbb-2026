# CPU port; original experiments/weight_space/other_horizon_trial.py, SHA256 3b2977e3418c48e5de7edb5d68d1394fc9d311e892ee93f7e27250c31e95b321
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Measure new serial-dynamics signals with held-out model weights and matched controls."""
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
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from hard_pair_budget import cohorts
from other_missed_breaks import CURRENT, rankings
from partition_evidence import ROOT, OUT
from white_checkpoint import SEEDS
from white_local_rank_weights import centered, weights
import white_loop as wl

def load_features(ids, tids):
    series = ids.append(tids)
    paths = []

    def read(table):
        out = []
        for part, ix in (('train', ids), ('test', tids)):
            path = OUT / f'{table}_{part}.parquet'
            paths.append(path)
            out.append(pd.read_parquet(path).loc[ix])
        return pd.concat(out)
    data = read('other_memory_horizons')
    phase = read('periodic').pre_fraction.to_numpy()
    p = ROOT / 'tune_out/generator_segments.parquet'
    gen = pd.read_parquet(p, columns=['id', 'part', 'p_bic', 'phi1_ar1'])
    gen = gen[gen.part == 'pre'].set_index('id').loc[series]
    phi = gen.phi1_ar1.to_numpy()
    serial = ((gen.p_bic != 0) | (gen.phi1_ar1 > 0.95)).to_numpy()
    X = data.filter(regex='^(all_|replicate_)').to_numpy().clip(-30, 45)
    X = np.arcsinh(X) * serial[:, None]
    H = np.column_stack([X, 0.65 * X * np.tanh(2.5 * phi[:, None]), 0.5 * X * phase[:, None]])
    return (data, gen, H, serial, [*paths, p])

def matched(ids, gen, score, y, miss, data):
    from scipy.stats import rankdata
    from white_signature_screen import OLD
    old = np.column_stack([rankdata(np.load(ROOT / f'tune_out/oof_{m}.npy')) / len(y) for m in OLD])
    low = (old < 0.5).all(1)
    assert np.array_equal(miss, low & (y == 1))
    phi = gen.phi1_ar1.to_numpy()[:len(y)]
    p = gen.p_bic.to_numpy()[:len(y)]
    family = np.select([phi > 0.95, p >= 5, p == 0, phi < -0.2, phi > 0.2], [0, 1, 2, 3, 4], 5)
    lens = pd.read_parquet(ROOT / 'clean/features_train.parquet', columns=['n_pre', 'n_post']).loc[ids]
    full = np.column_stack([old, score, phi, np.log(lens)])
    full = (full - full.mean(0)) / full.std(0).clip(0.01)
    full *= np.r_[np.full(len(OLD), 0.25), 1.5, 1.0, 0.7, 0.7]
    case, control = ([], [])
    for group in np.unique(family):
        a = np.flatnonzero(miss & (family == group) & (family != 2))
        b = np.flatnonzero(low & (y == 0) & (family == group))
        if len(a) and len(b):
            i, j = linear_sum_assignment(cdist(full[a], full[b], 'sqeuclidean'))
            case.extend(a[i])
            control.extend(b[j])
    assert len(set(control)) == len(control)
    s = data.all_memory_at_fixed_lag1.to_numpy()
    diff = s[case] - s[control]
    rng = np.random.default_rng(49821)
    boot = np.concatenate([rng.choice(diff, (256, len(diff))).mean(1) for _ in range(64)])
    return dict(n=len(diff), case=float(s[case].mean()), control=float(s[control].mean()), gap=float(diff.mean()), wins=float(np.mean(diff > 0)), interval=np.quantile(boot, [0.025, 0.975]).tolist(), median_absolute_current_logit_gap=float(np.median(abs(score[case] - score[control]))))

def main():
    q, ids, tids, y, wt, wn, white = wl.load()
    genall = cohorts(ids, y)[0]
    other = genall & ~white
    data, gen, h, serial, paths = load_features(ids, tids)
    assert np.array_equal(~serial[:len(y)], wn)
    dst = OUT / ('frozen_' + CURRENT)
    baseline = ROOT / f'new weights/wn_{CURRENT}_candidate'
    tag = 'local_rank_long_memory'
    digest = wl.ef.sha(baseline / 'weights.json')
    pred, scores = ({}, {})

    def step(tr, va, z, v):
        X, ref = centered(h[tr], z)
        V, _ = centered(h[va], v, ref)
        X *= serial[tr, None]
        V *= serial[va, None]
        w = 0.5 * weights(X, z, y[tr])
        return (v + V @ w, V, w)
    for s in [*SEEDS, 61817]:
        old, new, locked = [np.empty(len(y)) for _ in range(3)]
        for f, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=s).split(ids, y)):
            with np.load(dst / f's{s}_f{f}.npz') as a:
                assert np.array_equal(ids[tr], a['train_ids']) and np.array_equal(ids[va], a['validation_ids'])
                assert str(a['parent_sha']) == digest
                z, v = (a['oof_train'], a['valid'])
                locked[va] = a['locked']
            new[va] = step(tr, va, z, v)[0]
            old[va] = v
        if s == 7:
            match = matched(ids, gen, old, y, genall, data)
        record = {}
        for k, p in (('locked', locked), ('previous', old), ('new', new)):
            r = wl.score(y, p, wt, wn, white)
            rank = rankings(y, p)
            for label, mask in (('other_historical', other), ('historical_white', white), ('all_other', (y == 1) & ~white)):
                r[label + '_auc'] = float(rank[mask].mean() / sum(y == 0))
                r[label + '_caught_20'] = int(np.sum(rank[mask] >= 0.8 * sum(y == 0)))
            record[k] = r
        print(s, [round(record['new'][k], 8) for k in ('auc', 'target_vs_neg', 'other_historical_auc', 'remaining')], flush=True)
        scores[s] = record
        pred[f's{s}'] = new
    ok = all((r['new']['auc'] > r['previous']['auc'] and all((r['new'][k] >= r['locked'][k] for k in ('target_vs_neg', 'remaining'))) for r in scores.values()))
    result = dict(accepted=ok, scores=scores, matched=match)
    print(match, flush=True)
    (OUT / (tag + '.json')).write_text(json.dumps(result, indent=2))
    np.savez_compressed(OUT / (tag + '_oof.npz'), ids=ids.to_numpy(dtype=str), **pred)
    if not ok:
        return
    pack = ROOT / f'new weights/wn_{tag}_candidate'
    pack.mkdir(exist_ok=True)
    with np.load(dst / 'full.npz') as a:
        assert np.array_equal(a['train_ids'], ids) and np.array_equal(a['test_ids'], tids)
        v = a['test']
        new, V, w = step(np.arange(len(y)), len(y) + np.arange(len(tids)), a['oof_train'], v)
    sub = wl.ef.write_submission(expit(new), tids, pack / 'submission.csv')
    deps = [baseline / 'weights.json', Path(__file__).resolve(), *paths]
    rec = pack / 'weights.json'
    rec.write_text(json.dumps(dict(weights=w.tolist(), hashes={str(p.relative_to(ROOT)): wl.ef.sha(p) for p in deps}), indent=2))
    np.savez_compressed(pack / 'replay_inputs.npz', ids=tids.to_numpy(dtype=str), order=tids.get_indexer(sub.id), parent_logits=v, features=V)
    (pack / 'validation.json').write_text(json.dumps(dict(**result, recipe_sha256=wl.ef.sha(rec), submission_sha256=wl.ef.sha(pack / 'submission.csv')), indent=2))
    shutil.copyfile(baseline / 'replay.py', pack / 'replay.py')
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
