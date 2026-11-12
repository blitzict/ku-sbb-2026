# CPU port; original experiments/weight_space/white_walk_false_alarms.py, SHA256 fbdf3d833491dcb9e05422ab8f1dc3fec298907dbd53ee5b84f4d7ab757bf859
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Test whether innovation asymmetry can distinguish wandering controls from real breaks."""
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
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from white_checkpoint import SEEDS
from white_concordance import train
from white_freeze_first_layer import NAME
from white_local_rank_weights import centered
from white_nested import audit
from partition_evidence import OUT, ROOT
import white_loop as wl

def inputs(ids, tids):
    files = []

    def tab(name, cols):
        vs = []
        for p, ix in (('train', ids), ('test', tids)):
            path = OUT / f'{name}_{p}.parquet'
            files.append(path)
            vs.append(pd.read_parquet(path, columns=cols).loc[ix])
        return pd.concat(vs)
    x = tab('noise_tests', ['lr_cube_t', 'lr_cube_abs_t', 'lr_bim_shape_lr_sym', 'lr_skew_shape_lr_sym', 'lr_skew_q_gap'])
    shape = tab('mixture_law', ['gate'])
    sh = tab('tree_channels_claude', ['full_volatility_law', 'full_jump_law_state', 'full_jump_law_nullstate', 'full_trend', 'full_curve', 'full_ns3', 'full_phase_law'])
    gate = tab('bounds', ['phi1_ar1_pre']).iloc[:, 0].to_numpy() > 0.95
    check = pd.read_parquet(ROOT / 'tune_out/generator_segments.parquet', columns=['part', 'phi1_ar1', 'id'])
    check = check[check.part == 'pre'].set_index('id').loc[list(ids) + list(tids), 'phi1_ar1'].to_numpy() > 0.95
    assert np.array_equal(check, gate)
    fields = dict(cube=x.lr_cube_t, absolute_cube=x.lr_cube_abs_t, bimodal=x.lr_bim_shape_lr_sym, quantile_asymmetry=x.lr_skew_q_gap, asymmetry_change=x.lr_skew_shape_lr_sym, mixture_gate=shape.gate, volatility_vote=sh.full_volatility_law, state_vote=sh.iloc[:, 1:3].sum(1), trend_and_curve_vote=sh.iloc[:, 3:5].sum(1), classic_and_phase_vote=sh.iloc[:, 5:7].sum(1))
    return (pd.DataFrame(fields), gate, files)

def matched(x, g, z, y):
    positives = np.flatnonzero(g & (y == 1))
    neg = np.flatnonzero(g & (y == 0))
    neg = neg[z[neg] >= np.quantile(z[neg], 0.75)]
    left, right = linear_sum_assignment(abs(z[neg, None] - z[positives]))
    a, b = (neg[left], positives[right])
    sel = abs(z[a] - z[b]) < 0.1
    a, b = (a[sel], b[sel])
    report = dict(pairs=len(a), median_logit_gap=float(np.median(abs(z[a] - z[b]))), largest_logit_gap=float(abs(z[a] - z[b]).max()), gaps={})
    for name, col in x.items():
        v = col.to_numpy()
        d = v[a] - v[b]
        sc = v[g].std()
        report['gaps'][name] = dict(negative_minus_positive_sd=float(d.mean() / max(sc, 1e-10)), proportion_negative_lower=float(np.mean((d < 0) + 0.5 * (d == 0))))
    print('unique score-matched pairs', len(a), report['gaps']['cube'], report['gaps']['absolute_cube'], flush=True)
    return report

def design(raw, gate, tr, va, z, v, y):
    ii, jj = (gate[tr], gate[va])
    tx = raw[tr[ii]]
    vx = raw[va[jj]]
    med = np.median(tx, axis=0)
    scale = (np.quantile(tx, 0.82, axis=0) - np.quantile(tx, 0.18, axis=0)).clip(0.08)
    transform = lambda a: np.arcsinh(((a - med) / scale).clip(-20, 20))
    h, m = centered(transform(tx), z[ii])
    k, _ = centered(transform(vx), v[jj], m)

    def extra(h, s):
        cube = h[:, 0:2]
        shape = h[:, 2:3]
        votes = h[:, 6:7]
        return np.column_stack([h, 0.5 * np.tanh(s[:, None] + 0.5) * h, 0.6 * cube * np.tanh(shape), 0.6 * cube * np.tanh(votes), np.ones(len(s)), np.tanh(s / 1.7)])
    H, V = (extra(h, z[ii]), extra(k, v[jj]))
    F = np.zeros((len(tr), H.shape[1]))
    F[ii] = H
    U = np.zeros((len(va), H.shape[1]))
    U[jj] = V
    w = train(F, z, y[tr], np.ones(len(z)), 0.0)
    return (v + U @ w, U, w)

def main():
    q, ids, tids, y, wt, wn, miss = wl.load()
    tab, g, paths = inputs(ids, tids)
    n = len(y)
    assert not np.any(g[:n] & miss)
    base = np.load(OUT / (NAME + '_oof.npz'))
    assert np.array_equal(ids, base['ids'])
    probe = matched(tab.iloc[:n], g[:n], base['s7'], y)
    raw = tab.to_numpy()
    tag = 'local_rank_walk_false_alarms'
    cache = OUT / ('frozen_' + NAME)
    parent = ROOT / f'new weights/wn_{NAME}_candidate'
    digest = wl.ef.sha(parent / 'weights.json')
    scores, details, pred = ({}, {}, {})
    for s in SEEDS:
        z0 = base[f's{s}']
        new = np.empty(n)
        locked = np.empty(n)
        for f in range(5):
            key = f's{s}_f{f}_'
            tr, va = (q[key + 'tr'], q[key + 'va'])
            with np.load(cache / f's{s}_f{f}.npz') as a:
                assert str(a['parent_sha']) == digest and np.array_equal(ids[tr], a['train_ids']) and np.array_equal(ids[va], a['validation_ids'])
                z, v = (a['oof_train'], a['valid'])
            assert np.max(abs(v - z0[va])) < 1e-08
            new[va] = design(raw, g, tr, va, z, v, y)[0]
            locked[va] = q[key + 'zv']
        assert np.max(abs((z0 - new)[~g[:n]])) < 1e-08
        scores[s] = {k: wl.score(y, v, wt, wn, miss) for k, v in (('previous', z0), ('new', new), ('locked', locked))}
        scores[s]['walk_only'] = {k: roc_auc_score(y[g[:n]], v[g[:n]]) for k, v in (('previous', z0), ('new', new))}
        details[s] = audit(y, miss, z0, new)
        pred[f's{s}'] = new
        print(s, [(k, round(scores[s]['new'][k] - scores[s]['previous'][k], 7)) for k in ('auc', 'target_vs_neg', 'remaining')], scores[s]['walk_only'], flush=True)
    passed = all((v['new']['auc'] > v['previous']['auc'] for v in scores.values())) and all((v['new'][k] > v['locked'][k] for v in scores.values() for k in ('remaining', 'target_vs_neg')))
    result = dict(accepted=bool(passed), scores=scores, audits=details, observed_matches=probe)
    (OUT / (tag + '.json')).write_text(json.dumps(result, indent=2))
    np.savez_compressed(OUT / (tag + '_oof.npz'), ids=ids.to_numpy(dtype=str), **pred)
    if passed:
        with np.load(cache / 'full.npz') as a:
            assert np.array_equal(a['test_ids'], tids)
            z, v = (a['oof_train'], a['test'])
        pp, F, w = design(raw, g, np.arange(n), n + np.arange(len(tids)), z, v, y)
        folder = ROOT / f'new weights/wn_{tag}_candidate'
        folder.mkdir(exist_ok=True)
        rec, csv = (folder / 'weights.json', folder / 'submission.csv')
        sub = wl.ef.write_submission(expit(pp), tids, csv)
        deps = [parent / 'weights.json', Path(__file__).resolve(), *paths]
        rec.write_text(json.dumps(dict(weights=w.tolist(), hashes={str(p.relative_to(ROOT)): wl.ef.sha(p) for p in dict.fromkeys(deps)}), indent=2))
        np.savez_compressed(folder / 'replay_inputs.npz', ids=tids.to_numpy(dtype=str), order=tids.get_indexer(sub.id), parent_logits=v, features=F)
        (folder / 'validation.json').write_text(json.dumps(dict(**result, recipe_sha256=wl.ef.sha(rec), submission_sha256=wl.ef.sha(csv)), indent=2))
        shutil.copyfile(parent / 'replay.py', folder / 'replay.py')
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
