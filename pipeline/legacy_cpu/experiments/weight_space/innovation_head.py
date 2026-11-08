# CPU port; original experiments/weight_space/innovation_head.py, SHA256 cd193bfa544cd8c4189f8aa456b2aa87ba06b00c89851216a76d65c57eedd81f
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Route innovation evidence on top of the saved periodic sequence specialist."""
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
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits
import expert_fusion as ef
import residual_trial as rt
import seasonal_head as sh
PARENT = ef.PACKAGE / 'seasonal_seq_candidate/weights.json'

def backbone(data):
    ids, tids, y = data[1:4]
    tabs, x, g, H = sh.read_tables(data, True)
    art = json.loads(PARENT.read_text())
    mode, cols = (art['mode'], art['cols'])
    A = x[cols].to_numpy()
    splits, zall = rt.baseline(data)
    old = np.load(ef.OUT / 'seasonal_seq_oof.npz')
    assert np.array_equal(old['ids'], ids)
    folds = {}
    for seed, parts in splits.items():
        folds[seed] = []
        for tr, va, zr, zv in parts:
            hr, hv = sh.harmonic_parent(H, tr, va, zr, zv, y)
            zrt, zvt = (sh.base(mode, zr, hr, g[tr]), sh.base(mode, zv, hv, g[va]))
            q = sh.lookup(A, tr, g)
            fr, fv = (sh.features(mode, A[tr], g[tr], zrt, q), sh.features(mode, A[va], g[va], zvt, q))
            th = sh.fit(fr, zrt, y[tr])
            zr, zv = (zrt + fr @ th, zvt + fv @ th)
            assert np.max(abs(zv - old[f's{seed}_{mode}'][va])) < 1e-07
            folds[seed].append((tr, va, zr, zv))
    har = json.loads((ef.PACKAGE / 'harmonic_candidate/weights.json').read_text())
    hc = np.array(har['coefficients'])
    dh = rt.matrix('tails', ef.ranks(H, np.array(har['quantiles'])), zall, None) @ hc[:-1] + hc[-1]
    z = sh.base(mode, zall, dh, g)
    z = z + sh.features(mode, A, g, z, np.array(art['quantiles'])) @ np.array(art['weights'])
    probs = pd.read_csv(PARENT.parent / 'submission.csv', dtype={'id': str}).set_index('id').loc[tids].break_probability
    assert np.max(abs(expit(z[len(y):]) - probs)) < 1e-12
    return (folds, z, g)

def pack(data, net, transfer):
    ids, tids, y = data[1:4]

    def block(tag):
        return pd.concat([rt.frame(tag, 'train', ids), rt.frame(tag, 'test', tids)])
    p = block('innov_v2')
    selected = ['phi', 'pre_phi', 'post_phi', 'dynamics_bf', 'dynamics_delta', 'crossover_walk', 'raw_jump', 'walk_reset', 'pre_endpoint', 'post_initial']
    selected.extend(('shared_' + k for k in ('level', 'scale', 'median', 'mad', 'shape', 'tail_shape', 'skew_shape', 'abs_gap', 'pre_sd', 'post_sd', 'start_jump')))
    selected.extend(('separate_' + k for k in ('level', 'scale', 'shape', 'abs_gap')))
    x = p[selected].copy()
    if net:
        a = block('innov_v2_seq').to_numpy()
        orig, common, sep = (t.mean(1) for t in np.split(a, 3, axis=1))
        x['net_orig'], x['net_common'], x['net_sep'] = (orig, common, sep)
        x['net_change'], x['net_change_separate'] = (common - orig, sep - orig)
    if transfer:
        x = pd.concat([x, block('transfer_member')], axis=1)
    walk = np.maximum(p.walk, 0.65 * p.pre_walk).to_numpy()
    ar = np.clip((abs(p.phi.to_numpy()) - 0.08) / 0.25, 0, 1) * (1 - walk)
    return (x.to_numpy(), np.stack([walk, ar], axis=1), list(x.columns))

def set_gates(mode, G, season):
    w, ar = G.T
    if mode == 'walk':
        return [w * (1 - season)]
    if mode == 'family':
        return [w * (1 - season), ar * (1 - season)]
    return [w * (1 - season), ar * (1 - season), np.ones(len(w))]

def learn(X, G, season, z, y, train, mode):
    result = []
    for g in set_gates(mode, G, season):
        q = sh.lookup(X, train, g)
        f = sh.features('compact', X[train], g[train], z, q)
        penalty = 130.0 if g.min() > 0.9 else 75.0
        theta = sh.fit(f, z, y[train], penalty=penalty)
        result.append({'q': q.tolist(), 'weights': theta.tolist()})
        z = z + f @ theta
    return result

def apply(X, G, season, z, indices, mode, params):
    for g, parameters in zip(set_gates(mode, G, season), params):
        f = sh.features('compact', X[indices], g[indices], z, np.array(parameters['q']))
        z = z + f @ np.array(parameters['weights'])
    return z

def replay(recipe, dest):
    a = json.loads(recipe.read_text())
    for path, h in a['hashes'].items():
        assert ef.sha(ef.ROOT / path) == h, path
    data = ef.read_data()
    _, ids, tids, y, *_ = data
    _, _, g, _ = sh.read_tables(data, True)
    x, G, cols = pack(data, a['net'], a['transfer'])
    assert cols == a['columns']
    p = pd.read_csv(PARENT.parent / 'submission.csv', dtype={'id': str}).set_index('id').loc[tids].break_probability
    z = apply(x, G, g, logit(p.to_numpy()), np.arange(len(y), len(x)), a['mode'], a['params'])
    ef.write_submission(expit(z), tids, dest)

def main(net, transfer):
    data = ef.read_data()
    _, ids, tids, y, wt, *_ = data
    folds, z, g = backbone(data)
    x, G, cols = pack(data, net, transfer)
    name = 'innov_branch' + ('_net' if net else '') + ('_transfer' if transfer else '')
    modes = ['walk', 'family', 'pooled']
    scores, archive = ({}, {})
    for seed, parts in folds.items():
        pred = {k: np.empty(len(y)) for k in ['anchor', *modes]}
        for tr, va, zr, zv in parts:
            pred['anchor'][va] = zv
            for mode in modes:
                params = learn(x, G, g, zr, y, tr, mode)
                pred[mode][va] = apply(x, G, g, zv, va, mode, params)
        scores[seed] = {k: ef.score(y, p, wt) for k, p in pred.items()}
        archive.update({f's{seed}_{k}': p for k, p in pred.items()})
        print(seed, scores[seed], flush=True)
    eligible = [k for k in modes if all((s[k][m] - s['anchor'][m] >= 0.0003 for s in scores.values() for m in ('auc', 'weighted_auc')))]
    chosen = max(eligible, key=lambda k: np.mean([s[k]['auc'] for s in scores.values()])) if eligible else None
    report = {'scores': scores, 'selected': chosen, 'net': net, 'transfer': transfer}
    (ef.OUT / f'{name}_results.json').write_text(json.dumps(report, indent=2))
    np.savez(ef.OUT / f'{name}_oof.npz', ids=ids.to_numpy(dtype=str), **archive)
    if chosen:
        dest = ef.PACKAGE / f'{name}_candidate'
        dest.mkdir(exist_ok=True)
        params = learn(x, G, g, z[:len(y)], y, np.arange(len(y)), chosen)
        tags = ['innov_v2'] + (['innov_v2_seq'] if net else []) + (['transfer_member'] if transfer else [])
        files = [Path(__file__).resolve(), PARENT, PARENT.parent / 'submission.csv']
        files.extend((ef.OUT / f'{t}_{p}.parquet' for t in tags for p in ('train', 'test')))
        hashes = {str(p.relative_to(ef.ROOT)): ef.sha(p) for p in files}
        recipe = dest / 'weights.json'
        recipe.write_text(json.dumps({**report, 'mode': chosen, 'columns': cols, 'params': params, 'hashes': hashes}, indent=2))
        replay(recipe, dest / 'submission.csv')
    print('Selected', chosen, flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--net', action='store_true')
    p.add_argument('--transfer', action='store_true')
    p.add_argument('--replay', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    with threadpool_limits(limits=2):
        if args.replay:
            replay(args.replay, args.output)
        else:
            main(args.net, args.transfer)
