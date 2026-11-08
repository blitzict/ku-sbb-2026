# Fixed winning-recipe CPU adaptation v1
"""A local correction to an archived ensemble for identified periodic series."""
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
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits
import expert_fusion as ef
import residual_trial as rt
MODES = _rebuild.select(('local',), ('local', 'hybrid', 'compact'))
PERIODIC = ['fraction', 'pre_fraction', 'post_fraction', 'phase_gain', 'freq_gain', 'period_diff', 'amplitude_change', 'phase_change', 'common_level', 'common_scale', 'common_mad', 'common_shape', 'common_phi1', 'separate_level', 'separate_scale', 'separate_mad', 'separate_shape', 'separate_absmean', 'separate_phi1', 'separate_phi2', 'separate_seasonal', 'separate_sqphi', 'separate_logscale']
HARM = ['raw_shared', 'ar1_shared', 'raw_phase_bf', 'ar1_phase_bf', 'raw_freq_lr', 'ar1_freq_lr', 'raw_angle', 'ar1_angle', 'ar1_demod_level', 'ar1_demod_scale']

def read_tables(data, seq):
    ids, tids = data[1:3]
    tags = ['harmonic', 'periodic'] + (['periodic_seq'] if seq else [])
    tabs = [pd.concat([rt.frame(t, 'train', ids), rt.frame(t, 'test', tids)]) for t in tags]
    h, p = tabs[:2]
    x = pd.concat([p[PERIODIC], h[HARM]], axis=1)
    if seq:
        a = tabs[2].to_numpy()
        orig, common, sep = (part.mean(1) for part in np.split(a, 3, axis=1))
        x['net_common'] = common
        x['net_separate'] = sep
        x['net_common_delta'] = common - orig
        x['net_separate_delta'] = sep - orig
    gate = p.gate.to_numpy() * expit((p.period.to_numpy() - 9.0) / 1.5) * expit((38.0 - p.period.to_numpy()) / 2.0)
    return (tags, x, gate, h.to_numpy())

def lookup(H, tr, gate):
    return ef.distribution_maps(H[tr[gate[tr] > 0.5]])

def features(mode, H, gate, z, quantiles):
    g = gate[:, None]
    r = ef.ranks(H, quantiles)
    zr = np.tanh(z / 2)[:, None]
    features = [np.ones_like(zr), zr, r, np.maximum(r - 0.45, 0) * 2, np.minimum(r + 0.45, 0) * 2]
    if mode != 'compact':
        features.append(r * zr)
    return np.column_stack(features) * g

def fit(x, z, y, penalty=100.0):

    def objective(beta):
        v = z + x @ beta
        return (np.logaddexp(0, v).sum() - y @ v + 0.5 * penalty * (beta @ beta), x.T @ (expit(v) - y) + penalty * beta)
    res = minimize(objective, np.zeros(x.shape[1]), jac=True, method='L-BFGS-B', options={'maxiter': 150, 'ftol': 1e-12})
    assert res.success, res.message
    return res.x

def harmonic_parent(H, tr, va, zr, zv, y):
    q = ef.distribution_maps(H[tr])
    fr = rt.matrix('tails', ef.ranks(H[tr], q), zr, None)
    fv = rt.matrix('tails', ef.ranks(H[va], q), zv, None)
    th = ef.fit_head(fr, zr, y[tr])
    return (fr @ th[:-1] + th[-1], fv @ th[:-1] + th[-1])

def base(mode, z, dh, gate):
    return z + (gate if mode != 'hybrid' else 0.25 + 0.75 * gate) * dh

def replay(recipe, path):
    art = json.loads(recipe.read_text())
    for f, h in art['hashes'].items():
        assert ef.sha(ef.ROOT / f) == h, f
    data = ef.read_data()
    ids, tids = data[1:3]
    _, x, g, H = read_tables(data, art['sequence'])
    orig, harm = [pd.read_csv(ef.PACKAGE / p / 'submission.csv', dtype={'id': str}).set_index('id').loc[tids].break_probability.to_numpy() for p in ('expert_fusion_candidate', 'harmonic_candidate')]
    g = g[len(ids):]
    z = base(art['mode'], logit(orig), logit(harm) - logit(orig), g)
    f = features(art['mode'], x[art['cols']].to_numpy()[len(ids):], g, z, np.array(art['quantiles']))
    ef.write_submission(expit(z + f @ np.array(art['weights'])), tids, path)

def run(seq):
    data = ef.read_data()
    _, ids, tids, y, wt, *_ = data
    tags, x, g, H = read_tables(data, seq)
    name = 'seasonal' + ('_seq' if seq else '')
    old = np.load(ef.OUT / 'harmonic_oof.npz')
    assert np.array_equal(ids, old['ids'])
    folds, zall = rt.baseline(data)
    scores, archive = ({}, {})
    columns = {k: list(x.columns) for k in MODES}
    columns['compact'] = [k for k in x if k.startswith('net_') or (k in PERIODIC and k not in {'common_shape', 'common_phi1', 'separate_sqphi', 'separate_phi2', 'common_mad', 'separate_mad'})]
    for split, parts in folds.items():
        pred = {k: np.empty(len(y)) for k in ['harmonic', 'gate_only', *MODES]}
        for tr, va, zr, zv in parts:
            hr, hv = harmonic_parent(H, tr, va, zr, zv, y)
            pred['harmonic'][va] = zv + hv
            pred['gate_only'][va] = base('local', zv, hv, g[va])
            for mode in MODES:
                A = x[columns[mode]].to_numpy()
                q = lookup(A, tr, g)
                zrt, zvt = (base(mode, zr, hr, g[tr]), base(mode, zv, hv, g[va]))
                th = fit(features(mode, A[tr], g[tr], zrt, q), zrt, y[tr])
                pred[mode][va] = zvt + features(mode, A[va], g[va], zvt, q) @ th
        assert max(abs(pred['harmonic'] - old[f's{split}_tails'])) < 1e-07
        scores[split] = {k: ef.score(y, p, wt) for k, p in pred.items()}
        archive.update({f's{split}_{k}': p for k, p in pred.items()})
        print(split, scores[split], flush=True)
    eligible = [k for k in MODES if all((s[k][m] - s['harmonic'][m] >= 0.0003 for s in scores.values() for m in ('auc', 'weighted_auc')))]
    chosen = _rebuild.select('local', max(eligible, key=lambda k: np.mean([s[k]['auc'] for s in scores.values()])) if eligible else None)
    report = _rebuild.annotate({'scores': scores, 'selected': chosen, 'sequence': seq})
    (ef.OUT / f'{name}_results.json').write_text(json.dumps(report, indent=2))
    np.savez(ef.OUT / f'{name}_oof.npz', ids=ids.to_numpy(dtype=str), **archive)
    if chosen:
        dest = ef.PACKAGE / f'{name}_candidate'
        dest.mkdir(exist_ok=True)
        parent = ef.PACKAGE / 'harmonic_candidate/weights.json'
        par = json.loads(parent.read_text())
        z = zall[:len(y)]
        f = rt.matrix('tails', ef.ranks(H[:len(y)], np.array(par['quantiles'])), z, None)
        hp = np.array(par['coefficients'])
        z = base(chosen, z, f @ hp[:-1] + hp[-1], g[:len(y)])
        cols = columns[chosen]
        A = x[cols].to_numpy()
        q = lookup(A, np.arange(len(y)), g)
        th = fit(features(chosen, A[:len(y)], g[:len(y)], z, q), z, y)
        files = [Path(__file__).resolve(), rt.ANCHOR, rt.ANCHOR.parent / 'submission.csv', parent, parent.parent / 'submission.csv']
        files += [ef.OUT / f'{t}_{p}.parquet' for t in tags for p in ('train', 'test')]
        hashes = {str(f.relative_to(ef.ROOT)): ef.sha(f) for f in files}
        recipe = dest / 'weights.json'
        recipe.write_text(json.dumps({**report, 'mode': chosen, 'cols': cols, 'quantiles': q.tolist(), 'weights': th.tolist(), 'hashes': hashes}, indent=2))
        replay(recipe, dest / 'submission.csv')
    print('Selected', chosen, flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--sequence', action='store_true')
    p.add_argument('--replay', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    with threadpool_limits(limits=2):
        if args.replay:
            replay(args.replay, args.output)
        else:
            run(args.sequence)
