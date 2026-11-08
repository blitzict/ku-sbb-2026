# CPU port; original experiments/weight_space/physical_870.py, SHA256 abff17f3a2720b3f961411878d6e5c873656c5c475b328da59099cd792f17dda
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Program new physical evidence channels into the saved ensemble readout."""
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
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
import expert_fusion as ef
import innovation_head as ih
import mechanism_edit as me
import seasonal_head as sh
BASE_MODE = 'phase_ar_mix_transfer'
BASE = ef.PACKAGE / f'mechanism_{BASE_MODE}_candidate/weights.json'

def fit(x, z, y, penalty):

    def objective(w):
        v = z + x @ w
        return (np.logaddexp(0, v).sum() - y @ v + 0.5 * penalty * (w @ w), x.T @ (expit(v) - y) + penalty * w)
    r = minimize(objective, np.zeros(x.shape[1]), jac=True, method='L-BFGS-B', options={'maxiter': 650, 'ftol': 1e-11, 'maxcor': 20})
    if not r.success:
        raise RuntimeError(r.message)
    return r.x

def evidence(factors, summary, z, gate):
    v = np.clip(factors, -10, 30)
    return gate[:, None] * np.column_stack([np.ones(len(z)), np.tanh(z / 2), np.clip(summary - z, -5, 9), np.log1p(np.maximum(v, 0)), v / 6])

def tags(mode):
    return me.inputs(BASE_MODE) + ['ar_law']

def extra(mode, t, z, g, ix):
    law = t['ar_law'].iloc[ix]
    f = law[['level_bf', 'scale_bf', 'memory_bf', 'tail_bf']].to_numpy()
    gate = 1 - g[ix]
    summary = law['aggregate'].to_numpy()
    blocks = [evidence(f, summary, z, gate)]
    if 'routed' in mode:
        for fam in ('ar', 'walk', 'white'):
            w = gate * law['p_' + fam].to_numpy()
            fi = law[[fam + '_level', fam + '_scale', 'memory_bf', 'tail_bf']].to_numpy()
            blocks.append(evidence(fi, summary, z, w))
            blocks.append(w[:, None] * np.column_stack([np.log1p(law[fam + '_jump']), np.log(law.df)]))
    return np.column_stack(blocks)

def design(mode, t, z, g, ix):
    return np.column_stack([me.encode(BASE_MODE, t, z, g, ix), extra(mode, t, z, g, ix)])

def replay(recipe, out):
    art = json.loads(recipe.read_text())
    for file, h in art['hashes'].items():
        assert ef.sha(ef.ROOT / file) == h, file
    data = ef.read_data()
    ids, tids = data[1:3]
    _, _, gate, _ = sh.read_tables(data, True)
    p = pd.read_csv(ih.PARENT.parent / 'submission.csv', dtype={'id': str}).set_index('id').loc[tids, 'break_probability']
    z = logit(p.to_numpy())
    t = me.tables(data, tags(art['mode']))
    x = design(art['mode'], t, z, gate, np.arange(len(ids), len(gate)))
    ef.write_submission(expit(z + x @ np.asarray(art['weights'])), tids, out)

def run(mode):
    data = ef.read_data()
    _, ids, tids, y, wt, *_ = data
    folds, zall, g = ih.backbone(data)
    use = list(dict.fromkeys(tags(mode)))
    t = me.tables(data, use)
    store = np.load(ef.OUT / f'mechanism_{BASE_MODE}_oof.npz')
    assert np.array_equal(store['ids'], ids)
    scores, archive = ({}, {})
    weights = [15.0, 60.0]
    for seed, parts in folds.items():
        pred = {k: np.empty(len(y)) for k in ['base', *weights]}
        for tr, va, zr, zv in parts:
            a, b = (me.encode(BASE_MODE, t, z, g, i) for i, z in ((tr, zr), (va, zv)))
            beta = sh.fit(a, zr, y[tr], penalty=40.0)
            pred['base'][va] = zv + b @ beta
            a = np.column_stack([a, extra(mode, t, zr, g, tr)])
            b = np.column_stack([b, extra(mode, t, zv, g, va)])
            for penalty in weights:
                theta = fit(a, zr, y[tr], penalty)
                pred[penalty][va] = zv + b @ theta
        assert np.max(abs(pred['base'] - store[f's{seed}_40.0'])) < 1e-07
        scores[seed] = {k: ef.score(y, v, wt) for k, v in pred.items()}
        archive.update({f's{seed}_{k}': v for k, v in pred.items()})
        print(seed, scores[seed], flush=True)
    eligible = [p for p in weights if all((s[p][k] - s['base'][k] >= 0.0003 for s in scores.values() for k in ('auc', 'weighted_auc')))]
    selected = max(eligible, key=lambda p: np.mean([s[p]['auc'] for s in scores.values()])) if eligible else None
    report = {'mode': mode, 'scores': scores, 'selected': selected}
    name = 'physical870_' + mode
    (ef.OUT / f'{name}_results.json').write_text(json.dumps(report, indent=2))
    np.savez(ef.OUT / f'{name}_oof.npz', ids=ids.to_numpy(dtype=str), **archive)
    if selected:
        dest = ef.PACKAGE / f'{name}_candidate'
        dest.mkdir(exist_ok=True)
        tr = np.arange(len(y))
        theta = fit(design(mode, t, zall[tr], g, tr), zall[tr], y, selected)
        files = [Path(__file__).resolve(), ih.PARENT, ih.PARENT.parent / 'submission.csv', BASE]
        files.extend((ef.OUT / f'{tag}_{part}.parquet' for tag in use for part in ('train', 'test')))
        hashes = {str(p.relative_to(ef.ROOT)): ef.sha(p) for p in files}
        recipe = dest / 'weights.json'
        recipe.write_text(json.dumps({**report, 'weights': theta.tolist(), 'hashes': hashes}, indent=2))
        replay(recipe, dest / 'submission.csv')
    print('Selected', selected, flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', default='routed')
    p.add_argument('--replay', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    with threadpool_limits(limits=2):
        if args.replay:
            replay(args.replay, args.output)
        else:
            run(args.mode)
