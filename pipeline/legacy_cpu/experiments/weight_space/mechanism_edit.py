# Fixed winning-recipe CPU adaptation v1
"""Small, physically gated corrections of the verified sequence specialist."""
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
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits
import expert_fusion as ef
import innovation_head as ih
import residual_trial as rt
import seasonal_head as sh

def tables(data, tags):
    ids, tids = data[1:3]
    return {t: pd.concat([rt.frame(t, 'train', ids), rt.frame(t, 'test', tids)]) for t in tags}

def encode(mode, t, z, g, ix):
    f = []
    season = g[ix, None]

    def add(gate, X):
        f.append(gate * np.column_stack(X))
    if 'phase' in mode:
        ph = t['phase_law'].iloc[ix]
        v = ph[['shape_bf', 'level_bf', 'amplitude_bf', 'phase_bf', 'scale_bf', 'reset_bf']].to_numpy()
        v = np.clip(v, -10, 30)
        ev = np.clip(ph.mixture_evidence, -5, 20)
        u = np.log1p(np.maximum(v, 0))
        add(season, [np.ones(len(z)), np.tanh(z / 2), np.clip(ev - z, -4, 8), u, v / 6, np.log1p(np.clip(ph.temporal, 0, 20)), np.log1p(np.clip(ph.shape_abs, 0, 20))])
    if 'ar' in mode:
        inn = t['innov_v2'].iloc[ix]
        a = t['innov_v2_seq'].iloc[ix].to_numpy()
        original, c, s = (p.mean(1) for p in np.split(a, 3, axis=1))
        walk = np.maximum(inn.walk, 0.6 * inn.pre_walk).to_numpy()
        ar = np.clip((np.maximum(abs(inn.pre_phi), abs(inn.phi)).to_numpy() - 0.06) / 0.35, 0, 1)
        d = np.column_stack([c - original, s - original, c - z, s - z]).clip(-7, 7)
        for gate in ((1 - season) * walk[:, None], (1 - season) * (1 - walk[:, None]) * ar[:, None]):
            add(gate, [np.ones(len(z)), d, np.tanh(z / 2)[:, None] * d, np.maximum(d - 1, 0)])
    if 'mix' in mode:
        m = t['mixture_law'].iloc[ix]
        gate = m.gate.to_numpy()[:, None] * (1 - 0.75 * season)
        v = m.drop(columns=['gate']).to_numpy()
        add(gate, [np.ones(len(z)), np.tanh(z / 2), np.arcsinh(np.clip(v, -80, 80)), np.clip(m.mix_evidence - z, -5, 10)])
    if 'transfer' in mode:
        x = t['transfer_member'].iloc[ix].to_numpy()
        dd = np.clip(x - z[:, None], -7, 7)
        add(np.ones_like(season), [dd, dd * np.tanh(z[:, None] / 2), season * dd])
    return np.column_stack(f)

def inputs(mode):
    tags = []
    for token, names in (('phase', ['phase_law']), ('ar', ['innov_v2_seq', 'innov_v2']), ('mix', ['mixture_law']), ('transfer', ['transfer_member'])):
        if token in mode:
            tags.extend(names)
    return tags

def replay(recipe, dest):
    a = json.loads(recipe.read_text())
    for name, h in a['hashes'].items():
        assert ef.sha(ef.ROOT / name) == h, name
    data = ef.read_data()
    ids, tids = data[1:3]
    _, _, g, _ = sh.read_tables(data, True)
    mode = a['mode']
    t = tables(data, inputs(mode))
    p = pd.read_csv(ih.PARENT.parent / 'submission.csv', dtype={'id': str}).set_index('id').loc[tids].break_probability
    z = logit(p.to_numpy())
    f = encode(mode, t, z, g, np.arange(len(ids), len(g)))
    ef.write_submission(expit(z + f @ np.array(a['weights'])), tids, dest)

def run(mode):
    data = ef.read_data()
    _, ids, tids, y, wt, *_ = data
    folds, zall, g = ih.backbone(data)
    tags = inputs(mode)
    t = tables(data, tags)
    scores, archive = ({}, {})
    penalties = _rebuild.select([40.0], [40.0, 160.0])
    for seed, parts in folds.items():
        pred = {k: np.empty(len(y)) for k in ['anchor', *penalties]}
        for tr, va, zr, zv in parts:
            pred['anchor'][va] = zv
            a, b = (encode(mode, t, zr, g, tr), encode(mode, t, zv, g, va))
            for penalty in penalties:
                th = sh.fit(a, zr, y[tr], penalty=penalty)
                pred[penalty][va] = zv + b @ th
        scores[seed] = {k: ef.score(y, p, wt) for k, p in pred.items()}
        archive.update({f's{seed}_{k}': p for k, p in pred.items()})
        print(seed, scores[seed], flush=True)
    kept = [k for k in penalties if all((s[k][m] - s['anchor'][m] >= 0.0003 for s in scores.values() for m in ('auc', 'weighted_auc')))]
    chosen = _rebuild.select(40.0, max(kept, key=lambda k: np.mean([s[k]['auc'] for s in scores.values()])) if kept else None)
    name = 'mechanism_' + mode
    report = _rebuild.annotate({'mode': mode, 'scores': scores, 'selected': chosen})
    (ef.OUT / f'{name}_results.json').write_text(json.dumps(report, indent=2))
    np.savez(ef.OUT / f'{name}_oof.npz', ids=ids.to_numpy(dtype=str), **archive)
    if chosen:
        dest = ef.PACKAGE / f'{name}_candidate'
        dest.mkdir(exist_ok=True)
        tr = np.arange(len(y))
        th = sh.fit(encode(mode, t, zall[tr], g, tr), zall[tr], y, penalty=chosen)
        files = [Path(__file__).resolve(), ih.PARENT, ih.PARENT.parent / 'submission.csv']
        files.extend((ef.OUT / f'{tag}_{p}.parquet' for tag in tags for p in ('train', 'test')))
        hashes = {str(p.relative_to(ef.ROOT)): ef.sha(p) for p in files}
        path = dest / 'weights.json'
        path.write_text(json.dumps({**report, 'weights': th.tolist(), 'hashes': hashes}, indent=2))
        replay(path, dest / 'submission.csv')
    print('Selected', chosen, flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', default='phase_ar')
    p.add_argument('--replay', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    with threadpool_limits(limits=2):
        if args.replay:
            replay(args.replay, args.output)
        else:
            run(args.mode)
