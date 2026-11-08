# Fixed winning-recipe CPU adaptation v1
"""Condition each expert's residual weight on which measurable change it sees."""
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
from scipy.special import expit, softmax, logsumexp
from threadpoolctl import threadpool_limits
import white_loop as wl
from white_joint import read_member
from white_nested import gate_fields, law_design, fit, audit
from white_outer_priors import load, fit_on
from white_checkpoint import SEEDS
from partition_evidence import ROOT, OUT
CORE = ['net', 'lawtree_both', 'claude_law', 'claude_jump', 'lawshared', 'lawblock', 'lawconditional']

def compile_inputs(ids, tids, models, physical):
    paths = []

    def table(tag):
        pp = [OUT / f'{tag}_{part}.parquet' for part in ('train', 'test')]
        paths.extend(pp)
        return pd.concat([pd.read_parquet(p).loc[ix] for p, ix in zip(pp, (ids, tids))])
    g, t, wh = gate_fields(ids, tids)
    proc = table('innov_v2')
    walk = proc.pre_walk.to_numpy()
    pulse, state = [table('jump_law_' + v) for v in ('pulse', 'state')]
    constants = []
    members = []
    for a, target in ((pulse, 0.2 + 0.8 * wh), (state, 1 - 0.35 * wh)):
        constants.append(np.arcsinh(a.clip(-24, 60).to_numpy()) * target[:, None])
        h = a[['level2', 'scale2', 'shape0', 'shape1']].clip(-14, 35).to_numpy()
        members.append(((logsumexp(h, 1) - np.log(h.shape[1]))[:, None], target))
    w = table('walk_jump')
    constants.append(walk[:, None] * np.arcsinh(w.clip(-25, 60).to_numpy()))
    members.append((w.total.to_numpy()[:, None], walk))
    expert = []
    for model in models:
        score, p = read_member(model, ids, tids)
        paths += p
        members.append((score, np.ones(len(wh))))
        expert.append(score)
    E = np.column_stack(expert)
    if 'volatility' in physical:
        a = table('volatility_law')
        h = np.arcsinh(a.clip(-25, 60).to_numpy())
        reliability = np.maximum(a.pre_cluster, a.pooled_cluster).to_numpy()
        constants.extend([h, reliability[:, None] * h])
        joint = a[['new_memory', 'memory_off', 'memory_on', 'new_scale', 'new_law']].clip(-13, 40).to_numpy()
        members.append(((logsumexp(joint, 1) - np.log(joint.shape[1]))[:, None], 0.32 + 0.68 * reliability))
    if 'boundary' in physical:
        for stream in ('state', 'pulse'):
            h = table('boundary_law_' + stream)
            cols = [c for c in h if c.endswith(('_at', '_odds0.35', '_mid_weak', '_short_weak', '_one_strong'))]
            constants.append(0.55 * np.arcsinh(h[cols].clip(-22, 50).to_numpy()))
    const = np.column_stack(constants)
    osc = table('periodic')
    vol = table('state')
    curve = table('curve')
    ar = table('ar_law')
    osc_gate = osc.gate.to_numpy()
    vol_gate = vol.pre_cluster.to_numpy()

    def avg(v):
        return logsumexp(np.column_stack(v).clip(-9, 13) * 0.52, 1) / 0.52 - np.log(len(v)) / 0.52
    channels = [avg([state.level2, state.level4, pulse.level2]), avg([state.scale2, state.scale4, pulse.scale2]), avg([state.shape0, state.shape1, pulse.shape1]), avg([ar.memory_bf, vol.ar_change]), np.log(t.clip(0.015)) + curve.mixture.clip(-9, 13).to_numpy(), np.log(osc_gate.clip(0.015)) + np.maximum(osc.phase_gain, osc.freq_gain).clip(-9, 13).to_numpy(), np.log(walk.clip(0.015)) + w.total.clip(-9, 13).to_numpy(), np.log(vol_gate.clip(0.015)) + vol.vol_split_bf.clip(-9, 13).to_numpy()]
    posterior = softmax(np.column_stack([np.zeros(len(wh)), *channels]).clip(-10, 13) * 0.53, axis=1)
    regime = np.column_stack([wh, walk, t, osc_gate, vol_gate])

    def design(h, i, z, mode):
        f = [law_design(h, z, g[i], t[i]), const[i]]
        for a, v in members:
            d = v[i, None] * (a[i] - z[:, None]).clip(-7, 8)
            f.extend([d, d * np.tanh(z[:, None] / 2), d * wh[i, None], d * t[i, None], d * walk[i, None]])
        if mode != 'base':
            e = E[i]
            med = np.median(e, axis=1, keepdims=True)
            d = (e - med).clip(-6, 6)
            spread = np.mean(abs(d), axis=1, keepdims=True)
            consensus = np.column_stack([med[:, 0] - z, spread[:, 0], np.ptp(e, axis=1), np.tanh(med[:, 0] / 2)])
            f.extend([0.6 * d * spread.clip(0, 2), 0.6 * np.abs(d), 0.7 * np.tanh(d), 0.45 * consensus, 0.55 * (d[:, :, None] * posterior[i, None, :]).reshape(len(i), -1)])
            if mode == 'full':
                f.extend([0.45 * (d[:, :, None] * regime[i, None, :]).reshape(len(i), -1), 0.42 * ((e - z[:, None]).clip(-6, 6)[:, :, None] * posterior[i, None, :]).reshape(len(i), -1), 0.35 * (consensus[:, :, None] * posterior[i, None, :]).reshape(len(i), -1)])
        return np.column_stack(f)
    return (design, paths)

def run(mode, extra, seeds, tag, physical):
    q, ids, tids, y, wt, wn, miss = wl.load()
    n = len(y)
    models = CORE + extra
    build, paths = compile_inputs(ids, tids, models, physical)
    name = f'consult_{mode}_{tag}'
    scores = {}
    audits = {}
    oof = {}
    best = OUT / 'wn_physical_pulse_state_walk_net_lawtree_both_claude_law_claude_jump_lawshared_lawblock_lawconditional_rank_oof.npz'
    with np.load(best) as ref:
        assert np.array_equal(ref['ids'], ids)
        incumbent = {s: ref[f's{s}'] for s in seeds}
    for seed in seeds:
        base = np.empty(n)
        new = np.empty(n)
        for f in range(5):
            folder = OUT / f'law_outer{seed}'
            k = f's{seed}_f{f}_'
            tr, va, zr, zv = [q[k + a] for a in ('tr', 'va', 'zr', 'zv')]
            with np.load(folder / f'fold{f}_rows.npz') as a:
                assert np.array_equal(ids[tr], a['train_ids']) and np.array_equal(ids[va], a['validation_ids'])
            H = pd.read_parquet(folder / f'fold{f}_train.parquet').loc[ids].to_numpy()
            w = fit(build(H[tr], tr, zr, mode), zr, y[tr], True)
            new[va] = zv + build(H[va], va, zv, mode) @ w
            base[va] = zv
        scores[seed] = {k: wl.score(y, v, wt, wn, miss) for k, v in (('original', base), ('incumbent', incumbent[seed]), ('new', new))}
        audits[seed] = audit(y, miss, base, new)
        oof[f's{seed}'] = new
        print(seed, mode, {k: round(scores[seed]['new'][k], 7) for k in ('auc', 'remaining', 'target_vs_neg', 'weighted')}, flush=True)
    accepted = all((s['new']['auc'] > s['incumbent']['auc'] and s['new']['remaining'] >= s['original']['remaining'] and (s['new']['target_vs_neg'] > s['original']['target_vs_neg']) for s in scores.values()))
    rep = _rebuild.annotate(dict(name=name, models=models, physical=physical, mode=mode, scores=scores, audits=audits, accepted=accepted))
    (OUT / f'wn_{name}.json').write_text(json.dumps(rep, indent=2))
    np.savez_compressed(OUT / f'wn_{name}_oof.npz', ids=ids.to_numpy(dtype=str), **oof)
    if _rebuild.export(accepted) and len(seeds) == 6:
        fi, ft, M, mask, yy = load('physical')
        assert ids.equals(fi) and tids.equals(ft) and np.array_equal(y, yy)
        ix = np.arange(n)
        te = n + np.arange(len(tids))
        h, priors = fit_on(M, mask, y, ix)
        w = fit(build(h[ix], ix, q['zr_all'], mode), q['zr_all'], y, True)
        F = build(h[te], te, q['zt'], mode)
        folder = ROOT / f'new weights/wn_{name}_candidate'
        folder.mkdir(exist_ok=True)
        file = folder / 'submission.csv'
        recipe = folder / 'weights.json'
        sub = wl.ef.write_submission(expit(q['zt'] + F @ w), tids, file)
        inputs = [Path(__file__).resolve(), wl.PARENT, *paths]
        recipe.write_text(json.dumps(dict(name=name, models=models, weights=w.tolist(), priors=priors, hashes={str(p.relative_to(ROOT)): wl.ef.sha(p) for p in inputs}), indent=2))
        np.savez_compressed(folder / 'replay_inputs.npz', parent_logits=q['zt'], features=F, ids=tids.to_numpy(dtype=str), order=tids.get_indexer(sub.id))
        (folder / 'validation.json').write_text(json.dumps(dict(**rep, recipe_sha256=wl.ef.sha(recipe), submission_sha256=wl.ef.sha(file)), indent=2))
        shutil.copyfile(wl.PARENT.parent / 'replay.py', folder / 'replay.py')
    print(name, accepted, audits.get(7), flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('base', 'short', 'full'))
    p.add_argument('--extra', nargs='*', default=[])
    p.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    p.add_argument('--tag', default='physics')
    p.add_argument('--physical', nargs='*', choices=('volatility', 'boundary'), default=[])
    a = p.parse_args()
    with threadpool_limits(limits=1):
        run(a.mode, a.extra, a.seeds, a.tag, a.physical)
