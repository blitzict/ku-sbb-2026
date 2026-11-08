# Fixed winning-recipe CPU adaptation v1
"""Measure whether repeated evidence in boundary-preserving crops predicts a real break."""
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
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from white_checkpoint import SEEDS
from white_joint import read_member
from white_nested import fit, audit
from white_local_rank_weights import centered
from partition_evidence import ROOT, OUT
import white_loop as wl
BASE = 'local_rank_three_source_votes'
PREVIOUS = 'local_rank_early_mode_peaks'

def crop_scores(ids, tids):
    archive = ROOT / 'tune_out'
    order = archive / 'var/mat_ns3_plain.npz'
    with np.load(order, allow_pickle=True) as a:
        assert np.array_equal(a['ids'], ids) and np.array_equal(a['tid'], tids)
    paths = []

    def group(tags, views):
        scores = []
        for tag in tags:
            p = archive / f'ver_{tag}.npz'
            paths.append(p)
            with np.load(p) as a:
                assert a['kinds'].tolist() == views
                vals = np.hstack([a['oof'], a['test']]).T.astype(float)
                assert vals.shape == (len(ids) + len(tids), len(views))
            v = logit(vals.clip(1e-06, 1 - 1e-06))
            if len(views) == 3:
                weights = np.array([2.0, 0.125, 0.125])
                ref = np.r_[np.load(archive / f'oof_{tag}.npy'), np.load(archive / f'test_{tag}.npy')]
                assert np.max(abs(expit(v @ weights / weights.sum()) - ref)) < 2e-06
            scores.append(v)
        return np.mean(scores, axis=0)
    names = ['orig', 'c1', 'c2']
    initial = group([f'MECH3C12V_s{i}' for i in range(3)], names)
    panels = np.stack([initial, group(['MECH3C12V_k29_s0'], names), group(['MECH3C12V_k7_s0'], names)])
    harmonic = group([f'VARC4harm_ns3_s{i}' for i in range(2)], ['orig', 'rev', 'flip', 'revflip', 'c1', 'c2', 'c3', 'c4'])
    return (panels, harmonic, paths)

def run(mode):
    q, ids, tids, y, wt, wn, misses = wl.load()
    n = len(y)
    parent = ROOT / f'new weights/wn_{BASE}_candidate'
    spec = json.loads((parent / 'weights.json').read_text())
    assert spec['method'] == 'physical'
    paths = [parent / 'weights.json', Path(__file__).resolve()]
    pieces = []
    for m in spec['members']:
        x, p = read_member(m, ids, tids)
        pieces.append(x[:, 0])
        paths.extend(p)
    experts = np.column_stack(pieces)

    def table(tag, cols=None):
        parts = []
        for s, rows in (('train', ids), ('test', tids)):
            path = OUT / f'{tag}_{s}.parquet'
            paths.append(path)
            parts.append(pd.read_parquet(path, columns=cols).loc[rows])
        return pd.concat(parts)
    white = table('innov_v2', ['pre_white']).pre_white.to_numpy()
    panels, harmonic, pp = crop_scores(ids, tids)
    paths.extend(pp)
    original, c1, c2 = panels.mean(0).T
    diff = np.stack([c1 - original, c2 - original], axis=1)
    drift = diff.mean(1)
    spread = abs(c1 - c2)
    stable = np.exp(-0.5 * ((spread / 1.4) ** 2 + (drift / 1.8) ** 2))
    lengths = []
    for view in ('', 'c1', 'c2'):
        p = []
        for part, ix in (('train', ids), ('test', tids)):
            src = ROOT / f'tune_out/var/clean_{part}_{view}.parquet' if view else ROOT / f'clean/features_{part}.parquet'
            paths.append(src)
            tab = pd.read_parquet(src, columns=['n_pre', 'n_post'])
            a, b = tab.loc[ix + view].to_numpy().T
            p.append(a * b / (a + b))
        lengths.append(np.concatenate(p))
    ratios = np.stack(lengths[1:]).T / lengths[0][:, None]
    assert np.all((ratios > 0) & (ratios <= 1.0001))
    fields = np.column_stack([drift, spread, np.min(diff, 1), panels[:, :, 0].std(0), (diff * (1 - ratios)).sum(1) / (0.15 + (1 - ratios).sum(1)), np.mean(harmonic[:, 4:], 1) - harmonic[:, 0], np.std(harmonic[:, 4:], 1), np.mean(harmonic[:, 1:4], 1) - harmonic[:, 0], np.std(harmonic[:, :4], 1)])
    crop = np.tanh(fields / 2)
    cols = [f'{s}_{law}_{measure}' for s in ('state', 'pulse') for law, measure in (('location', 'replicate'), ('width', 'replicate'), ('shoulders', 'after_location_scale'), ('tails', 'after_location_scale'), ('all_knots', 'after_location_scale'))]
    evidence = np.arcsinh(table('quantile_alignment', cols).to_numpy().clip(-24, 55))
    if mode == 'replicated_directions':
        import white_replicated_directions as rd
        signed, new_paths = rd.data(ids, tids)
        paths.extend([*new_paths, Path(rd.__file__).resolve()])
        pre = table('volatility_law', ['pre_cluster']).to_numpy()[:, 0]
        family = np.column_stack([np.ones(len(white)), white, 1 - white, pre])
    if mode in ('expert_conflict', 'conflict_rule'):
        conflict_names = ['volatility_law', 'phase_law', 'trend', 'bounds', 'wnlaw', 'ar_law', 'ns3', 'jump_law_smoothvol']
        conflict = table('tree_channels_claude', ['full_' + c for c in conflict_names]).to_numpy()

    def features(i, z):
        c = crop[i]
        near = np.exp(-0.5 * ((z + 1.0) / 1.25) ** 2)
        if mode == 'crop':
            return np.column_stack([c, c * white[i, None], c * near[:, None], c * np.tanh(z[:, None] / 2)])
        gate = white[i] * near * stable[i]
        return evidence[i] * gate[:, None]

    def baseline(tr, zr, i, z):
        a, b, c = np.quantile(zr, [0.25, 0.5, 0.75])
        lo, med, hi = np.quantile(experts[tr], [0.25, 0.5, 0.75], axis=0)
        cal = (experts[i] - med) * ((c - a) / (hi - lo)) + b
        gate = 1.0 - 0.85 * white[i] * np.exp(-0.5 * ((z + 1.1) / 1.25) ** 2)
        return z + (cal - z[:, None]) * gate[:, None] @ np.asarray(spec['weights'])

    def correction(tr, z, i, v):
        if mode in ('expert_conflict', 'conflict_rule'):

            def chart(ix, score):
                regions = np.digitize(white[ix], [0.65, 0.85])
                tr_regions = np.digitize(white[tr], [0.65, 0.85])
                standardized = np.empty((len(ix), conflict.shape[1]))
                for reg in range(3):
                    neg = (y[tr] == 0) & (tr_regions == reg)
                    h = conflict[tr[neg]]
                    hx, mean = centered(h, z[neg])
                    out, _ = centered(conflict[ix[regions == reg]], score[regions == reg], mean)
                    sd = np.sqrt((hx * hx).mean(0) + 0.05 ** 2)
                    standardized[regions == reg] = np.tanh(out / (1.4 * sd))
                a = standardized
                agree = np.minimum(a[:, 0], a[:, 1])
                doubt = -np.mean(a[:, 4:7], axis=1)
                pair = np.minimum(np.maximum(0, agree), np.maximum(0, doubt))
                pair -= np.minimum(np.maximum(0, -np.maximum(a[:, 0], a[:, 1])), np.maximum(0, -doubt))
                gate = white[ix] * np.exp(-0.5 * ((score + 1.15) / 1.4) ** 2)
                if mode == 'conflict_rule':
                    return (gate * pair)[:, None]
                raw = np.column_stack([a, agree, doubt, pair, a[:, 0] * doubt, a[:, 1] * doubt, a[:, 2] * a[:, 3]])
                return np.column_stack([raw, raw * gate[:, None]])
            tx, vx = (chart(tr, z), chart(i, v))
            if mode == 'conflict_rule':
                return (vx, np.array([0.12]))
            tx, means = centered(tx, z)
            vx, _ = centered(vx, v, means)
            return (vx, fit(tx, z, y[tr], True))
        if mode == 'replicated_directions':
            out = rd.measure(signed, family, y, tr, i)
            sub = np.empty((len(tr), out.shape[1]))
            for a, b in StratifiedKFold(3, shuffle=True, random_state=5931).split(tr, y[tr]):
                sub[b] = rd.measure(signed, family, y, tr[a], tr[b])
            tx, means = centered(sub, z)
            vx, _ = centered(out, v, means)
            return (vx, fit(tx, z, y[tr], True))
        tx, means = centered(features(tr, z), z)
        vx, _ = centered(features(i, v), v, means)
        w = fit(tx, z, y[tr], True)
        return (vx, w)
    tag = 'local_rank_crop_replication_' + mode
    cache = OUT / ('frozen_' + PREVIOUS)
    scores, pred, detail = ({}, {}, {})
    with np.load(OUT / (BASE + '_oof.npz')) as ref:
        assert np.array_equal(ref['ids'], ids)
        for seed in SEEDS:
            old, new = (np.empty(n), np.empty(n))
            for f in range(5):
                k = f's{seed}_f{f}_'
                tr, va = (q[k + 'tr'], q[k + 'va'])
                with np.load(cache / f's{seed}_f{f}.npz') as a:
                    assert np.array_equal(a['train_ids'], ids[tr]) and np.array_equal(a['validation_ids'], ids[va])
                    zr, zv = (a['oof_train'], a['valid'])
                z, v = (baseline(tr, zr, tr, zr), baseline(tr, zr, va, zv))
                assert np.max(abs(v - ref[f's{seed}'][va])) < 1e-09
                F, w = correction(tr, z, va, v)
                old[va], new[va] = (v, v + F @ w)
            scores[seed] = {k: wl.score(y, t, wt, wn, misses) for k, t in (('previous', old), ('new', new))}
            detail[seed] = audit(y, misses, old, new)
            pred[f's{seed}'] = new
            print(mode, seed, {k: round(scores[seed]['new'][k] - scores[seed]['previous'][k], 7) for k in ('auc', 'target_vs_neg', 'remaining')}, flush=True)
    passed = all((v['new']['auc'] > v['previous']['auc'] for v in scores.values())) and all((scores[7]['new'][k] >= scores[7]['previous'][k] for k in ('remaining', 'target_vs_neg')))
    result = _rebuild.annotate(dict(parent=BASE, mode=mode, accepted=bool(passed), scores=scores, audits=detail))
    (OUT / (tag + '.json')).write_text(json.dumps(result, indent=2))
    np.savez_compressed(OUT / (tag + '_oof.npz'), ids=ids.to_numpy(dtype=str), **pred)
    if _rebuild.export(passed):
        tr, te = (np.arange(n), n + np.arange(len(tids)))
        with np.load(cache / 'full.npz') as a:
            assert np.array_equal(a['train_ids'], ids) and np.array_equal(a['test_ids'], tids)
            zr, zv = (a['oof_train'], a['test'])
        v = baseline(tr, zr, te, zv)
        p = pd.read_csv(parent / 'submission.csv', dtype={'id': str}).set_index('id').loc[tids, 'break_probability']
        assert np.max(abs(expit(v) - p)) < 1e-12
        F, w = correction(tr, baseline(tr, zr, tr, zr), te, v)
        dst = ROOT / f'new weights/wn_{tag}_candidate'
        dst.mkdir(exist_ok=True)
        rec, csv = (dst / 'weights.json', dst / 'submission.csv')
        sub = wl.ef.write_submission(expit(v + F @ w), tids, csv)
        rec.write_text(json.dumps(dict(weights=w.tolist(), mode=mode, hashes={str(p.relative_to(ROOT)): wl.ef.sha(p) for p in dict.fromkeys(paths)}), indent=2))
        np.savez_compressed(dst / 'replay_inputs.npz', ids=tids.to_numpy(dtype=str), order=tids.get_indexer(sub.id), parent_logits=v, features=F)
        (dst / 'validation.json').write_text(json.dumps(dict(**result, recipe_sha256=wl.ef.sha(rec), submission_sha256=wl.ef.sha(csv)), indent=2))
        shutil.copyfile(parent / 'replay.py', dst / 'replay.py')
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['crop', 'white_density', 'replicated_directions', 'expert_conflict', 'conflict_rule'])
    with threadpool_limits(limits=1):
        run(p.parse_args().mode)
