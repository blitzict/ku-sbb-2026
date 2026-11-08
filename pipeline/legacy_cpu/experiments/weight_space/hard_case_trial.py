# CPU port; original experiments/weight_space/hard_case_trial.py, SHA256 fda223f3308c69e34c98e76dc1ba33b28ce4394e6b914f45c53c52f472b70c6b
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Paired full-fold test of the two proposed mechanisms on the frozen 0.8705 chain."""
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
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import expert_fusion as ef
import extend_readout as er
import mechanism_edit as me
from hard_pair_budget import STEM, cohorts
from physical_870 import fit
PARENT = ef.PACKAGE / f'{STEM}_candidate/weights.json'
PENALTY = 80.0
MODES = ('ar', 'shape', 'both')
TAGS = ['hard_signals', 'state', 'periodic', 'curve']

def design(t, z, ix, mode):
    h, s, p, c = (t[k].iloc[ix] for k in TAGS)
    wave = p.gate.to_numpy() * expit((p.period.to_numpy() - 9) / 1.5) * expit((38 - p.period.to_numpy()) / 2)
    memory = np.maximum(abs(s.pre_ar1.to_numpy()), abs(s.pre_ar2.to_numpy()))
    g = expit((memory - 0.2) / 0.055) * (1 - wave) * (1 - s.walk.to_numpy())
    w = expit((0.15 - memory) / 0.045) * (1 - wave) * (1 - c.gate.to_numpy())
    out = []
    if mode in ('ar', 'both'):
        H = h.filter(regex='^ar').to_numpy().clip(-15, 40)
        a = h.ar1_joint_scale.to_numpy().clip(-10, 25)
        b = h.ar2_joint_scale.to_numpy().clip(-10, 25)
        out.append(g[:, None] * np.column_stack([np.ones(len(z)), np.tanh(z / 2), np.arcsinh(H), np.clip(a - z, -8, 10), np.clip(b - z, -8, 10), s.pre2.to_numpy() * (b - a).clip(-12, 12)]))
    if mode in ('shape', 'both'):
        S = h.filter(regex='^(core|tail|skew)_').to_numpy().clip(-7, 7)
        out.append(w[:, None] * np.column_stack([np.ones(len(z)), np.tanh(z / 2), S, S * np.tanh(z[:, None] / 2)]))
    return np.column_stack(out)

def diagnostic(z, y, miss, alarm, families, wt):

    def auc(mask):
        return roc_auc_score(y[mask], z[mask])
    row = ef.score(y, z, wt)
    row.update(missed_vs_negative=auc(miss | (y == 0)), positive_vs_alarm=auc(alarm | (y == 1)), unaffected=auc(~(miss | alarm)), missed_above_median=int((z[miss] > np.median(z)).sum()))
    row['missed_vs_same_family_neg'] = {}
    for f in np.unique(families):
        mask = (families == f) & (miss | (y == 0))
        if miss[mask].sum() >= 20 and (~miss[mask]).any():
            row['missed_vs_same_family_neg'][f] = auc(mask)
    return row

def replay(path, output):
    a = json.loads(path.read_text())
    for name, h in a['hashes'].items():
        assert ef.sha(ef.ROOT / name) == h, name
    data = ef.read_data()
    ids, tids = data[1:3]
    base = pd.read_csv(PARENT.parent / 'submission.csv', dtype={'id': str}).set_index('id').loc[tids, 'break_probability']
    z = logit(base.to_numpy())
    f = design(me.tables(data, TAGS), z, np.arange(len(ids), len(ids) + len(tids)), a['mode'])
    ef.write_submission(expit(z + f @ np.array(a['weights'])), tids, output)

def run():
    data = ef.read_data()
    ids, tids, y, wt = data[1:5]
    history = np.load(ef.OUT / f'{STEM}_oof.npz')
    assert np.array_equal(ids, history['ids'])
    miss, alarm = cohorts(ids, y)
    fam = pd.read_parquet(ef.O / 'generator_segments.parquet', columns=['part', 'id', 'phi1_ar1', 'p_bic'])
    a = fam.loc[fam.part == 'pre'].set_index('id').loc[ids]
    phi, pb = (a.phi1_ar1, a.p_bic)
    family = np.select([phi > 0.95, pb >= 5, pb == 0, phi < -0.2, phi > 0.2], ['RW', 'OSC', 'WN', 'negAR', 'posAR'], 'lowAR')
    folds, full = er.parent(data, ())
    frozen = json.loads(PARENT.read_text())
    mods = er.modules(frozen['spec'])
    _, tables = er.tables(data, mods)
    t = me.tables(data, TAGS)
    scores = {}
    archive = {}
    for seed, parts in folds.items():
        pred = {k: np.full(len(y), np.nan) for k in ('parent', *MODES)}
        for tr, va, zr, zv in parts:
            fr, fv = (er.features(mods, tables, zr, tr), er.features(mods, tables, zv, va))
            w = fit(fr, zr, y[tr], frozen['selected'])
            zr, zv = (zr + fr @ w, zv + fv @ w)
            assert np.max(abs(zv - history[f's{seed}_35.0'][va])) < 1e-07
            pred['parent'][va] = zv
            for mode in MODES:
                X = design(t, zr, tr, mode)
                V = design(t, zv, va, mode)
                co = fit(X, zr, y[tr], PENALTY)
                pred[mode][va] = zv + V @ co
        scores[seed] = {k: diagnostic(v, y, miss, alarm, family, wt) for k, v in pred.items()}
        archive.update({f's{seed}_{k}': v for k, v in pred.items()})
        print(seed, [(k, round(v['auc'], 7), round(v['missed_vs_negative'], 6)) for k, v in scores[seed].items()], flush=True)
    eligible = [mode for mode in MODES if all((s[mode]['auc'] - s['parent']['auc'] >= 0.0002 and all((s[mode][k] > s['parent'][k] for k in ('weighted_auc', 'missed_vs_negative'))) for s in scores.values()))]
    selected = max(eligible, key=lambda k: np.mean([s[k]['auc'] for s in scores.values()])) if eligible else None
    report = dict(penalty=PENALTY, selected=selected, scores=scores)
    np.savez_compressed(ef.OUT / 'hard_case_trial_oof.npz', ids=ids.to_numpy(dtype=str), **archive)
    (ef.OUT / 'hard_case_trial.json').write_text(json.dumps(report, indent=2))
    if selected:
        zr = full + er.features(mods, tables, full, np.arange(len(full))) @ np.array(frozen['weights'])
        tr = np.arange(len(ids))
        theta = fit(design(t, zr[tr], tr, selected), zr[tr], y, PENALTY)
        folder = ef.PACKAGE / f'hard_case_{selected}_candidate'
        folder.mkdir(exist_ok=True)
        files = [PARENT, PARENT.parent / 'submission.csv', Path(__file__).resolve()]
        files.extend((ef.OUT / f'{k}_{p}.parquet' for k in TAGS for p in ('train', 'test')))
        recipe = folder / 'weights.json'
        recipe.write_text(json.dumps(dict(mode=selected, weights=theta.tolist(), penalty=PENALTY, hashes={str(p.relative_to(ef.ROOT)): ef.sha(p) for p in files}, scores=scores), indent=2))
        replay(recipe, folder / 'submission.csv')
    print('selected', selected, flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--replay', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    with threadpool_limits(limits=2):
        if args.replay:
            replay(args.replay, args.output)
        else:
            run()
