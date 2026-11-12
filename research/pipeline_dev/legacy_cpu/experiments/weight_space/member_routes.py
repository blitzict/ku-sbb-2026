# CPU port; original experiments/weight_space/member_routes.py, SHA256 af2e77fde38875953bf21c6fb794f02ae4aa3f251661fb525bfcf94a0d596bb7
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Evaluate and replay a process-dependent weight edit on archived expert logits."""
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
from residual_trial import baseline, ANCHOR
MEMBERS = ('MLPPAIR_bag_w', 'VARC4hv2_ns3_wc', 'W_SEQdz_q', 'VARC4ic_ns3_wc', 'VARC4a2_ns3_wc', 'W_v6_dd_cat', 'v6_dd_lgb', 'v6_dd_mlp', 'v6_gh_xlg')
SEG = ef.O / 'generator_segments.parquet'

def members(source, ids, tids):
    paths = []
    if source == 'archive':
        sides = []
        for prefix, index in (('oof', ids), ('test', tids)):
            cols = []
            for name in MEMBERS:
                path = ef.O / f'{prefix}_{name}.npy'
                p = np.load(path)
                assert p.shape == (len(index),) and np.all((p >= 0) & (p <= 1)), path
                cols.append(logit(np.clip(p, 1e-06, 1 - 1e-06)))
                paths.append(path)
            sides.append(np.column_stack(cols))
        return (np.vstack(sides), paths)
    for name in source.split('+'):
        assert name.startswith('seed') and name[4:].isdigit()
        paths.append(ef.OUT / 'projected_member' / f'{name}.npz')
    arrays = []
    for path in paths:
        with np.load(path) as a:
            assert np.array_equal(a['ids'], ids) and np.array_equal(a['tids'], tids)
            arrays.append(np.r_[a['oof'], a['test']])
    return (np.mean(arrays, axis=0)[:, None], paths)

def context(ids, tids):
    cols = ['id', 'part', 'p_bic', 'phi1_ar1', 't_df', 'skew', 'kurt', 'n_pre', 'n_post', 'sqacf1', 'innov_sd4', 'spec_peak_share']
    t = pd.read_parquet(SEG, columns=cols, filters=[('part', '==', 'pre')]).set_index('id')
    assert t.index.is_unique and set(t.index) == set(ids) | set(tids)
    t = t.loc[ids.append(tids)]
    x = np.column_stack([t.p_bic.eq(0), t.phi1_ar1.lt(-0.2), t.p_bic.ge(7), t.phi1_ar1.gt(0.92), t.t_df.le(6), t['kurt'].lt(-0.5), t['skew'].abs().gt(0.5), t.sqacf1.gt(0.18), np.log(t.n_post / t.n_pre), np.log(t.innov_sd4), t.spec_peak_share])
    assert np.isfinite(x).all()
    return x

def design(mode, member, c, z):
    d = np.clip(member - z[:, None], -5, 5)
    base = np.tanh(z / 2)[:, None]
    f = [d, d * base]
    if mode in ('family', 'combined'):
        f.extend([c, c * base, (d[:, :, None] * c[:, None, :]).reshape(len(z), -1)])
    if mode == 'family_only':
        f = [c, c * base]
    if mode == 'combined':
        f.extend([np.maximum(d - 1, 0), np.maximum(-d - 1, 0)])
    return np.column_stack(f)

def main(source, replay=None, output=None):
    data = ef.read_data()
    _, ids, tids, y, wt, *_ = data
    member, paths = members(source, ids, tids)
    c = context(ids, tids)
    assert member.shape[0] == len(ids) + len(tids) and np.isfinite(member).all()
    tag = 'routes_' + source.replace('+', '_')
    paths += [SEG, ANCHOR, Path(__file__).resolve()]
    if replay:
        art = json.loads(replay.read_text())
        assert source == art['source']
        for path, expected in art['hashes'].items():
            assert ef.sha(ef.ROOT / path) == expected, path
        parent = pd.read_csv(ANCHOR.parent / 'submission.csv', dtype={'id': str}).set_index('id')
        z = logit(parent.loc[tids, 'break_probability'].to_numpy())
        F = design(art['mode'], member[len(y):], c[len(y):], z)
        th = np.array(art['weights'])
        ef.write_submission(expit(z + F @ th[:-1] + th[-1]), tids, output)
        return
    modes = ['weighted', 'family', 'family_only', 'combined']
    folds, zall = baseline(data)
    results, archive = ({}, {})
    for split, parts in folds.items():
        pred = {k: np.empty(len(y)) for k in ['anchor', *modes]}
        for tr, va, zr, zv in parts:
            pred['anchor'][va] = zv
            for mode in modes:
                th = ef.fit_head(design(mode, member[tr], c[tr], zr), zr, y[tr])
                pred[mode][va] = zv + design(mode, member[va], c[va], zv) @ th[:-1] + th[-1]
        results[split] = {k: ef.score(y, p, wt) for k, p in pred.items()}
        archive.update({f's{split}_{k}': p for k, p in pred.items()})
        print(split, results[split], flush=True)
    eligible = [k for k in modes if all((s[k][m] - s['anchor'][m] >= 0.0003 for s in results.values() for m in ('auc', 'weighted_auc')))]
    chosen = max(eligible, key=lambda k: np.mean([s[k]['auc'] for s in results.values()])) if eligible else None
    report = {'source': source, 'selected': chosen, 'scores': results}
    (ef.OUT / f'{tag}_results.json').write_text(json.dumps(report, indent=2))
    np.savez(ef.OUT / f'{tag}_oof.npz', ids=ids.to_numpy(dtype=str), **archive)
    if chosen:
        dest = ef.PACKAGE / f'{tag}_candidate'
        dest.mkdir(exist_ok=True)
        z = zall[:len(y)]
        theta = ef.fit_head(design(chosen, member[:len(y)], c[:len(y)], z), z, y)
        paths.append(ANCHOR.parent / 'submission.csv')
        hashes = {str(p.relative_to(ef.ROOT)): ef.sha(p) for p in paths}
        recipe = dest / 'weights.json'
        recipe.write_text(json.dumps({**report, 'mode': chosen, 'weights': theta.tolist(), 'hashes': hashes}, indent=2))
        main(source, recipe, dest / 'submission.csv')
    print('Selected', chosen, flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('source')
    p.add_argument('--replay', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    with threadpool_limits(limits=2):
        main(args.source, args.replay, args.output)
