# CPU port; original experiments/weight_space/white_segment_population.py, SHA256 9b110526337a764fde7fbd50a5ce50595abdfbcfb1911448057b056588e07b42
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Which real segment properties are more typical after a labelled break?"""
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
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from partition_evidence import ROOT, OUT
import white_loop as wl
SEG = 'n p_bic p_aic phi1 phi2 phi3 phi4 phi1_ar1 persist4 innov_sd4 mean sd skew kurt tail3 tail2 q_ratio racf1 racf2 racf3 racf5 racf10 sqacf1 sqacf2 sqacf5 t_df ll_t_minus_gauss spec_peak_share spec_peak_freq half_mean_diff'.split()
QT = np.array([0.02, 0.07, 0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85, 0.93, 0.98])

def properties(v):
    a = v - v.mean()
    design = np.column_stack([a[1:-1], a[:-2]])
    ar = np.linalg.solve(design.T @ design + np.eye(2) * (0.6 * a.var() + 1e-06), design.T @ a[2:])
    arr = [len(v), *ar]
    for u in (v, a[2:] - design @ ar, np.diff(v)):
        q = np.quantile(u, QT)
        scale = max((q[7] - q[3]) / 1.349, 0.15 * u.std(), 0.0001)
        z = ((u - q[5]) / scale).clip(-8, 8)
        arr.extend([q[5], np.log(scale), *list((q - q[5]) / scale)])
        for w in (np.tanh(z), np.log(0.22 + abs(z)), np.sign(z) * np.log1p(abs(z))):
            zz = w - w.mean()
            var = zz @ zz + 1e-07
            arr.extend([w.mean(), w.std(), *(zz[k:] @ zz[:-k] / var for k in (1, 2, 3, 5, 8, 13))])
        arr.extend([np.cos(d * z).mean() for d in (0.7, 1.5, 2.6, 3.8, 5.2)])
        arr.extend([np.sin(d * z).mean() for d in (0.7, 1.5, 2.6, 3.8, 5.2)])
    return arr

def prepare(part, ix):
    cache = OUT / f'individual_segment_chart_{part}.npz'
    if cache.exists():
        with np.load(cache) as a:
            assert np.array_equal(ix, a['ids'])
            return a['stats']
    d = pd.read_parquet(ROOT / 'tune_out/generator_segments.parquet', columns=['id', 'part', *SEG])
    single = [d[d.part == side].set_index('id').loc[ix, SEG].to_numpy(np.float32) for side in ('pre', 'post')]
    with np.load(ROOT / f'tune_out/seq_cache_{part}.npz', allow_pickle=True) as a:
        ids = a['ids'].astype(str)
        assert np.array_equal(ids, ix)
        x, b, n = (a[k] for k in ('x', 'b', 'n'))
    raw = np.array([[properties(z[:j].astype(float)), properties(z[j:end].astype(float))] for z, j, end in zip(x, b, n)], np.float32)
    out = np.concatenate([np.stack(single, 1), raw], 2)
    out = np.nan_to_num(out, nan=0.0, posinf=1000.0, neginf=-1000.0).clip(-1000.0, 1000.0)
    np.savez_compressed(cache, ids=ids, stats=out)
    print('individual segments', part, out.shape, flush=True)
    return out

def evidence(model, x):
    n = len(x)
    v = model.predict(x.reshape(2 * n, -1), raw_score=True, num_threads=2).reshape(n, 2, 4)
    prior = v[:, :, 2] - v[:, :, 0]
    after = v[:, :, 3] - v[:, :, 1]
    stage = v[:, :, 3] - v[:, :, 2] - (v[:, :, 1] - v[:, :, 0])
    return np.column_stack([prior[:, 0], after[:, 1], stage[:, 1], stage[:, 0], stage[:, 1] - stage[:, 0], after[:, 1] - after[:, 0], v[:, 1, 3] - v[:, 1, 2] - (v[:, 0, 3] - v[:, 0, 2]), v[:, 1, 1] - v[:, 1, 0] - (v[:, 0, 1] - v[:, 0, 0]), logsumexp(v[:, 1, 2:], axis=1) - logsumexp(v[:, 1, :2], axis=1)])

def main():
    _, ids, tids, y, wt, wn, miss = wl.load()
    x = prepare('train', ids)
    t = prepare('test', tids)
    dest = OUT / 'segment_population_models'
    dest.mkdir(exist_ok=True)
    pt = []
    oof = np.empty((len(ids), 9))
    for f, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(ids, y)):
        ck = dest / f'fold{f}.txt'
        if ck.exists():
            model = lgb.Booster(model_file=str(ck))
        else:
            target = 2 * y[tr, None] + np.array([0, 1])[None, :]
            params = dict(objective='multiclass', num_class=4, num_leaves=15, max_depth=5, min_data_in_leaf=155, feature_fraction=0.68, learning_rate=0.048, lambda_l2=23.0, max_bin=79, num_threads=2, deterministic=True, force_col_wise=True, seed=201 + f, verbosity=-1)
            model = lgb.train(params, lgb.Dataset(x[tr].reshape(2 * len(tr), -1), label=target.ravel()), num_boost_round=360)
            model.save_model(str(ck))
        oof[va] = evidence(model, x[va])
        pt.append(evidence(model, t))
    cols = 'pre_propensity post_propensity post_specificity pre_specificity orientation post_change pos_forward neg_forward post_unassigned'.split()
    for part, idx, v in (('train', ids, oof), ('test', tids, np.mean(pt, 0))):
        pd.DataFrame(v, index=idx, columns=cols).rename_axis('id').to_parquet(OUT / f'segment_population_{part}.parquet')
    for j, name in enumerate(cols[:6]):
        print(name, [round(roc_auc_score(y[g], oof[g, j]), 5) for g in (np.ones(len(y), bool), ~miss, miss | (y == 0))], flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
