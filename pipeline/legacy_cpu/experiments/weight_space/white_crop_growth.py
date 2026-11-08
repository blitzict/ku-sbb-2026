# CPU port; original experiments/weight_space/white_crop_growth.py, SHA256 702588e98dcc2228cb8c2952a999de58b83daf4251ea006cd56c2f1d97c0845a
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Measure how the evidence for individual breaks grows with real observation time."""
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
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.special import expit
from threadpoolctl import threadpool_limits
from white_rank_boost import rank_objective
from partition_evidence import ROOT, OUT
import white_loop as wl
LAWS = {'volatility_law': 'new_memory new_law new_scale memory_on memory_off scale_jump1 scale_jump3', 'mixture_law': 'm_bf d_bf s_bf p_bf all_bf', 'periodic': 'phase_gain freq_gain amplitude_change phase_change common_level separate_scale', 'curve': 'obs_step slope_step global_linear global_quad mixture adjusted_level adjusted_scale', 'ar_law': 'level_bf scale_bf memory_bf white_level walk_level white_scale walk_scale', 'noise_tests': 'lr_level_lbf_t lr_scale_lbf_t lr_gn_shape_lr_sym lr_bim_shape_lr_sym lr_ar_lr', 'innov_v2': 'dynamics_bf walk_reset shared_level shared_scale shared_shape'}
for src in ('state', 'pulse', 'smoothvol', 'prestate', 'nullstate'):
    LAWS['jump_law_' + src] = 'level0 level2 level4 scale0 scale2 scale4 shape0 shape1'

def view(part, ix, v):
    d = OUT / ('view_' + v) if v else OUT
    clean = d / 'clean' if v else ROOT / 'clean'
    n = pd.read_parquet(clean / f'features_{part}.parquet', columns=['n_pre', 'n_post']).loc[ix].to_numpy()
    raw = []
    for name, cols in LAWS.items():
        a = pd.read_parquet(d / f'{name}_{part}.parquet', columns=cols.split()).loc[ix]
        raw.append(a.to_numpy(dtype=np.float32))
    return (n, np.column_stack(raw).clip(-35, 100))

def compile(part, ix, views):
    n, x = view(part, ix, '')
    ns, vs = zip(*(view(part, ix, v) for v in views))
    h = 1 / (1 / n[:, 0] + 1 / n[:, 1])
    ratios = np.stack([1 / (1 / d[:, 0] + 1 / d[:, 1]) / h for d in ns], 1).clip(0.05, 1)
    lens = np.stack([d / n for d in ns], 1)
    values = np.stack(vs, 1)
    decay = 1 - ratios[:, :, None]
    z = np.arcsinh(x)
    v = np.arcsinh(values)
    remainder = v - np.arcsinh(x[:, None, :] * ratios[:, :, None])
    drop = z[:, None, :] - v
    slope = (drop * decay).sum(1) / (0.09 + (decay ** 2).sum(1))
    side = (lens[:, :, 0] - lens[:, :, 1])[:, :, None]
    growth = np.column_stack([remainder.mean(1), remainder.std(1), slope, (drop * side).sum(1) / (0.12 + (side ** 2).sum(1)), (v > 0).mean(1), np.quantile(v, 0.25, axis=1), np.quantile(v, 0.75, axis=1), ratios, lens.reshape(len(ix), -1)])
    return (z.astype(np.float32), growth.astype(np.float32))

def run(views):
    q, ids, tids, y, _, _, _ = wl.load()
    tag = 'lawgrowth_' + '_'.join(views)
    raw = [compile(p, ix, views) for p, ix in (('train', ids), ('test', tids))]
    with np.load(ROOT / 'tune_out/var/mat_ns3_c7.npz', allow_pickle=True) as m:
        assert np.array_equal(ids, m['ids'].astype(str)) and np.array_equal(tids, m['tid'].astype(str))
        base = [np.column_stack([m[k], r[0]]).astype(np.float32) for k, r in zip(('X', 'Xt'), raw)]
    white = pd.read_parquet(OUT / 'innov_v2_train.parquet', columns=['pre_white']).loc[ids].to_numpy()[:, 0]
    pred = {}
    dest = OUT / (tag + '_models')
    dest.mkdir(exist_ok=True)
    for mode in ('control', 'growth'):
        oof = np.empty(len(y))
        tests = []
        for f, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(ids, y)):
            assert np.array_equal(q[f's0_f{f}_va'], va)
            law = OUT / 'law_outer0'
            with np.load(law / f'fold{f}_rows.npz') as a:
                assert np.array_equal(a['validation_ids'], ids[va])
            features = []
            for p, ix, b, r in zip(('train', 'test'), (ids, tids), base, raw):
                parts = [b, pd.read_parquet(law / f'fold{f}_{p}.parquet').loc[ix].to_numpy()]
                if mode == 'growth':
                    parts.append(r[1])
                features.append(np.column_stack(parts).astype(np.float32))
            x, t = features
            ck = dest / f'{mode}{f}.txt'
            if ck.exists():
                model = lgb.Booster(model_file=str(ck))
            else:
                objective = rank_objective(y[tr], np.ones(len(tr)), white[tr])
                model = lgb.train(dict(objective=objective, metric='None', learning_rate=0.036, num_leaves=27, max_depth=6, min_data_in_leaf=80, lambda_l2=15.0, feature_fraction=0.66, verbosity=-1, deterministic=True, force_col_wise=True, num_threads=2, max_bin=66, seed=615 + f), lgb.Dataset(x[tr], label=y[tr]), num_boost_round=620)
                model.save_model(str(ck))
            oof[va] = expit(model.predict(x[va], raw_score=True, num_threads=2))
            tests.append(expit(model.predict(t, raw_score=True, num_threads=2)))
            print(tag, mode, f, roc_auc_score(y[va], oof[va]), flush=True)
        suffix = '_control' if mode == 'control' else ''
        np.savez_compressed(OUT / (tag + suffix + '.npz'), ids=ids.to_numpy(dtype=str), tids=tids.to_numpy(dtype=str), oof=oof, test=np.mean(tests, 0))
        pred[mode] = oof
    for mode, v in pred.items():
        print(mode, roc_auc_score(y, v), flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('views', nargs='+')
    with threadpool_limits(limits=2):
        run(p.parse_args().views)
