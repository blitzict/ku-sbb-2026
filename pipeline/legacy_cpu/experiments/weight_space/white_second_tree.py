# CPU port; original experiments/weight_space/white_second_tree.py, SHA256 480267b225335f353140d4118727bf5582ce9139618cd596faf88e861e0d13be
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Train an additional fold-local physical tree, with a matched residual head."""
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
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from shape_bank import matrix
from white_mech_tree import extra
from white_checkpoint import CACHE
from partition_evidence import OUT, ROOT
import white_loop as wl

def input_frame(part, index, more):
    src = matrix(part, 'state').loc[index]
    blocks = [src, extra(part, index, 'both')]
    with np.load(ROOT / 'tune_out/var/mat_ns3_plain.npz', allow_pickle=True) as a:
        assert np.array_equal(a['ids' if part == 'train' else 'tid'], index)
        blocks.append(pd.DataFrame(a['X' if part == 'train' else 'Xt'], index=index).add_prefix('ns3_'))
    tables = ['volatility_law', 'hetero_jump', 'walk_jump', 'trend_population', 'memory_law1', 'memory_law2', 'conditional_scores', 'boundary_law_state', 'boundary_law_pulse', 'vol_smoothing', *more]
    for t in tables:
        tab = pd.read_parquet(OUT / f'{t}_{part}.parquet').loc[index]
        blocks.append(tab.add_prefix(t + '_'))
    df = pd.concat(blocks, axis=1).astype(np.float32)
    assert df.columns.is_unique
    return df

def run(name, more):
    q, ids, tids, y, *_ = wl.load()
    cp = np.load(CACHE)
    aa, bb = (input_frame('train', ids, more), input_frame('test', tids, more))
    assert list(aa.columns) == list(bb.columns)
    oof, orr = (np.empty(len(y)) for _ in range(2))
    tests = []
    res = []
    dest = OUT / (name + '_models')
    dest.mkdir(exist_ok=True)
    for f, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(ids, y)):
        k = f's0_f{f}_'
        assert np.array_equal(q[k + 'va'], va) and np.array_equal(q[k + 'tr'], tr)
        prior = OUT / 'law_outer0'
        rows = np.load(prior / f'fold{f}_rows.npz')
        assert np.array_equal(rows['validation_ids'], ids[va])
        x, t = [pd.concat([base, pd.read_parquet(prior / f'fold{f}_{p}.parquet').loc[ix].add_prefix('fold_prior_')], axis=1) for base, p, ix in ((aa, 'train', ids), (bb, 'test', tids))]
        if f == 0:
            (dest / 'columns.json').write_text(json.dumps(list(x.columns)))
        for mode in ('standalone', 'residual'):
            path = dest / f'{mode}{f}.txt'
            if path.exists():
                model = lgb.Booster(model_file=str(path))
            else:
                if mode == 'standalone':
                    params = dict(learning_rate=0.03, max_depth=6, num_leaves=35, min_data_in_leaf=73, lambda_l2=18.0, feature_fraction=0.62)
                    ds = lgb.Dataset(x.iloc[tr], label=y[tr])
                    rounds = 850
                else:
                    params = dict(learning_rate=0.026, max_depth=3, num_leaves=8, min_data_in_leaf=150, lambda_l2=54.0, feature_fraction=0.55, boost_from_average=False)
                    z = cp[k + 'zr']
                    wt = 0.45 + 1.0 * np.exp(-0.7 * abs(z))
                    ds = lgb.Dataset(x.iloc[tr], label=y[tr], init_score=z, weight=wt)
                    rounds = 200
                params.update(objective='binary', verbosity=-1, num_threads=2, max_bin=80, deterministic=True, force_col_wise=True, seed=6409 + f)
                model = lgb.train(params, ds, num_boost_round=rounds)
                model.save_model(str(path))
            v = model.predict(x.iloc[va], raw_score=True, num_threads=2)
            u = model.predict(t, raw_score=True, num_threads=2)
            if mode == 'standalone':
                oof[va] = expit(v)
                tests.append(expit(u))
            else:
                orr[va] = expit(cp[k + 'zv'] + v)
                res.append(u)
        print(name, f, 'direct and residual', roc_auc_score(y[va], oof[va]), roc_auc_score(y[va], orr[va]), flush=True)
    for tag, p, pt in (('', oof, np.mean(tests, 0)), ('_res', orr, expit(cp['zt'] + np.mean(res, 0)))):
        np.savez_compressed(OUT / (name + tag + '.npz'), ids=ids.to_numpy(dtype=str), tids=tids.to_numpy(dtype=str), oof=p, test=pt)
        print(name + tag, 'OOF', roc_auc_score(y, p), flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('name')
    p.add_argument('--tables', nargs='*', default=[])
    a = p.parse_args()
    with threadpool_limits(limits=2):
        run(a.name, a.tables)
