# CPU port; original experiments/weight_space/white_mech_tree.py, SHA256 8f6fcbebdddf2601707c44847e546f53697c2c4a63ca92a248cd2000572631e9
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Train a fold-local nonlinear readout of complete intervention likelihoods."""
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
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from partition_evidence import OUT, ROOT
from shape_bank import matrix
from white_shapes import anchors, dirichlet_log

def extra(part, ix, mode):
    blocks = []
    for stream in ('pulse', 'state') if mode == 'both' else ('pulse',):
        blocks.append(pd.read_parquet(OUT / f'jump_law_{stream}_{part}.parquet').loc[ix].add_prefix(stream + '_jump_'))
    with np.load(OUT / f'deconv_pulse_measure_{part}.npz') as q:
        assert np.array_equal(q['ids'], ix)
        d, v = (q['D'], q['V'])
    se = np.sqrt(np.diagonal(v, axis1=1, axis2=2)).clip(1e-06)
    chol = np.linalg.cholesky(v)
    wh = np.linalg.solve(chol, d[:, :, None])[:, :, 0]
    zz = np.column_stack([d, se, d / se, wh, wh ** 2])
    blocks.append(pd.DataFrame(zz, index=ix).add_prefix('jack'))
    with np.load(OUT / f'shape_counts_{part}.npz') as a:
        assert np.array_equal(a['ids'], ix)
        h = a['hist'].astype(float)
    p = anchors()
    hh = []
    for conc in (200.0, 1800.0):
        a, b, j = (dirichlet_log(data, p, conc) for data in (h[:, 0], h[:, 1], h.sum(1)))
        null = logsumexp(j, axis=1) - np.log(len(p))
        u, w = (t - logsumexp(t, axis=1, keepdims=True) for t in (a, b))
        hh.extend([u.clip(-30, 0), w.clip(-30, 0), (a + b - j).clip(-30, 50), (logsumexp(a, 1) + logsumexp(b, 1) - 2 * np.log(len(p)) - null)[:, None]])
    blocks.append(pd.DataFrame(np.column_stack(hh), index=ix).add_prefix('emp_shape'))
    with np.load(OUT / f'wnlaw_physical_{part}.npz') as a:
        assert np.array_equal(a['ids'], ix)
        w = a['values']
        k = 31
    s, l, v, j = np.split(w[:, :4 * k], 4, axis=1)
    descriptors = np.column_stack([(l - s).clip(-15, 45), (v - s).clip(-15, 45), (j - s).clip(-15, 45), (s - logsumexp(s, axis=1, keepdims=True)).clip(-30, 0)])
    blocks.append(pd.DataFrame(descriptors, index=ix).add_prefix('fixed_law'))
    return pd.concat(blocks, axis=1).astype(np.float32)

def run(mode):
    pair = [matrix(p, 'state') for p in ('train', 'test')]
    ids, tids = [p.index for p in pair]
    for i, (part, index) in enumerate((('train', ids), ('test', tids))):
        pair[i] = pd.concat([pair[i], extra(part, index, mode)], axis=1)
    assert list(pair[0].columns) == list(pair[1].columns)
    y = pd.read_csv(ROOT / 'data/train_labels.csv', dtype={'id': str}).set_index('id').loc[ids, 'label'].to_numpy()
    oof = np.empty(len(y))
    tests = []
    roots = OUT / f'lawtree_{mode}_models'
    roots.mkdir(exist_ok=True)
    for f, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(ids, y)):
        pri = OUT / 'law_outer0'
        with np.load(pri / f'fold{f}_rows.npz') as a:
            assert np.array_equal(a['validation_ids'], ids[va])
        x, t = [pd.concat([old, pd.read_parquet(pri / f'fold{f}_{p}.parquet').loc[old.index].add_prefix('local_prior')], axis=1).astype(np.float32) for old, p in zip(pair, ('train', 'test'))]
        if f == 0:
            (roots / 'columns.json').write_text(json.dumps(list(x.columns)))
        path = roots / f'fold{f}.txt'
        if path.exists():
            m = lgb.Booster(model_file=str(path))
        else:
            params = dict(objective='binary', verbosity=-1, num_threads=2, learning_rate=0.03, max_depth=5, num_leaves=26, min_data_in_leaf=80, lambda_l2=16.0, feature_fraction=0.72, max_bin=95, deterministic=True, force_col_wise=True, seed=337 + f)
            m = lgb.train(params, lgb.Dataset(x.iloc[tr], label=y[tr]), num_boost_round=720)
            m.save_model(str(path))
        oof[va] = m.predict(x.iloc[va], num_threads=2)
        tests.append(m.predict(t, num_threads=2))
        print('law tree', mode, f, round(roc_auc_score(y[va], oof[va]), 6), flush=True)
    path = OUT / f'lawtree_{mode}.npz'
    np.savez_compressed(path, ids=ids.to_numpy(dtype=str), tids=tids.to_numpy(dtype=str), oof=oof, test=np.mean(tests, axis=0))
    print(mode, 'out of fold', roc_auc_score(y, oof), flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('pulse', 'both'))
    with threadpool_limits(limits=2):
        run(p.parse_args().mode)
