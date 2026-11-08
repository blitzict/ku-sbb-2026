# CPU port; original experiments/weight_space/white_pair_trees.py, SHA256 3d5dd02dbc3dffe4e3f6b31a6fc0906f1aacdbe275ae70d02ac6046733ec35d5
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Fit symmetric trees to genuinely difficult positive-negative comparisons on real views."""
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
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRanker, Pool
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
ROOT = _cpu_workspace()
OUT = ROOT / 'tune_out/weight_space'
MECH = {'periodic': 'gate period fraction phase_gain freq_gain phase_change amplitude_change separate_level separate_scale separate_shape', 'curve': 'null_trend jump_trend linear quadratic curvature slope obs_step slope_step mixture gate', 'innov_v2': 'white walk pre_white post_white pre_walk post_walk dynamics_bf shared_shape shared_level shared_scale raw_jump walk_reset', 'ar_law': 'df tail_bf white_jump ar_jump walk_jump level_bf scale_bf memory_bf memory_shift p_walk p_white', 'mixture_law': 'gate pre_bim post_bim d_bf p_bf m_bf s_bf mix_evidence'}

def features(q, view, part, ix):
    v = 'V_' if part == 'train' else 'VT_'
    key = v + view if view else 'X' if part == 'train' else 'Xt'
    d = OUT / ('view_' + view) if view else OUT
    cols = [q[key].astype(np.float32)]
    for k, names in MECH.items():
        cols.append(pd.read_parquet(d / f'{k}_{part}.parquet', columns=names.split()).loc[ix].to_numpy())
    for k in ('volatility_law', 'jump_law_state', 'jump_law_pulse', 'jump_law_prestate', 'native_law_state'):
        tab = pd.read_parquet(d / f'{k}_{part}.parquet').loc[ix]
        if k.startswith('jump_'):
            tab = tab.filter(regex='^(pre\\d+|(level|scale|shape)\\d+)$')
        if k.startswith('native'):
            tab = tab.filter(regex='^native_(null|free_scale_from|free_mean_from)\\d+$')
        cols.append(tab.to_numpy())
    return np.column_stack(cols).astype(np.float32)

def pairs(y, z, seed):
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y)
    neg = np.flatnonzero(y == 0)
    neg = neg[np.argsort(z[neg])]
    ix = np.searchsorted(z[neg], z[pos])
    n = len(neg)
    local = (ix[:, None] + rng.normal(0, 0.09 * n, (len(pos), 14))).astype(int).clip(0, n - 1)
    global_ = rng.integers(n, size=(len(pos), 6))
    opponent = neg[np.column_stack([local, global_])]
    return np.column_stack([np.repeat(pos, opponent.shape[1]), opponent.ravel()]).astype(np.int32)

def run(mode):
    views = ('', 'c1', 'c2')
    tag = 'law_symmetric_' + mode
    dest = OUT / (tag + '_models')
    dest.mkdir(exist_ok=True)
    with np.load(ROOT / 'tune_out/var/mat_ns3_c7.npz', allow_pickle=True) as q:
        ids, tids = (q['ids'].astype(str), q['tid'].astype(str))
        y = q['y'].astype(int)
        raw = [features(q, v, 'train', ids) for v in views]
        xt = [features(q, v, 'test', tids) for v in views]
    ref = pd.read_csv(ROOT / 'data/train_labels.csv', dtype={'id': str}).set_index('id').loc[ids, 'label']
    assert np.array_equal(ref, y)
    oof = np.empty(len(y))
    pt = []
    old = np.load(OUT / 'white_loop_parent.npz')
    for f, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(ids, y)):
        z = old[f's0_f{f}_zr']
        assert np.array_equal(tr, old[f's0_f{f}_tr'])
        train = []
        valid = []
        test = []
        for v, base, bt in zip(views, raw, xt):
            d = (OUT / ('view_' + v) if v else OUT) / 'law_outer0'
            with np.load(d / f'fold{f}_rows.npz') as a:
                assert np.array_equal(ids[va], a['validation_ids'])
            u = pd.read_parquet(d / f'fold{f}_train.parquet').loc[ids].to_numpy(np.float32)
            b = pd.read_parquet(d / f'fold{f}_test.parquet').loc[tids].to_numpy(np.float32)
            train.append(np.column_stack([base[tr], u[tr]]))
            valid.append(np.column_stack([base[va], u[va]]))
            test.append(np.column_stack([bt, b]))
        ck = dest / f'fold{f}.cbm'
        isrank = mode == 'pairs'
        maker = CatBoostRanker if isrank else CatBoostClassifier
        params = dict(iterations=790, depth=6, learning_rate=0.05, l2_leaf_reg=10.0, rsm=0.54, random_seed=1541 + f, thread_count=2, border_count=52, verbose=False, allow_writing_files=False)
        m = maker(**params, loss_function='PairLogit' if isrank else 'Logloss')
        if ck.exists():
            m.load_model(str(ck))
        else:
            yy = np.tile(y[tr], len(views))
            ww = np.repeat([1.0, 0.48, 0.48], len(tr))
            if isrank:
                a = pairs(y[tr], z, 650 + f)
                pair = np.vstack([a + i * len(tr) for i in range(len(views))])
                ds = Pool(np.vstack(train), yy, group_id=np.repeat(np.arange(len(views)), len(tr)), pairs=pair, pairs_weight=np.repeat([1.0, 0.48, 0.48], len(a)))
            else:
                ds = Pool(np.vstack(train), yy, weight=ww)
            m.fit(ds)
            m.save_model(str(ck))

        def predict(arr):
            val = [m.predict(x, thread_count=2) if isrank else m.predict(x, prediction_type='RawFormulaVal', thread_count=2) for x in arr]
            return expit((val[0] + 0.1 * val[1] + 0.1 * val[2]) / 1.2)
        oof[va] = predict(valid)
        pt.append(predict(test))
        print(tag, f, roc_auc_score(y[va], oof[va]), flush=True)
    np.savez_compressed(OUT / (tag + '.npz'), ids=ids, tids=tids, oof=oof, test=np.mean(pt, 0))
    print('real-view symmetric tree', tag, roc_auc_score(y, oof), flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('pairs', 'binary'))
    with threadpool_limits(limits=2):
        run(p.parse_args().mode)
