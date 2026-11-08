# CPU port; original experiments/weight_space/white_independent_rank_folds.py, SHA256 7cb9514ec9b96336b780a048789dccca856f8a1b0851ba75446bf4a1688b376a
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Average rank trees only after each original series was held out of every member."""
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
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from white_rank_boost import crop_data, rank_objective
from partition_evidence import ROOT, OUT

def source_objective(labels, white):
    vote = np.array([1.0, 0.13, 0.13]) / 1.26
    objective = rank_objective(labels, np.ones(len(labels)), white)

    def derivatives(raw, dataset):
        views = raw.reshape(3, -1)
        source = vote @ views
        gradient, curvature = objective(source, dataset)
        p = expit(views)
        grad = 0.78 * vote[:, None] * gradient + 0.22 * vote[:, None] * (p - labels)
        hess = 0.78 * vote[:, None] * curvature + 0.22 * vote[:, None] * p * (1 - p)
        return ((2 * grad).ravel(), (2 * hess).ravel())
    return derivatives

def run(seeds, linked=False, splits=5):
    views = ('', 'c1', 'c2')
    with np.load(ROOT / 'tune_out/var/mat_ns3_c7.npz', allow_pickle=True) as a:
        ids, tids = (a['ids'].astype(str), a['tid'].astype(str))
        y = a['y'].astype(bool)
        X = [crop_data(v, 'train', ids, a['V_' + v if v else 'X']) for v in views]
        T = [crop_data(v, 'test', tids, a['VT_' + v if v else 'Xt']) for v in views]
    assert np.array_equal(pd.read_csv(ROOT / 'data/train_labels.csv', dtype={'id': str}).set_index('id').loc[ids, 'label'], y)
    gates = [pd.read_parquet((OUT / ('view_' + v) if v else OUT) / 'innov_v2_train.parquet', columns=['pre_white']).loc[ids, 'pre_white'].to_numpy() for v in views]
    probs = []
    tests = []
    for s in seeds:
        prefix = 'law_linked_rank' if linked else 'law_independent_rank'
        partition = str(s) if splits == 5 else f'{splits}x{s}'
        name = f'{prefix}_s{partition}'
        dest = OUT / (name + '_models')
        dest.mkdir(exist_ok=True)
        pv = np.empty(len(y))
        pt = []
        for f, (tr, va) in enumerate(StratifiedKFold(splits, shuffle=True, random_state=s).split(ids, y)):
            vv = []
            tt = []
            xx = []
            for v, x, t in zip(views, X, T):
                path = (OUT / ('view_' + v) if v else OUT) / f'law_outer{partition}'
                with np.load(path / f'fold{f}_rows.npz') as r:
                    assert np.array_equal(r['train_ids'], ids[tr]) and np.array_equal(r['validation_ids'], ids[va])
                prior = pd.read_parquet(path / f'fold{f}_train.parquet').loc[ids].to_numpy(np.float32)
                xx.append(np.column_stack([x[tr], prior[tr]]))
                vv.append(np.column_stack([x[va], prior[va]]))
                tt.append(np.column_stack([t, pd.read_parquet(path / f'fold{f}_test.parquet').loc[tids].to_numpy(np.float32)]))
            ck = dest / f'fold{f}.txt'
            if ck.exists():
                m = lgb.Booster(model_file=str(ck))
            else:
                wt = np.repeat([1.0, 0.45, 0.45], len(tr))
                target = np.tile(y[tr], 3)
                obj = source_objective(y[tr], gates[0][tr]) if linked else rank_objective(target, wt, np.concatenate([g[tr] for g in gates]))
                m = lgb.train(dict(objective=obj, metric='None', learning_rate=0.035, num_leaves=29, max_depth=6, min_data_in_leaf=78, lambda_l2=12.0, feature_fraction=0.56, verbosity=-1, deterministic=True, force_col_wise=True, num_threads=2, max_bin=70, seed=2851 + f), lgb.Dataset(np.vstack(xx), label=target), num_boost_round=750)
                m.save_model(str(ck))

            def predict(a):
                z = [m.predict(v, raw_score=True, num_threads=2) for v in a]
                return expit((z[0] + 0.13 * z[1] + 0.13 * z[2]) / 1.26)
            pv[va] = predict(vv)
            pt.append(predict(tt))
            print(name, f, roc_auc_score(y[va], pv[va]), flush=True)
        pt = np.mean(pt, 0)
        np.savez_compressed(OUT / (name + '.npz'), oof=pv, test=pt, ids=ids, tids=tids)
        probs.append(logit(pv))
        tests.append(logit(pt))
        print(name, 'OOF', roc_auc_score(y, pv), flush=True)
    if linked or splits != 5:
        return
    with np.load(OUT / 'lawrank_crops.npz') as a:
        assert np.array_equal(ids, a['ids']) and np.array_equal(tids, a['tids'])
        probs.append(logit(a['oof']))
        tests.append(logit(a['test']))
    name = 'law_independent_rank_bag_' + '_'.join(map(str, seeds))
    oof = expit(np.mean(probs, 0))
    test = expit(np.mean(tests, 0))
    np.savez_compressed(OUT / (name + '.npz'), ids=ids, tids=tids, oof=oof, test=test)
    print(name, roc_auc_score(y, oof), flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('seeds', nargs='+', type=int)
    p.add_argument('--linked', action='store_true')
    p.add_argument('--splits', type=int, choices=(5, 10), default=5)
    a = p.parse_args()
    with threadpool_limits(limits=2):
        run(a.seeds, a.linked, a.splits)
