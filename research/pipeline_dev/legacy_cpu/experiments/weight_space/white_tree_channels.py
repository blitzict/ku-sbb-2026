# CPU port; original experiments/weight_space/white_tree_channels.py, SHA256 c40db019590ae1eb6c6c1eb82c64f3af5954d98eb3b280310909bcb60f441861
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Resolve an existing rank forest's log odds into contributions by physical mechanism."""
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
import lightgbm as lgb
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from white_rank_boost import crop_data, TABLES
from partition_evidence import OUT, ROOT

def family(source, col):
    if source == 'raw':
        k = col.lower()
        if any((p in k for p in ('garch', 'ewstd', 'vol', 'mstd', 'sqacf', 'arch'))):
            return 'raw_volatility'
        if any((p in k for p in ('lsq', 'slope', 'trend', 'level', 'mean', 'cusum'))):
            return 'raw_level'
        if any((p in k for p in ('ar1', 'ar2', 'acf', 'persist', 'pacf', 'spec', 'lags', 'period'))):
            return 'raw_memory'
        if any((p in k for p in ('kurt', 'skew', 'bim', 'gn_', 'tail', 'shape', 'cvm', 'mmd', 'rank'))):
            return 'raw_shape'
        return 'raw_other'
    if source in ('periodic', 'phase_law'):
        return 'oscillation'
    if source in ('walk_jump', 'curve', 'trend'):
        return 'drift'
    if source in ('state', 'ar_law', 'innov_v2'):
        return 'dynamics'
    if source in ('mixture_law', 'native_law_state'):
        return 'mixture'
    if source in ('noise_tests', 'bounds', 'partition', 'volatility_law'):
        return source
    if source.startswith('jump_law'):
        if col.startswith('level'):
            return 'jump_location'
        if col.startswith('scale'):
            return 'jump_width'
        if col.startswith('shape'):
            return 'jump_shape'
        return 'jump_other'
    return 'fold_prior'

def groups(cols, partition):
    labels = [family('raw', c) for c in cols]
    for tag in (*TABLES, 'volatility_law', 'walk_jump', 'native_law_state', 'state', *['jump_law_' + s for s in ('pulse', 'state', 'nullstate', 'prestate', 'smoothvol')]):
        frame = pd.read_parquet(OUT / f'{tag}_test.parquet')
        names = frame.columns
        if tag.startswith(('jump_law', 'native_law')):
            names = [c for c in names if '_pair' not in c and '_law' not in c]
        labels.extend((family(tag, c) for c in names))
    names = pd.read_parquet(OUT / f'law_outer{partition}/fold0_test.parquet').columns
    labels.extend((family('prior', c) for c in names))
    unique = sorted(set(labels))
    return (unique, np.array([unique.index(s) for s in labels]))

def paths(model, group):
    trees = model.dump_model()['tree_info']
    H = 1 + int(group.max())
    size = max((t['num_leaves'] for t in trees))
    lookup = np.zeros((len(trees), size, H + 1), np.float32)

    def visit(tree, node, path):
        if 'leaf_index' in node:
            lookup[tree, node['leaf_index']] = path
            return
        g = group[node['split_feature']]
        for child in ('left_child', 'right_child'):
            c = node[child]
            increment = c.get('leaf_value', c.get('internal_value')) - node['internal_value']
            nxt = path.copy()
            nxt[g] += increment
            visit(tree, c, nxt)
    for j, tree in enumerate(trees):
        node = tree['tree_structure']
        init = np.zeros(H + 1)
        init[-1] = node['internal_value']
        visit(j, node, init)
    return lookup

def evaluate(m, x, lookup):
    idx = m.predict(x, pred_leaf=True, num_threads=2)
    out = np.zeros((len(x), 2, lookup.shape[2]), float)
    for k in range(lookup.shape[0]):
        out[:, int(k >= 180), :] += lookup[k, idx[:, k]]
    z = m.predict(x, raw_score=True, num_threads=2)
    assert np.max(abs(z - out.sum((1, 2)))) < 2e-05
    return out

def run(seed, linked=False, splits=5):
    partition = str(seed) if splits == 5 else f'{splits}x{seed}'
    which = 'linked' if linked else 'independent'
    name = f'law_{which}_rank_s{partition}'
    dst = OUT / f'tree_channels_{which}_{partition}.npz'
    views = ('', 'c1', 'c2')
    with np.load(ROOT / 'tune_out/var/mat_ns3_c7.npz', allow_pickle=True) as a:
        ids, tids, y = (a['ids'].astype(str), a['tid'].astype(str), a['y'])
        names, g = groups(a['cols'], partition)
        X = [crop_data(v, 'train', ids, a['V_' + v if v else 'X']) for v in views]
        T = [crop_data(v, 'test', tids, a['VT_' + v if v else 'Xt']) for v in views]
    train = np.empty((len(ids), 3, 2, len(names) + 1), np.float32)
    test = np.zeros((len(tids), 3, 2, len(names) + 1), float)
    reported = []
    w = np.array([1, 0.13, 0.13]) / 1.26
    with np.load(OUT / (name + '.npz')) as ref:
        assert np.array_equal(ref['ids'], ids) and np.array_equal(ref['tids'], tids)
        for f, (tr, va) in enumerate(StratifiedKFold(splits, shuffle=True, random_state=seed).split(ids, y)):
            d = OUT / (name + '_models') / f'fold{f}.txt'
            model = lgb.Booster(model_file=str(d))
            assert model.num_feature() == len(g)
            chart = paths(model, g)
            outs = []
            for j, v in enumerate(views):
                root = (OUT / ('view_' + v) if v else OUT) / f'law_outer{partition}'
                with np.load(root / f'fold{f}_rows.npz') as a:
                    assert np.array_equal(a['train_ids'], ids[tr]) and np.array_equal(a['validation_ids'], ids[va])
                p = pd.read_parquet(root / f'fold{f}_train.parquet').loc[ids[va]].to_numpy(np.float32)
                t = pd.read_parquet(root / f'fold{f}_test.parquet').loc[tids].to_numpy(np.float32)
                train[va, j] = evaluate(model, np.column_stack([X[j][va], p]), chart)
                tt = evaluate(model, np.column_stack([T[j], t]), chart)
                test[:, j] += tt / splits
                outs.append(tt.sum((1, 2)))
            reported.append(expit(w @ np.stack(outs)))
            check = expit(train[va].sum((2, 3)) @ w)
            assert np.max(abs(check - ref['oof'][va])) < 2e-06
            print(name, 'verified fold', f, flush=True)
        assert np.max(abs(np.mean(reported, axis=0) - ref['test'])) < 2e-06
    np.savez_compressed(dst, ids=ids, tids=tids, names=np.array(names + ['offset']), oof=train, test=test.astype(np.float32))

def tables(inputs):
    reference = None
    for name in inputs:
        p = OUT / f'tree_channels_{name}.npz'
        with np.load(p) as a:
            if reference is None:
                ids, tids, names = [a[k].copy() for k in ('ids', 'tids', 'names')]
                reference = [[], []]
            assert np.array_equal(a['ids'], ids) and np.array_equal(a['tids'], tids) and np.array_equal(a['names'], names)
            for rows, k in zip(reference, ('oof', 'test')):
                rows.append(a[k])
    for part, ix, arr in zip(('train', 'test'), (ids, tids), reference):
        x = np.mean(arr, axis=0)
        z = x.sum(2)
        w = np.array([1, 0.13, 0.13]) / 1.26
        full = np.einsum('nvg,v->ng', z, w)
        crop = z[:, 1:].mean(1) - z[:, 0]
        late = np.einsum('nvg,v->ng', x[:, :, 1], w)
        df = pd.DataFrame(np.column_stack([b[:, :-1] for b in (full, crop, late)]), index=pd.Index(ix, name='id'), columns=[a + '_' + g for a in ('full', 'crop', 'late') for g in names[:-1]])
        df.to_parquet(OUT / ('tree_channels_' + '_'.join(inputs) + f'_{part}.parquet'))
    print('all OOF and test scores reconstructed', len(ids), len(tids), len(names) - 1, 'physical groups', flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=29)
    p.add_argument('--linked', action='store_true')
    p.add_argument('--splits', type=int, default=5)
    p.add_argument('--tables', nargs='+')
    args = p.parse_args()
    with threadpool_limits(limits=2):
        tables(args.tables) if args.tables else run(args.seed, args.linked, args.splits)
