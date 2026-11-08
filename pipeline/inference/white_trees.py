"""Saved tree specialists: build new query matrices, load boosters, predict.

Training/OOF arrays are retained verbatim. Fold and view averaging follows
each frozen source's test path, including averaging logits before sigmoid
where the original model did so.
"""
import importlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRanker
from scipy.special import expit, logit

from .common import require, replace_test_fields, option

WS = Path('tune_out/weight_space')


def _save(model_root, work, name, ids, **values):
    relative = WS / (name + '.npz')
    replace_test_fields(model_root, work, relative, tids=np.asarray(ids).astype(str), **values)
    return [Path(work) / relative]


def _prior(work, view, fold, ids, seed=0):
    folder = Path(work) / WS
    if view:
        folder /= 'view_' + view
    return pd.read_parquet(folder / f'law_outer{seed}/fold{fold}_test.parquet').loc[ids]


def _booster(model_root, name, fold, prefix='fold'):
    return lgb.Booster(model_file=str(require(Path(model_root) / WS / f'{name}_models/{prefix}{fold}.txt')))


def _variants(work):
    return np.load(Path(work) / 'tune_out/var/mat_ns3_c7.npz', allow_pickle=True)


def mech(stage, model_root, work):
    module = importlib.import_module('white_mech_tree')
    mode = stage['argv'][1]
    base = module.matrix('test', 'state')
    ids = base.index
    base = pd.concat([base, module.extra('test', ids, mode)], axis=1)
    name = 'lawtree_' + mode
    columns = json.loads(require(Path(model_root) / WS / f'{name}_models/columns.json').read_text())
    values = []
    for fold in range(5):
        frame = pd.concat([base, _prior(work, '', fold, ids).add_prefix('local_prior')], axis=1).astype(np.float32)
        if list(frame.columns) != columns:
            raise ValueError(f'{name}: query columns differ from saved model')
        values.append(_booster(model_root, name, fold).predict(frame, num_threads=1))
    return _save(model_root, work, name, ids, test=np.mean(values, axis=0))


def symmetric(stage, model_root, work):
    module = importlib.import_module('white_symmetric_tree')
    args = stage['argv'][1:]
    views = [x for x in args[args.index('--views') + 1:] if not x.startswith('--')]
    extra = '--extras' in args
    name = 'symtree' + ('_extra' if extra else '') + '_' + '_'.join(['orig', *views])
    with _variants(work) as data:
        ids = data['tid'].astype(str)
        tables = [module.read(view, 'test', ids, data['VT_' + view if view else 'Xt'], extra)
                  for view in ('', *views)]
    weights = np.r_[1., np.repeat(.18, len(views))]
    weights /= weights.sum()
    tests = []
    for fold in range(5):
        model = CatBoostClassifier()
        model.load_model(str(require(Path(model_root) / WS / f'{name}_models/fold{fold}.cbm')))
        margins = []
        for view, table in zip(('', *views), tables):
            x = np.column_stack([table, _prior(work, view, fold, ids).to_numpy(np.float32)])
            margins.append(model.predict(x, prediction_type='RawFormulaVal', thread_count=1))
        tests.append(expit(np.stack(margins, axis=1) @ weights))
    return _save(model_root, work, name, ids, test=np.mean(tests, axis=0))


def rank(stage, model_root, work):
    module = importlib.import_module('white_rank_boost')
    mode = stage['argv'][1]
    views = ['', 'c1', 'c2', 'rev'] if mode == 'crops' else ['']
    with _variants(work) as data:
        ids = data['tid'].astype(str)
        if mode == 'crops':
            tables = [module.crop_data(v, 'test', ids, data['VT_' + v if v else 'Xt']) for v in views]
        else:
            names = ['native_law_state', 'filtered_step', 'markov_copula', 'component_moves_state', 'component_moves_pulse']
            tables = [module.input_frame('test', pd.Index(ids), names).to_numpy(np.float32)]
    weights = np.array([1., *[.11 for _ in views[1:]]])
    weights /= weights.sum()
    tests, name = [], 'lawrank_' + mode
    for fold in range(5):
        model = _booster(model_root, name, fold)
        frames = [np.column_stack([table, _prior(work, view, fold, ids).to_numpy(np.float32)])
                  for view, table in zip(views, tables)]
        tests.append(expit(sum(w * model.predict(x, raw_score=True, num_threads=1) for w, x in zip(weights, frames))))
    return _save(model_root, work, name, ids, test=np.mean(tests, axis=0))


def independent(stage, model_root, work):
    module = importlib.import_module('white_independent_rank_folds')
    args = stage['argv'][1:]
    seeds = [int(x) for x in args[:next((i for i, x in enumerate(args) if x.startswith('--')), len(args))]]
    linked = '--linked' in args
    splits = option(stage, '--splits', 5, int)
    views = ('', 'c1', 'c2')
    with _variants(work) as data:
        ids = data['tid'].astype(str)
        tables = [module.crop_data(v, 'test', ids, data['VT_' + v if v else 'Xt']) for v in views]
    written, logits = [], []
    for seed in seeds:
        partition = str(seed) if splits == 5 else f'{splits}x{seed}'
        name = ('law_linked_rank' if linked else 'law_independent_rank') + '_s' + partition
        tests = []
        for fold in range(splits):
            model = _booster(model_root, name, fold)
            z = [model.predict(np.column_stack([table, _prior(work, view, fold, ids, partition).to_numpy(np.float32)]),
                               raw_score=True, num_threads=1) for view, table in zip(views, tables)]
            tests.append(expit((z[0] + .13 * z[1] + .13 * z[2]) / 1.26))
        values = np.mean(tests, axis=0)
        written += _save(model_root, work, name, ids, test=values)
        logits.append(logit(values))
    if not linked and splits == 5:
        with np.load(Path(work) / WS / 'lawrank_crops.npz') as data:
            if not np.array_equal(data['tids'], ids):
                raise ValueError('Rank bag query order differs')
            logits.append(logit(data['test']))
        written += _save(model_root, work, 'law_independent_rank_bag_' + '_'.join(map(str, seeds)), ids,
                         test=expit(np.mean(logits, axis=0)))
    return written


def crop(stage, model_root, work):
    module = importlib.import_module('white_crop_laws')
    views = [x for x in stage['argv'][1:] if not x.startswith('--')]
    for view in views:
        module.make('test', view)
        for tag in ('pulse', 'state'):
            module.extract('test', tag + view)
    with _variants(work) as data:
        ids = data['tid'].astype(str)
        tables = [module.design(data['VT_' + v if v else 'Xt'], ids, 'test', v) for v in ('', *views)]
    name = 'lawcrop_' + '_'.join(views)
    tests = []
    for fold in range(5):
        model = _booster(model_root, name, fold)
        tests.append(np.stack([model.predict(x, num_threads=1) for x in tables], axis=1))
    mean = np.mean(tests, axis=0)
    weights = np.r_[.65, np.full(len(views), .35 / len(views))]
    return _save(model_root, work, name, ids, test=mean @ weights, test_views=mean)


def second(stage, model_root, work):
    module = importlib.import_module('white_second_tree')
    name = stage['argv'][1]
    args = stage['argv']
    more = args[args.index('--tables') + 1:] if '--tables' in args else []
    ids = pd.read_parquet(Path(work) / 'clean/features_test.parquet', columns=[]).index
    base = module.input_frame('test', ids, more)
    columns = json.loads(require(Path(model_root) / WS / f'{name}_models/columns.json').read_text())
    tests, residuals = [], []
    for fold in range(5):
        frame = pd.concat([base, _prior(work, '', fold, ids).add_prefix('fold_prior_')], axis=1)
        if list(frame.columns) != columns:
            raise ValueError(f'{name}: query columns differ from saved model')
        tests.append(expit(_booster(model_root, name, fold, 'standalone').predict(frame, raw_score=True, num_threads=1)))
        residuals.append(_booster(model_root, name, fold, 'residual').predict(frame, raw_score=True, num_threads=1))
    with np.load(module.CACHE) as offsets:
        if not np.array_equal(offsets['tids'], ids):
            raise ValueError('Residual parent query identifiers differ')
        parent = offsets['zt']
    return (_save(model_root, work, name, ids, test=np.mean(tests, axis=0))
            + _save(model_root, work, name + '_res', ids, test=expit(parent + np.mean(residuals, axis=0))))


def pairs(stage, model_root, work):
    module = importlib.import_module('white_pair_trees')
    mode, views = stage['argv'][1], ('', 'c1', 'c2')
    name = 'law_symmetric_' + mode
    with _variants(work) as data:
        ids = data['tid'].astype(str)
        tables = [module.features(data, v, 'test', ids) for v in views]
    tests = []
    for fold in range(5):
        model = CatBoostRanker() if mode == 'pairs' else CatBoostClassifier()
        model.load_model(str(require(Path(model_root) / WS / f'{name}_models/fold{fold}.cbm')))
        z = []
        for view, table in zip(views, tables):
            matrix = np.column_stack([table, _prior(work, view, fold, ids).to_numpy(np.float32)])
            kwargs = {} if mode == 'pairs' else {'prediction_type': 'RawFormulaVal'}
            z.append(model.predict(matrix, thread_count=1, **kwargs))
        tests.append(expit((z[0] + .1 * z[1] + .1 * z[2]) / 1.2))
    return _save(model_root, work, name, ids, test=np.mean(tests, axis=0))


ADAPTERS = {
    'white_mech_tree.py': mech, 'white_symmetric_tree.py': symmetric,
    'white_rank_boost.py': rank, 'white_independent_rank_folds.py': independent,
    'white_crop_laws.py': crop, 'white_second_tree.py': second, 'white_pair_trees.py': pairs,
}


def supported(stage):
    return Path(stage['argv'][0]).name in ADAPTERS


def predict_stage(stage, model_root, work):
    return ADAPTERS[Path(stage['argv'][0]).name](stage, Path(model_root), Path(work))
