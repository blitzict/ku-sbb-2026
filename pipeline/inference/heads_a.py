"""Saved-model prediction for early head group A.

white_tenfold_priors  ten-fold physical priors: saved fold<k>_prior.json applied to the new wnlaw rows
transfer_member       saved transfer_model/<mode>_fold<k>.txt boosters on noise_tests + innov_v2 + periodic rows
shape_bank            saved shape_bank_<mode>/fold<k>.txt boosters + columns.json on the shape-bank matrix
periodic_replay       saved W_SEQ checkpoints (new weights/models) replayed on the common/separate streams;
replay_trend_sequence the trend-stream form of periodic_replay

Training/OOF files are copied unchanged from model_root; only test rows are recomputed. Nothing is refitted.
"""
import hashlib
import importlib
import json
import shutil
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import logsumexp

from .common import destination, option, replace_test_fields, require, save_table

WS = Path('tune_out/weight_space')


def _keep(stage, model_root, work, produced):
    #train side and saved model files are copied verbatim so later readers find them
    written = []
    done = {Path(p).resolve() for p in produced}
    for relative in stage['outputs']:
        target = destination(work, relative)
        if target in done:
            continue
        source = require(Path(model_root) / relative)
        if target.exists():
            target.unlink()  #never write through a link into model_root
        shutil.copy2(source, target)
        written.append(target)
    return list(produced) + written


def tenfold(stage, model_root, work):
    from white_law_head import unpack
    seed = int(stage['argv'][1])
    name = f'law_outer10x{seed}'
    produced = []
    for view in ('', 'c1', 'c2'):
        rel = WS / ('view_' + view) / name if view else WS / name
        query = Path(work) / (rel.parent)
        with np.load(query / 'wnlaw_physical_test.npz') as data:
            tids, observed = pd.Index(data['ids']), data['values'].astype(float)
        for fold in range(10):
            #every view uses the original view's saved prior
            prior = json.loads(require(Path(model_root) / WS / name / f'fold{fold}_prior.json').read_text())
            _, L, F = unpack(observed, np.asarray(prior['law']))
            score = logsumexp(L + np.log(prior['change']), 1) - logsumexp(L + np.log(prior['null']), 1)
            v = np.column_stack((score, F)).clip(-80, 80)
            if not np.isfinite(v).all():
                raise ValueError(f'{name}: non-finite prior features')
            frame = pd.DataFrame(v, index=pd.Index(tids, name='id'), columns=[f'v{j}' for j in range(v.shape[1])])
            target = destination(work, rel / f'fold{fold}_test.parquet')
            save_table(target, frame)
            produced.append(target)
    return _keep(stage, model_root, work, produced)


def _booster(path):
    return lgb.Booster(model_file=str(require(path)))


def transfer(stage, model_root, work):
    out = Path(work) / WS
    tests = pd.read_parquet(out / 'noise_tests_test.parquet')
    dynamics = pd.read_parquet(out / 'innov_v2_test.parquet').loc[tests.index]
    periodic = pd.read_parquet(out / 'periodic_test.parquet').loc[tests.index]
    x = pd.concat([tests, dynamics.add_prefix('innovation_'), periodic.add_prefix('periodic_')], axis=1)
    if not (x.columns.is_unique and np.isfinite(x).all().all()):
        raise ValueError('transfer_member: bad query matrix')
    cols = list(tests.columns)
    result = {}
    for mode in ('pooled', 'noise', 'stationary'):
        active = list(x.columns) if mode == 'pooled' else cols
        xt = x[active]
        ts = []
        for fold in range(5):
            model = _booster(Path(model_root) / WS / f'transfer_model/{mode}_fold{fold}.txt')
            if model.feature_name() != [c.replace(' ', '_') for c in active]:
                raise ValueError(f'transfer_member {mode}: query columns differ from saved model')
            ts.append(model.predict(xt, raw_score=True, num_threads=1))
        result[mode] = np.mean(ts, axis=0)
    target = destination(work, WS / 'transfer_member_test.parquet')
    save_table(target, pd.DataFrame(result, index=x.index))
    return _keep(stage, model_root, work, [target])


def shape_bank(stage, model_root, work):
    module = importlib.import_module('shape_bank')
    mode = stage['argv'][1]
    te = module.matrix('test', mode)
    columns = json.loads(require(Path(model_root) / WS / f'shape_bank_{mode}/columns.json').read_text())
    if list(te.columns) != columns:
        raise ValueError(f'shape_bank {mode}: query columns differ from saved model')
    tests = [_booster(Path(model_root) / WS / f'shape_bank_{mode}/fold{fold}.txt').predict(te, raw_score=True, num_threads=1)
             for fold in range(5)]
    target = destination(work, WS / f'shape_bank_{mode}_test.parquet')
    save_table(target, pd.DataFrame({'score': np.mean(tests, axis=0)}, index=te.index))
    return _keep(stage, model_root, work, [target])


def _seq_parts(tag):
    #nn_seq definitions and trained argv of one saved sequence model (same as neural.nn_seq)
    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from pipeline.inference import source
    from pipeline.run_stages import ROOT
    script = 'experiments/v6/nn_seq.py'
    stages = json.loads((ROOT / 'pipeline/full_cpu_stages.json').read_text())['stages']
    argv = [s['argv'][1:] for s in stages if s['argv'][0] == script and option(s, '--tag') == tag]
    if len(argv) != 1:
        raise ValueError(f'{tag}: expected one nn_seq training stage, found {len(argv)}')
    a = source.parse_args(script, argv[0])
    if [c for c in a.aux.split(',') if c]:
        raise ValueError(f'{tag}: aux channels are not part of the recipe')
    ns = {'a': a, 'dev': 'cpu', 'L': 1024, 'P': a.P, 'AUG': set(x for x in a.aug.split(',') if x), 'CH': a.chans.split(','),
          'torch': torch, 'nn': nn, 'F': F, 'math': math, 'np': np, 'pd': pd}
    source.definitions(script, ['view', 'channels', 'mquant', 'Block', 'Net', 'run', 'predict'], ns)
    return a, ns


def replay(stage, model_root, work):
    import torch
    torch.set_num_threads(1)
    module = importlib.import_module('periodic_replay')
    seq = module.seq
    prefix = 'trend' if Path(stage['argv'][0]).name == 'replay_trend_sequence.py' else option(stage, '--prefix', 'periodic')
    tids, gpu, gate = module.streams('test', prefix)
    manifest = json.loads(require(Path(model_root) / 'new weights/model_manifest.json').read_text())
    hashes = {Path(row['file']): row['sha256'] for row in manifest['files']}
    tags = list(seq.RUNS)
    parts = {tag: _seq_parts(tag) for tag in tags}
    cols = [p + '_' + t for p in ('original', 'common', 'separate') for t in tags]
    rows = torch.as_tensor(np.arange(len(tids)), device='cpu')
    aux = torch.zeros((len(tids), 0), dtype=torch.float32)
    active = gate > 0.075
    test = np.zeros((len(tids), len(cols)))
    produced = []

    def checkpoint(tag, fold, seed):
        rel = Path('models') / tag / f'fold{fold}_seed{seed}.pt'
        ck = require(Path(model_root) / 'new weights' / rel)
        if hashlib.sha256(ck.read_bytes()).hexdigest() != hashes[rel]:
            raise ValueError(f'checkpoint hash differs from manifest: {ck}')
        return torch.load(ck, weights_only=True, map_location='cpu')

    for fold in range(5):
        #original column: seqpart pt, the nn_seq test logits (views x rows, seed mean) of each saved model
        before = []
        for tag in tags:
            a, ns = parts[tag]
            pt = []
            for sd in a.seeds:
                net = ns['Net'](len(ns['CH']), a.ch, [int(d) for d in a.dil.split(',')], a.k, a.drop, 0)
                net.load_state_dict(checkpoint(tag, fold, sd))
                net.eval()
                pt.append(ns['predict'](net, gpu[0], gpu[3], gpu[4], aux))
            before.append(np.mean(pt, 0))
        before = np.stack(before).mean(1).T
        models = []
        for tag, (hq, seed) in seq.RUNS.items():
            seq.a.hq = hq
            model = seq.Net().eval().to('cpu')
            model.load_state_dict(checkpoint(tag, fold, seed))
            models.append((hq, model))
        scores = [before]
        for x in gpu[1:3]:
            pred = module.predict(models, x[rows], gpu[3][rows], gpu[4][rows])
            result = before.copy()
            result[active] = pred.mean(1).T[active]
            scores.append(result)
        output = np.column_stack(scores)
        test += output / 5
        relative = WS / f'{prefix}_replay/fold{fold}.npz'
        replace_test_fields(model_root, work, relative, tids=tids.to_numpy(dtype=str), test=output)
        produced.append(Path(work) / relative)
    if not np.isfinite(test).all():
        raise ValueError(f'{prefix} replay: non-finite test scores')
    target = destination(work, WS / f'{prefix}_seq_test.parquet')
    save_table(target, pd.DataFrame(test, index=tids, columns=cols))
    return _keep(stage, model_root, work, produced + [target])


ADAPTERS = {
    'white_tenfold_priors.py': tenfold, 'transfer_member.py': transfer, 'shape_bank.py': shape_bank,
    'periodic_replay.py': replay, 'replay_trend_sequence.py': replay,
}


def supported(stage):
    return Path(stage['argv'][0]).name in ADAPTERS


def predict_stage(stage, model_root, work):
    return ADAPTERS[Path(stage['argv'][0]).name](stage, Path(model_root), Path(work))
