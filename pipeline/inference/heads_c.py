"""Saved law-head networks (group C): rebuild query inputs, load fold weights, predict.

Covers white_shared_net, white_mixture_program, white_predictive_density, white_fourier_kernel,
white_rank_likelihood, white_forecast_density and white_block_kernel. Each fold's normalization
(knots / center+scale / mu+sd) is read from the saved fold archive and only applied to the query
rows; networks are loaded from their saved state dicts. Training/OOF fields are kept verbatim,
query fields (tids, test, per-fold pt/Ht) are replaced; weight files are copied unchanged.

Precision: the original accelerator run used bfloat16 autocast in the predict loops of
white_shared_net, white_fourier_kernel and white_block_kernel; the CPU port uses float32.
A model_root carrying CPU_REBUILD_WORKSPACE.json is scored in float32, any other model_root
with CPU bfloat16 autocast in those three loops (closest available match to its training).
"""
import contextlib
import importlib
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import ndtri

from .common import destination, replace_test_fields, require

WS = Path('tune_out/weight_space')


def _cpu_port(model_root):
    return (Path(model_root) / 'CPU_REBUILD_WORKSPACE.json').is_file()


def _precision(model_root):
    #original accelerator weights were evaluated under bf16 autocast
    if _cpu_port(model_root):
        return contextlib.nullcontext()
    return torch.autocast('cpu', dtype=torch.bfloat16)


def _module(name, work):
    module = importlib.import_module(name)
    out = Path(module.OUT).resolve()
    if not out.is_relative_to(Path(work).resolve()):
        raise ValueError(f'{name}: OUT {out} is not inside the prediction workspace')
    return module


def _fresh(module, cache, build):
    #query caches are derived from the new series; never reuse a stale one
    path = Path(module.OUT) / cache
    path.unlink(missing_ok=True)
    try:
        return build()
    finally:
        path.unlink(missing_ok=True)


def _state(path):
    state = torch.load(require(path), map_location='cpu', weights_only=True)
    if not all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point()):
        raise ValueError(f'non-finite weights in {path}')
    return state


def _load(net, path):
    net.load_state_dict(_state(path))
    net.eval()
    return net


def _fold(model_root, relative):
    with np.load(require(Path(model_root) / relative), allow_pickle=True) as archive:
        return {key: archive[key] for key in archive.files}


def _law(work, fold, tids):
    return pd.read_parquet(Path(work) / WS / f'law_outer0/fold{fold}_test.parquet').loc[tids].to_numpy()


def _copy(stage, model_root, work, written):
    #weight files and untouched fold archives are part of the declared outputs
    done = {Path(p).resolve() for p in written}
    out = list(written)
    for relative in stage['outputs']:
        target = destination(work, relative)
        if target in done:
            continue
        shutil.copyfile(require(Path(model_root) / relative), target)
        out.append(target)
    return out


def _finish(stage, model_root, work, name, tids, tests, folds=None):
    written = []
    for relative, fields in (folds or {}).items():
        replace_test_fields(model_root, work, relative, **fields)
        written.append(destination(work, relative))
    relative = WS / (name + '.npz')
    replace_test_fields(model_root, work, relative, tids=np.asarray(tids).astype(str), test=np.mean(tests, axis=0))
    written.append(destination(work, relative))
    return _copy(stage, model_root, work, written)


def _apply_knots(x, knots):
    #white_shared_net.normalize with saved knots, query rows only
    prob = np.linspace(0.002, 0.998, 95)
    result = np.zeros(x.shape, np.float32)
    for i, q in enumerate(knots.T):
        unique, rev, counts = np.unique(q, return_inverse=True, return_counts=True)
        if len(unique) < 2 or not np.isfinite(unique).all():
            continue
        level = np.bincount(rev, weights=prob) / counts
        v = x[:, i]
        ok = np.isfinite(v)
        result[ok, i] = ndtri(np.interp(v[ok], unique, level))
    return np.ascontiguousarray(result)


def shared(stage, model_root, work):
    module = _module('white_shared_net', work)
    torch.set_num_threads(1)
    b = module.inputs('test')
    tids = b.index
    full = b.to_numpy(np.float32)
    folder = WS / 'lawshared_models'
    tests, folds = [], {}
    for fold in range(5):
        saved = _fold(model_root, folder / f'fold{fold}.npz')
        h = pd.read_parquet(Path(work) / WS / f'law_outer0/fold{fold}_test.parquet').loc[tids].to_numpy(dtype=np.float32)
        raw = np.column_stack([full, h])
        if raw.shape[1] != saved['knots'].shape[1]:
            raise ValueError(f'lawshared fold {fold}: {raw.shape[1]} query columns, model has {saved["knots"].shape[1]}')
        x = torch.as_tensor(_apply_knots(raw, saved['knots']))
        net = _load(module.Net(x.shape[1]), Path(model_root) / folder / f'fold{fold}.pt')
        ix = np.arange(len(tids))
        vals = []
        with torch.no_grad():
            for start in range(0, len(ix), 384):
                v = x[torch.as_tensor(ix[start:start + 384])]
                with _precision(model_root):
                    vals.append(net(v).float().sigmoid().mean(1))
        pt = torch.cat(vals).numpy()
        tests.append(pt)
        folds[folder / f'fold{fold}.npz'] = {'pt': pt}
    return _finish(stage, model_root, work, 'lawshared', tids, tests, folds)


def conditional(stage, model_root, work):
    module = _module('white_mixture_program', work)
    torch.set_num_threads(1)
    tids, tx, tc, sizes = module.evidence_table('test')
    x = torch.tensor(tx)
    te = np.arange(len(tids))
    folder = WS / 'conditional_physics_models'
    tests = []
    for fold in range(5):
        saved = _fold(model_root, folder / f'fold{fold}_input.npz')
        c = torch.tensor(((tc - saved['center']) / saved['scale']).clip(-8, 8))
        net = _load(module.PhysicalOdds(c.shape[1], sizes), Path(model_root) / folder / f'fold{fold}.pt')
        tests.append(module.predict(net, x, c, te))
    return _finish(stage, model_root, work, 'lawconditional', tids, tests)


def predictive(stage, model_root, work):
    module = _module('white_predictive_density', work)
    torch.set_num_threads(1)
    tids, th = _fresh(module, 'predictive_density_test.npz', lambda: module.observations('test'))
    hist = torch.tensor(th, dtype=torch.float32)
    aux = module.phys('test', tids)
    te = np.arange(len(tids))
    folder = WS / 'predictive_density_models'
    tests = []
    for fold in range(5):
        saved = _fold(model_root, folder / f'fold{fold}.npz')
        x = np.column_stack([aux, np.arcsinh(_law(work, fold, tids))]).astype(np.float32)
        x = torch.tensor(((x - saved['mu']) / saved['sd']).clip(-7, 7))
        model = _load(module.EmpiricalDensity(x.shape[1]), Path(model_root) / folder / f'fold{fold}.pt')
        with torch.no_grad():
            tests.append(torch.cat([model(hist[j], x[j])[0].sigmoid()
                                    for j in np.array_split(te, max(1, len(te) // 220))]).numpy())
    return _finish(stage, model_root, work, 'lawpredictive', tids, tests)


def fourier(stage, model_root, work):
    module = _module('white_fourier_kernel', work)
    torch.set_num_threads(1)
    tids, v = _fresh(module, 'fourier_law_test.npz', lambda: module.load('test'))
    p = module.phys('test', tids)
    te = np.arange(len(tids))
    folder = WS / 'fourier_law_models'
    tests = []
    for fold in range(5):
        saved = _fold(model_root, folder / f'fold{fold}.npz')
        aux = np.column_stack([p, np.arcsinh(_law(work, fold, tids))]).astype(np.float32)
        x = torch.tensor(((v - saved['center']) / saved['dev']).clip(-10, 10), dtype=torch.float32)
        c = torch.tensor(((aux - saved['mu']) / saved['sd']).clip(-8, 8), dtype=torch.float32)
        model = _load(module.ContrastNet(x.shape[1], x.shape[2], c.shape[1]), Path(model_root) / folder / f'fold{fold}.pt')
        with torch.no_grad(), _precision(model_root):
            tests.append(torch.cat([model(x[j], c[j]).float().sigmoid()
                                    for j in np.array_split(te, max(1, len(te) // 400))]).numpy())
    return _finish(stage, model_root, work, 'lawfourier', tids, tests)


def rankdensity(stage, model_root, work):
    module = _module('white_rank_likelihood', work)
    torch.set_num_threads(1)
    tids, counts, quant = _fresh(module, 'rank_likelihood_test.npz', lambda: module.prepare('test'))
    cnt, q = torch.tensor(counts), torch.tensor(quant)
    aux = module.phys('test', tids)
    te = np.arange(len(tids))
    folder = WS / 'rank_likelihood_models'
    tests = []
    for fold in range(5):
        saved = _fold(model_root, folder / f'fold{fold}.npz')
        x = torch.as_tensor(((aux - saved['center']) / saved['scale']).clip(-7, 7), dtype=torch.float32)
        net = _load(module.RankLaws(x.shape[1], cnt.shape[1]), Path(model_root) / folder / f'fold{fold}.pt')
        with torch.no_grad():
            tests.append(np.concatenate([net(cnt[j].float(), q[j].float(), x[j]).sigmoid().numpy()
                                         for j in np.array_split(te, max(1, len(te) // 130))]))
    return _finish(stage, model_root, work, 'lawrankdensity', tids, tests)


def forecast(stage, model_root, work):
    module = _module('white_forecast_density', work)
    torch.set_num_threads(1)
    tids, n, b, z, c = _fresh(module, 'ar_predictive_test.npz', lambda: module.prepare('test'))
    gpu = tuple(torch.tensor(v) for v in (z, c, n, b))
    aux = module.phys('test', tids)
    te = np.arange(len(tids))
    folder = WS / 'forecast_density_models'
    tests, folds = [], {}
    for fold in range(5):
        saved = _fold(model_root, folder / f'fold{fold}.npz')
        model = _load(module.Dynamics(c.shape[-1] - 3), Path(model_root) / folder / f'forecast{fold}.pt')
        H = module.chart(model, gpu, te)
        x = np.column_stack([aux, np.arcsinh(H)]).astype(np.float32)
        xx = torch.tensor(((x - saved['mu']) / saved['sd']).clip(-7, 7))
        head = torch.nn.Sequential(
            torch.nn.Linear(x.shape[1], 128), torch.nn.LayerNorm(128), torch.nn.SiLU(), torch.nn.Dropout(0.21),
            torch.nn.Linear(128, 80), torch.nn.LayerNorm(80), torch.nn.SiLU(), torch.nn.Dropout(0.18),
            torch.nn.Linear(80, 1))
        head = _load(head, Path(model_root) / folder / f'head{fold}.pt')
        with torch.no_grad():
            tests.append(head(xx)[:, 0].sigmoid().numpy())
        folds[folder / f'fold{fold}.npz'] = {'Ht': H}
    return _finish(stage, model_root, work, 'lawforecast', tids, tests, folds)


def block(stage, model_root, work):
    module = _module('white_block_kernel', work)
    torch.set_num_threads(1)
    tids, lines, planes, counts = _fresh(module, 'block_kernel_test.npz', lambda: module.charts('test'))
    #training holds l/p as bf16 (exact for the float16 charts) and n as float32
    values = [torch.tensor(lines, dtype=torch.bfloat16), torch.tensor(planes, dtype=torch.bfloat16),
              torch.tensor(counts, dtype=torch.float32)]
    if _cpu_port(model_root):
        #plain float32 einsum cannot mix bf16 inputs with float32 kernels; the cast is lossless
        values = [v.float() for v in values]
    ad = module.phys('test', tids)
    te = np.arange(len(tids))
    folder = WS / 'block_kernel_models'
    tests, folds = [], {}
    for fold in range(5):
        saved = _fold(model_root, folder / f'fold{fold}.npz')
        aux = np.column_stack([ad, np.arcsinh(_law(work, fold, tids))])
        data = [*values, torch.tensor(((aux - saved['mu']) / saved['sd']).clip(-7, 7), dtype=torch.float32)]
        net = _load(module.LearnedKernel(aux.shape[1]), Path(model_root) / folder / f'fold{fold}.pt')
        out = []
        with torch.no_grad():
            for start in range(0, len(te), 256):
                j = torch.as_tensor(te[start:start + 256])
                with _precision(model_root):
                    out.append(net(*(s[j] for s in data)).float().sigmoid())
        pt = torch.cat(out).numpy()
        tests.append(pt)
        folds[folder / f'fold{fold}.npz'] = {'pt': pt}
    return _finish(stage, model_root, work, 'lawblock', tids, tests, folds)


ADAPTERS = {
    'white_shared_net.py': shared, 'white_mixture_program.py': conditional,
    'white_predictive_density.py': predictive, 'white_fourier_kernel.py': fourier,
    'white_rank_likelihood.py': rankdensity, 'white_forecast_density.py': forecast,
    'white_block_kernel.py': block,
}


def supported(stage):
    return Path(stage['argv'][0]).name in ADAPTERS


def predict_stage(stage, model_root, work):
    return ADAPTERS[Path(stage['argv'][0]).name](stage, Path(model_root), Path(work))
