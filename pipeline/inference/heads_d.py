"""Saved neural law heads and physical readouts (group D): load fitted state, predict new query rows.

Every adapter rebuilds the query-side inputs for the new series from the prediction workspace with the training
source's own functions, loads the fitted parameters (state dicts, normalization, rank knots, readout weights) from
model_root and writes the declared outputs with training/OOF fields copied verbatim and query fields replaced.
Nothing is refitted: rank knots, centers/scales and prior laws come from the saved fold records.
"""
import hashlib
import importlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit, logsumexp

from .common import destination, replace_test_fields, require, save_npz

WS = Path('tune_out/weight_space')
P = np.linspace(0.002, 0.998, 47)


def _module(name):
    torch.set_num_threads(1)
    return importlib.import_module(name)


def _state(path):
    state = torch.load(require(path), map_location='cpu', weights_only=True)
    if not all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point()):
        raise ValueError(f'non-finite weights in {path}')
    return state


def _net(model, state):
    model.load_state_dict(state)
    model.eval()
    return model


def _npz(path):
    with np.load(require(path), allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def _chunks(count, size):
    #same chunk count as the training predict loops (np.array_split over the query rows)
    return np.array_split(np.arange(count), max(1, count // size))


def _main(model_root, work, name, tids, **values):
    relative = WS / (name + '.npz')
    replace_test_fields(model_root, work, relative, tids=np.asarray(tids), **values)
    return [Path(work) / relative]


def _record(model_root, work, relative, **values):
    #fold record: keep validation/train fields, replace query fields
    replace_test_fields(model_root, work, relative, **values)
    return [Path(work) / relative]


def _apply_ranks(raw, q):
    """white_view_network.ranks with the saved per-fold knots q (47, d); no quantiles are estimated."""
    out = np.zeros_like(raw, dtype=np.float16)
    if q.shape[1] != raw.shape[2]:
        raise ValueError(f'rank knots have {q.shape[1]} columns, query has {raw.shape[2]}')
    for k, knots in enumerate(q.T):
        knots = knots[np.isfinite(knots)]
        v, i, c = np.unique(knots, return_inverse=True, return_counts=True)
        if len(v) < 2:
            continue
        levels = np.bincount(i, weights=P[:len(knots)]) / c
        arr = raw[:, :, k].copy()
        arr = np.where(np.isfinite(arr), arr, np.median(v))
        out[:, :, k] = (2 * np.interp(arr, v, levels) - 1).astype(np.float16)
    return out


def _prior(work, seed, fold, ids, view=''):
    folder = Path(work) / WS / (f'view_{view}' if view else '') / f'law_outer{seed}'
    return pd.read_parquet(folder / f'fold{fold}_test.parquet').loc[ids]


#white_retrieval.py

RETRIEVAL_TABLES = ['ar_law', 'curve', 'state', 'periodic', 'phase_law', 'trend', 'bounds', 'innov_v2', 'mixture_law',
                    'noise_tests', 'walk_jump', 'volatility_law', 'native_law_state']


def _retrieval_block(root, view, part, ix):
    #white_retrieval.inputs for one (view, part); root is the workspace holding that part's tables
    folder = Path(root) / WS / (f'view_{view}' if view else '')
    terms = []
    for name in RETRIEVAL_TABLES:
        tab = pd.read_parquet(folder / f'{name}_{part}.parquet').loc[ix]
        if name == 'native_law_state':
            tab = tab[[c for c in tab if '_pair' not in c and '_from' not in c and ('_to' not in c)]]
        terms.append(tab.to_numpy(dtype=np.float32))
    for stream in ('pulse', 'state', 'prestate', 'nullstate', 'smoothvol'):
        tab = pd.read_parquet(folder / f'jump_law_{stream}_{part}.parquet').loc[ix]
        cols = [c for c in tab if '_law' not in c and '_from' not in c and ('_to' not in c)]
        terms.append(tab[cols].to_numpy(dtype=np.float32))
    return np.column_stack(terms)


def retrieval(stage, model_root, work):
    module = _module('white_retrieval')
    with np.load(Path(work) / 'tune_out/var/mat_ns3_c7.npz', allow_pickle=True) as source:
        tids = source['tid'].astype(str)
    query = np.stack([_retrieval_block(work, v, 'test', tids) for v in ('', 'c1', 'c2')], 1)
    tests = []
    for f in range(5):
        record = _npz(Path(model_root) / WS / f'lawretrieval_models/fold{f}.npz')
        bank_ids, knots = record['bank_ids'].astype(str), record['knots']
        #memory bank: saved training rows (view 0) through the saved encoder, read from the trained workspace
        bank_raw = _retrieval_block(model_root, '', 'train', bank_ids)[:, None, :]
        bank_x = torch.as_tensor(_apply_ranks(bank_raw, knots))
        x = torch.as_tensor(_apply_ranks(query, knots))
        net = _net(module.Retrieval(x.shape[-1]), _state(Path(model_root) / WS / f'lawretrieval_models/fold{f}.pt'))
        yy = torch.tensor(pd.read_csv(require(Path(model_root) / 'data/train_labels.csv'), dtype={'id': str})
                          .set_index('id').loc[bank_ids, 'label'].to_numpy(), dtype=torch.float32)
        with torch.no_grad():
            out = [net.encode(bank_x[j, 0].float()) for j in _chunks(len(bank_ids), 700)]
            #bank row identifiers only mask a query row against itself; query rows are never bank rows
            ref = (torch.cat([b[1] for b in out]).permute(1, 2, 0).contiguous(), torch.cat([b[2] for b in out]), yy,
                   torch.arange(len(bank_ids)))
            result = []
            for j in _chunks(len(tids), 210):
                js = torch.as_tensor(j)
                own = js + len(bank_ids) + 1_000_000_000
                scores = [net(x[js, k].float(), own, ref)[0] for k in range(3)]
                result.append(((scores[0] + 0.1 * (scores[1] + scores[2])) / 1.2).sigmoid().cpu().numpy())
        tests.append(np.concatenate(result))
    return _main(model_root, work, 'lawretrieval', tids, test=np.mean(tests, 0))


#white_patch_forecast.py

def patch_forecast(stage, model_root, work):
    module = _module('white_patch_forecast')
    from white_block_kernel import phys
    tids, patch = module.make('test')
    patch = patch.astype(np.float32)
    aux = phys('test', tids)
    tests = []
    for f in range(5):
        folder = Path(model_root) / WS / 'patch_forecast_models'
        r = _npz(folder / f'fold{f}.npz')
        x = torch.as_tensor(((patch - r['mu']) / r['sd']).clip(-7, 7), dtype=torch.float32)
        p = torch.as_tensor(((aux - r['m']) / r['s']).clip(-7, 7), dtype=torch.float32)
        model = _net(module.Predictor(x.shape[-1], p.shape[-1]), _state(folder / f'fold{f}.pt'))
        with torch.no_grad():
            tests.append(np.concatenate([model(x[j], p[j])[0].sigmoid().cpu().numpy() for j in _chunks(len(tids), 180)]))
    return _main(model_root, work, 'lawpatchforecast', tids, test=np.mean(tests, 0))


#white_restart_density.py

def restart(stage, model_root, work):
    module = _module('white_restart_density')
    from white_block_kernel import phys
    from torch import nn
    tids, h = module.make('test')
    H = torch.as_tensor(h)
    aux = phys('test', tids)
    written, tests = [], []
    folder = Path(model_root) / WS / 'restart_density_models'
    for f in range(5):
        r = _npz(folder / f'fold{f}.npz')
        net = _net(module.Densities(H.shape[-1] - module.T - 8), _state(folder / f'density{f}.pt'))
        feat = module.chart(net, H, np.arange(len(tids)))
        xx = np.column_stack([aux, np.arcsinh(feat)]).astype(np.float32)
        x = torch.tensor(((xx - r['center']) / r['scale']).clip(-7, 7))
        head = nn.Sequential(nn.Linear(x.shape[1], 144), nn.LayerNorm(144), nn.SiLU(), nn.Dropout(0.22), nn.Linear(144, 80),
                             nn.LayerNorm(80), nn.SiLU(), nn.Dropout(0.18), nn.Linear(80, 1))
        head = _net(head, _state(folder / f'head{f}.pt'))
        with torch.no_grad():
            tests.append(head(x)[:, 0].sigmoid().cpu().numpy())
        written += _record(model_root, work, WS / f'restart_density_models/fold{f}.npz', test_evidence=feat)
    return _main(model_root, work, 'lawrestart', tids, test=np.mean(tests, 0)) + written


#white_rank_dependence.py

def rank_dependence(stage, model_root, work):
    module = _module('white_rank_dependence')
    from white_block_kernel import phys
    seed = int(stage['argv'][1])
    tag = f'law_rank_dependence_s{seed}'
    tids, b, lengths, cuts = module.charts('test')
    chart = torch.as_tensor(b, dtype=torch.float16)
    scale = torch.as_tensor(np.sqrt(cuts * (lengths - cuts) / lengths), dtype=torch.float32)
    physical = phys('test', tids)
    written, outputs = [], []
    for f in range(5):
        relative = WS / f'{tag}_models/fold{f}.npz'
        r = _npz(Path(model_root) / relative)
        features = np.column_stack((physical, np.arcsinh(_prior(work, seed, f, tids).to_numpy())))
        c = torch.as_tensor(((features - r['mu']) / r['sd']).clip(-8, 8), dtype=torch.float32)
        model = _net(module.RankPairs(c.shape[1]), _state(Path(model_root) / WS / f'{tag}_models/fold{f}.pt'))
        with torch.no_grad():
            pred = torch.cat([model(chart[j], c[j], scale[j]).float().sigmoid() for j in _chunks(len(tids), 180)]).cpu().numpy()
        outputs.append(pred)
        written += _record(model_root, work, relative, test=pred)
    return _main(model_root, work, tag, tids, test=np.mean(outputs, axis=0)) + written


#white_shock_quantiles.py and white_ordered_paths.py (same detector, different curves)

def _detector(model_root, work, builder, folder, tags):
    module = _module('white_shock_quantiles')
    from white_block_kernel import phys
    tids, raw = builder('test')
    aux = phys('test', tids)
    written, tests = [], []
    for f in range(5):
        relative = WS / folder / f'fold{f}.npz'
        r = _npz(Path(model_root) / relative)
        laws = pd.read_parquet(Path(work) / WS / f'law_outer29/fold{f}_test.parquet').loc[tids].to_numpy()
        context = np.column_stack([aux, np.arcsinh(laws)])
        x = torch.tensor(((raw - r['mu']) / r['sd']).clip(-7, 7), dtype=torch.float32)
        p = torch.tensor(((context - r['m']) / r['s']).clip(-7, 7), dtype=torch.float32)
        net = _net(module.Detector(context.shape[1], raw.shape[1]), _state(Path(model_root) / WS / folder / f'fold{f}.pt'))
        with torch.no_grad():
            pred = torch.cat([net(x[j], p[j]).float().sigmoid() for j in _chunks(len(tids), 255)]).cpu().numpy()
        tests.append(pred)
        written += _record(model_root, work, relative, pt=pred)
    mean = np.mean(tests, 0)
    for k, tag in enumerate(tags):
        written = _main(model_root, work, tag, tids, test=mean[:, k]) + written
    return written


def shock_quantiles(stage, model_root, work):
    module = _module('white_shock_quantiles')
    return _detector(model_root, work, module.curves, 'shock_quantile_models',
                     ('law_shock_quantile_control', 'law_shock_quantiles'))


def ordered_paths(stage, model_root, work):
    module = _module('white_ordered_paths')
    return _detector(model_root, work, module.charts, 'ordered_path_models', ('law_path_control', 'law_ordered_paths'))


#white_physical_net.py

PHYSICAL_FIELDS = [('state', ['pre2', 'pre_ar1', 'pre_cluster', 'post_cluster', 'walk']), ('curve', ['gate']),
                   ('periodic', ['gate']), ('pulse', ['gate']), ('innov_v2', ['white', 'pre_white'])]


def physical_net(stage, model_root, work):
    module = _module('white_physical_net')
    module.seq.a.hq = 0
    epochs = int(stage['argv'][stage['argv'].index('--epochs') + 1]) if '--epochs' in stage['argv'] else 8
    tids, x, n, b = module.signals('test')
    gpu = [torch.as_tensor(v) for v in (x, n, b)]
    field = np.column_stack([pd.read_parquet(Path(work) / WS / f'{name}_test.parquet', columns=cols).loc[tids].to_numpy()
                             for name, cols in PHYSICAL_FIELDS])
    views = [(False, False), (True, True), (True, False), (False, True)]
    written = []
    for fold in range(5):
        relative = WS / f'physical_ft_{epochs}/fold{fold}.npz'
        r = _npz(Path(model_root) / relative)
        features = np.column_stack([np.arcsinh(_prior(work, 0, fold, tids).to_numpy()), field])
        aux = torch.as_tensor(((features - r['center']) / r['sd']).clip(-6, 6), dtype=torch.float32)
        net = _net(module.Detector(aux.shape[1]), _state(Path(model_root) / WS / f'physical_ft_{epochs}/fold{fold}.pt'))
        pt = module.predict(net, gpu, aux, np.arange(len(tids)), views)
        written += _record(model_root, work, relative, pt=pt)
    return written


#white_extended.py (physical readouts) and white_checkpoint.py (frozen parent offsets)

def _physical_design(work, names, tids, z):
    """white_extended.evaluate design() for the query rows; every term is row-wise."""
    from white_nested import law_design
    folder = Path(work) / WS

    def frame(tag, cols=None):
        return pd.read_parquet(folder / f'{tag}_test.parquet', columns=cols).loc[tids]
    white = frame('innov_v2', ['pre_white']).pre_white.to_numpy()
    curve = frame('curve', ['gate']).gate.to_numpy()
    gate = ((1 - frame('innov_v2', ['pre_walk']).pre_walk.to_numpy()) * (1 - frame('periodic', ['gate']).gate.to_numpy())
            * (0.15 + 0.85 * np.maximum(white, frame('innov_v2', ['white']).white.to_numpy()))
            * (1 - 0.7 * frame('state', ['pre2']).pre2.to_numpy()))
    walk = frame('innov_v2').pre_walk.to_numpy()
    static, features = [], []
    for name in names:
        if name in ('pulse', 'state'):
            a = frame('jump_law_' + name)
            h = np.arcsinh(a.clip(-24, 60).to_numpy())
            target = 0.2 + 0.8 * white if name == 'pulse' else 1 - 0.35 * white
            static.append(h * target[:, None])
            joint = a[['level2', 'scale2', 'shape0', 'shape1']].clip(-14, 35).to_numpy()
            joint = np.logaddexp.reduce(joint, axis=1) - np.log(joint.shape[1])
            features.append((joint[:, None], target))
        elif name == 'walk':
            a = frame('walk_jump')
            static.append(walk[:, None] * np.arcsinh(a.clip(-25, 60).to_numpy()))
            features.append((a.total.to_numpy()[:, None], walk))
        elif name == 'net':
            tests = []
            for f in range(5):
                with np.load(folder / f'physical_ft_8/fold{f}.npz') as b:
                    tests.append(b['pt'])
            features.append((np.mean(tests, axis=0)[:, None], np.ones(len(white))))
        elif name in ('shape', 'deconv'):
            raise NotImplementedError(f'{name}: member not in the recipe readouts')
        else:
            with np.load(folder / f'{name}.npz') as a:
                if not np.array_equal(a['tids'].astype(str), tids):
                    raise ValueError(f'{name}: query identifiers differ')
                features.append((logit(a['test'].clip(1e-06, 1 - 1e-06))[:, None], np.ones(len(white))))
    constants = np.column_stack(static) if static else np.zeros((len(white), 0))

    def design(h):
        blocks = [law_design(h, z, gate, curve), constants]
        for a, weight in features:
            d = weight[:, None] * (a - z[:, None]).clip(-7, 8)
            blocks.extend([d, d * np.tanh(z[:, None] / 2), d * white[:, None], d * curve[:, None], d * walk[:, None]])
        return np.column_stack(blocks)
    return design


def _physical_priors(work, tids, params):
    """white_outer_priors.fit_on query rows with the saved law/null/change parameters."""
    from white_law_head import unpack
    with np.load(Path(work) / WS / 'wnlaw_physical_test.npz') as data:
        ids, values = data['ids'].astype(str), data['values'].astype(float)
    if not np.array_equal(ids, tids):
        raise ValueError('law evidence query identifiers differ')
    _, L, F = unpack(values, np.asarray(params['law']))
    lr = logsumexp(L + np.log(params['change']), axis=1) - logsumexp(L + np.log(params['null']), axis=1)
    return np.column_stack([lr, F]).clip(-80, 80)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extended(stage, model_root, work):
    from finalize import write_submission
    args = stage['argv'][1:]
    names = [x for x in args[:next((i for i, x in enumerate(args) if x.startswith('--')), len(args))]]
    weighted = '--rank' in args
    name = 'physical_' + '_'.join(names) + ('_rank' if weighted else '')
    package = Path('new weights') / f'wn_{name}_candidate'
    recipe = json.loads(require(Path(model_root) / package / 'weights.json').read_text())
    with np.load(Path(work) / WS / 'white_loop_parent.npz') as q:
        tids, zt = q['tids'].astype(str), q['zt']
    h = _physical_priors(work, tids, recipe['priors'])
    f = _physical_design(work, names, tids, zt)(h)
    w = np.array(recipe['weights'])
    written = []
    #training/OOF report and fitted readout: verbatim
    for relative in (WS / f'wn_{name}.json', WS / f'wn_{name}_oof.npz', package / 'weights.json', package / 'replay.py'):
        shutil.copyfile(require(Path(model_root) / relative), destination(work, relative))
        written.append(Path(work) / relative)
    file = destination(work, package / 'submission.csv')
    sub = write_submission(expit(zt + f @ w), pd.Index(tids), file)
    save_npz(destination(work, package / 'replay_inputs.npz'), parent_logits=zt, features=f, ids=tids,
             order=pd.Index(tids).get_indexer(sub.id))
    validation = json.loads(require(Path(model_root) / package / 'validation.json').read_text())
    validation.update(recipe_sha256=_sha(Path(work) / package / 'weights.json'), submission_sha256=_sha(file))
    destination(work, package / 'validation.json').write_text(json.dumps(validation, indent=2))
    return written + [file, Path(work) / package / 'replay_inputs.npz', Path(work) / package / 'validation.json']


def frozen_parent(stage, model_root, work):
    folder = Path(work) / 'new weights/wn_physical_pulse_state_walk_net_lawtree_both_claude_law_rank_candidate'
    with np.load(Path(work) / WS / 'white_loop_parent.npz') as q:
        tids = q['tids'].astype(str)
    sub = pd.read_csv(folder / 'submission.csv', dtype={'id': str}).set_index('id').loc[tids, 'break_probability']
    relative = WS / 'frozen874_offsets.npz'
    replace_test_fields(model_root, work, relative, tids=tids, zt=logit(sub.to_numpy()))
    return [Path(work) / relative]


#white_conditional_weights.py

def conditional_weights(stage, model_root, work):
    module = _module('white_conditional_weights')
    from white_block_kernel import phys
    module.seq.a.hq = 0
    tids, views, n, b = module.streams('test')
    data = [torch.tensor(v) for v in views]
    n, b = torch.tensor(n), torch.tensor(b)
    aux = phys('test', tids)
    vol = pd.read_parquet(Path(work) / WS / 'volatility_law_test.parquet').loc[tids].to_numpy()
    aux = np.column_stack([aux, np.arcsinh(vol.clip(-22, 40))])
    tests = []
    for fold in range(5):
        state = _state(Path(model_root) / WS / f'physical_ft_8/fold{fold}.pt')
        base = module.seq.Net().eval()
        base.load_state_dict({k[9:]: v for k, v in state.items() if k.startswith('sequence.')})
        x = torch.tensor(module.capture(base, data, n, b), dtype=torch.float32)
        r = _npz(Path(model_root) / WS / f'conditional_weight_models/fold{fold}_recipe.npz')
        c = np.column_stack([aux, np.arcsinh(_prior(work, 0, fold, tids).to_numpy())])
        c = torch.tensor(((c - r['center']) / r['scale']).clip(-7, 7), dtype=torch.float32)
        model = _net(module.TaskVectors(base.head, c.shape[1]), _state(Path(model_root) / WS / f'conditional_weight_models/fold{fold}.pt'))
        with torch.no_grad():
            tests.append(torch.cat([model(x[j], c[j]).float().sigmoid() for j in _chunks(len(tids), 400)]).cpu().numpy())
    return _main(model_root, work, 'lawvectors', tids, test=np.mean(tests, 0))


#white_weight_ensemble.py

def weight_ensemble(stage, model_root, work):
    module = _module('white_weight_ensemble')
    seed = int(stage['argv'][1])
    name = f'law_weight_ensemble_s{seed}'
    folder = Path(work) / WS
    with np.load(Path(work) / 'tune_out/var/mat_ns3_c7.npz', allow_pickle=True) as a:
        tids = a['tid'].astype(str)
        blocks = []
        for v in module.VIEWS:
            root = folder / ('view_' + v) if v else folder
            cols = [a['VT_' + v if v else 'Xt']]
            for tag, fields in module.LAWS.items():
                cols.append(pd.read_parquet(root / f'{tag}_test.parquet', columns=fields.split()).loc[tids].to_numpy())
            for tag in ('bounds', 'trend', 'phase_law', 'state'):
                cols.append(pd.read_parquet(root / f'{tag}_test.parquet').loc[tids].to_numpy())
            blocks.append(np.column_stack(cols).astype(np.float32))
    raw = np.stack(blocks, axis=1)
    with np.load(folder / 'frozen_local_rank_family_agreement/full.npz') as a:
        if not np.array_equal(a['test_ids'].astype(str), tids):
            raise ValueError('frozen offset query identifiers differ')
        offsets = torch.tensor(a['test'], dtype=torch.float32)
    K = module.K
    written, tests = [], []
    for f in range(5):
        relative = WS / f'{name}_models/fold{f}.npz'
        r = _npz(Path(model_root) / relative)
        prior = np.stack([_prior(work, seed, f, tids, v).to_numpy(np.float32) for v in module.VIEWS], 1)
        x = torch.as_tensor(_apply_ranks(np.concatenate([raw, prior], axis=2), r['knots']), dtype=torch.float32)
        net = _net(module.Network(x.shape[2]), _state(Path(model_root) / WS / f'{name}_models/fold{f}.pt'))
        val = []
        with torch.no_grad():
            for j in _chunks(len(tids), 240):
                p = []
                for v in range(3):
                    h = x[j, v][:, None, :].expand(-1, K, -1)
                    p.append(torch.stack([w.float().sigmoid().mean(1) for w in net(h, offsets[j])], 1))
                val.append((p[0] + 0.13 * p[1] + 0.13 * p[2]) / 1.26)
        pred = torch.cat(val).cpu().numpy()
        tests.append(pred)
        written += _record(model_root, work, relative, pt=pred)
    mean = np.mean(tests, 0)
    return (_main(model_root, work, name, tids, test=mean[:, 0]) + _main(model_root, work, name + '_res', tids, test=mean[:, 1])
            + written)


ADAPTERS = {
    'white_retrieval.py': retrieval, 'white_patch_forecast.py': patch_forecast, 'white_restart_density.py': restart,
    'white_rank_dependence.py': rank_dependence, 'white_shock_quantiles.py': shock_quantiles,
    'white_ordered_paths.py': ordered_paths, 'white_physical_net.py': physical_net, 'white_extended.py': extended,
    'white_checkpoint.py': frozen_parent, 'white_conditional_weights.py': conditional_weights,
    'white_weight_ensemble.py': weight_ensemble,
}


def supported(stage):
    return Path(stage['argv'][0]).name in ADAPTERS


def predict_stage(stage, model_root, work):
    return ADAPTERS[Path(stage['argv'][0]).name](stage, Path(model_root), Path(work))
