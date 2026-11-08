# CPU port; original experiments/weight_space/white_outer_priors.py, SHA256 fcc68809d868f7e70a7759eb6a3b3ad11623c0cfeaa05bb7d64f73de0da5be17
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Compute train, validation and test features from one outer-training prior."""
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
from scipy.special import logsumexp
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
import white_loop as wl
from white_law_head import LAWS, weights, unpack
from partition_evidence import OUT, ROOT

def load(tag):
    data = []
    ids = []
    for part in ('train', 'test'):
        with np.load(OUT / f'wnlaw_{tag}_{part}.npz') as a:
            ids.append(pd.Index(a['ids']))
            data.append(a['values'].astype(float))
    g = pd.read_parquet(ROOT / 'tune_out/generator_segments.parquet', columns=['id', 'part', 'p_bic'])
    wn = g[g.part == 'pre'].set_index('id').loc[ids[0], 'p_bic'].to_numpy() == 0
    y = pd.read_csv(ROOT / 'data/train_labels.csv', dtype={'id': str}).set_index('id').loc[ids[0], 'label'].to_numpy()
    return (*ids, np.vstack(data), wn, y)

def fit_on(M, wn, y, train):
    ref = dict(normal=0.43, student=0.24, contam=0.13, bimodal=0.16, expower=0.04)
    prior = np.array([ref[tag] / sum((z[0] == tag for z in LAWS)) for tag, _, _ in LAWS])
    ix = train[wn[train]]
    k = len(LAWS)
    law = weights(M[ix, -2 * k:-k], prior, 150.0)
    p, L, F = unpack(M, law)
    params = [weights(L[ix[y[ix] == label]], p, 180.0) for label in (0, 1)]
    lr = logsumexp(L + np.log(params[1]), axis=1) - logsumexp(L + np.log(params[0]), axis=1)
    return (np.column_stack([lr, F]).clip(-80, 80), dict(law=law.tolist(), null=params[0].tolist(), change=params[1].tolist()))

def generate(seed, tag):
    ids, tids, M, wn, y = load(tag)
    folder = OUT / f'law_outer{seed}' if tag == 'physical' else OUT / f'law_outer{seed}_{tag}'
    folder.mkdir(exist_ok=True)
    for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(ids, y)):
        f, params = fit_on(M, wn, y, tr)
        for part, index, arr in (('train', ids, f[:len(y)]), ('test', tids, f[len(y):])):
            frame = pd.DataFrame(arr, index=index, columns=[f'v{j}' for j in range(f.shape[1])])
            frame.to_parquet(folder / f'fold{fold}_{part}.parquet')
        (folder / f'fold{fold}_prior.json').write_text(json.dumps(params))
        np.savez(folder / f'fold{fold}_rows.npz', train_ids=ids[tr].to_numpy(dtype=str), validation_ids=ids[va].to_numpy(dtype=str))
        print('verified prior fit', seed, fold, len(tr), len(va), flush=True)
    return folder
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    p.add_argument('--tag', default='physical')
    a = p.parse_args()
    with threadpool_limits(limits=1):
        for seed in a.seeds:
            generate(seed, a.tag)
