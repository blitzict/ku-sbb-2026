# CPU port; original experiments/weight_space/white_crop_partition.py, SHA256 73df67178c931e4bc3602a4a52ebdb684adf9b4c575d010b2756275e725baa33
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Project priors from an independent source split onto each corresponding real crop."""
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
from scipy.special import logsumexp
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from white_law_head import unpack
from partition_evidence import ROOT, OUT

def generate(view, seed):
    base = OUT / f'law_outer{seed}'
    root = OUT / f'view_{view}'
    dest = root / f'law_outer{seed}'
    dest.mkdir(exist_ok=True)
    arrays = []
    for part in ('train', 'test'):
        with np.load(root / f'wnlaw_physical_{part}.npz') as a:
            arrays.append((a['ids'], a['values'].astype(float)))
    ids = arrays[0][0]
    y = pd.read_csv(ROOT / 'data/train_labels.csv', dtype={'id': str}).set_index('id').loc[ids, 'label']
    for f, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(ids, y)):
        with np.load(base / f'fold{f}_rows.npz') as a:
            assert np.array_equal(a['train_ids'], ids[tr]) and np.array_equal(a['validation_ids'], ids[va])
        prior = json.loads((base / f'fold{f}_prior.json').read_text())
        for part, (ix, M) in zip(('train', 'test'), arrays):
            _, L, F = unpack(M, np.asarray(prior['law']))
            score = logsumexp(L + np.log(prior['change']), 1) - logsumexp(L + np.log(prior['null']), 1)
            value = np.column_stack([score, F]).clip(-80, 80)
            frame = pd.DataFrame(value, index=pd.Index(ix, name='id'), columns=[f'v{i}' for i in range(value.shape[1])])
            check = pd.read_parquet(root / f'law_outer0/fold{f}_{part}.parquet', columns=[])
            assert check.index.equals(frame.index)
            assert np.isfinite(value).all()
            frame.to_parquet(dest / f'fold{f}_{part}.parquet')
        np.savez_compressed(dest / f'fold{f}_rows.npz', train_ids=ids[tr], validation_ids=ids[va])
    print('independent base-training split ready', view, seed, flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('seed', type=int)
    p.add_argument('views', nargs='+')
    a = p.parse_args()
    with threadpool_limits(limits=1):
        for view in a.views:
            generate(view, a.seed)
