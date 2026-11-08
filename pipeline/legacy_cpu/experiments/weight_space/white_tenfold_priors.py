# CPU port; original experiments/weight_space/white_tenfold_priors.py, SHA256 7a360a766aa129a723d35826e9eeb3c73a988b3054dbf487a72619936ed7743f
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Build the same physical prior from 90% of the sources for a ten-fold tree."""
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
from white_outer_priors import load, fit_on
from white_law_head import unpack
from partition_evidence import OUT

def main(seed):
    ids, tids, orig, mask, y = load('physical')
    name = f'law_outer10x{seed}'
    folds = list(StratifiedKFold(10, shuffle=True, random_state=seed).split(ids, y))
    for view in ('', 'c1', 'c2'):
        root = OUT / ('view_' + view) if view else OUT
        dest = root / name
        dest.mkdir(exist_ok=True)
        if view:
            data = []
            for part, ix in (('train', ids), ('test', tids)):
                with np.load(root / f'wnlaw_physical_{part}.npz') as a:
                    assert np.array_equal(ix, a['ids'])
                    data.append(a['values'].astype(float))
            observed = np.vstack(data)
        for f, (tr, va) in enumerate(folds):
            marker = dest / f'fold{f}_rows.npz'
            if marker.exists():
                with np.load(marker) as a:
                    assert np.array_equal(ids[tr], a['train_ids']) and np.array_equal(ids[va], a['validation_ids'])
                continue
            if view:
                prior = json.loads((OUT / name / f'fold{f}_prior.json').read_text())
                _, L, F = unpack(observed, np.asarray(prior['law']))
                score = logsumexp(L + np.log(prior['change']), 1) - logsumexp(L + np.log(prior['null']), 1)
                v = np.column_stack((score, F)).clip(-80, 80)
            else:
                v, prior = fit_on(orig, mask, y, tr)
                (dest / f'fold{f}_prior.json').write_text(json.dumps(prior))
            assert np.isfinite(v).all()
            for part, ix, values in (('train', ids, v[:len(ids)]), ('test', tids, v[len(ids):])):
                frame = pd.DataFrame(values, index=pd.Index(ix, name='id'), columns=[f'v{j}' for j in range(v.shape[1])])
                frame.to_parquet(dest / f'fold{f}_{part}.parquet')
            np.savez_compressed(marker, train_ids=ids[tr].to_numpy(dtype=str), validation_ids=ids[va].to_numpy(dtype=str))
        print('ten-fold priors ready', name, view or 'original', flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('seed', type=int)
    with threadpool_limits(limits=1):
        main(p.parse_args().seed)
