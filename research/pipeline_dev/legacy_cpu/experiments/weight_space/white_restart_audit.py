# CPU port; original experiments/weight_space/white_restart_audit.py, SHA256 fa014b406ae402414094ccbe42a4351ed145f7a8578976d792e10013ec1eaf0b
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Isolate actual cold-start evidence from the break classifier which also sees it."""
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
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from partition_evidence import OUT
import white_loop as wl

def reduce(raw):
    x = raw.reshape(len(raw), 4, 4, 4, 2)
    c = {}
    for contrast, title in ((1, 'cold_over_continuation'), (3, 'positive_over_negative')):
        for channel, name in enumerate(('raw', 'pulse')):
            for k, size in enumerate((1, 3, 6, 12)):
                a = x[:, contrast, k, :, channel]
                c[f'{title}_{name}_adjusted{size}'] = a[:, 2]
                c[f'{title}_{name}_forward_reverse{size}'] = a[:, 0] - a[:, 1]
    return c

def run():
    q, ids, tids, y, wt, wn, miss = wl.load()
    oof = np.empty((len(y), 128))
    pt = []
    for f in range(5):
        with np.load(OUT / f'restart_density_models/fold{f}.npz') as a:
            va = q[f's0_f{f}_va']
            assert np.array_equal(ids[va], a['validation_ids'])
            oof[va] = a['evidence']
            pt.append(a['test_evidence'])
    for part, ix, a in (('train', ids, oof), ('test', tids, np.mean(pt, 0))):
        x = pd.DataFrame(reduce(a), index=ix).rename_axis('id')
        assert np.isfinite(x).all().all()
        x.to_parquet(OUT / f'cold_start_effect_{part}.parquet')
        if part == 'train':
            for name in ('cold_over_continuation_raw_adjusted6', 'positive_over_negative_raw_adjusted6'):
                print(name, [round(roc_auc_score(y[m], x.loc[m, name]), 5) for m in (np.ones(len(y), bool), ~miss, miss | (y == 0), miss | wn & (y == 0))], flush=True)
if __name__ == '__main__':
    run()
