# CPU port; original experiments/weight_space/white_repeat_probe.py, SHA256 0dc6086428f35ece76c915611454fec94ae8833e3b6b5e25846b3c6b884a2641
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Audit repeat-lag candidates and expose their compact, label-blind scores."""
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
from scipy.special import expit
from partition_evidence import OUT
import white_loop as wl

def derive(a):
    out = {}
    for src in ('level', 'innovation', 'amplitude', 'short_memory'):
        h = a.filter(like=src + '_3_64_').copy()
        h.columns = h.columns.str.removeprefix(src + '_3_64_')
        reliability = expit((np.maximum(h.pre, h.post) - 1.0) / 0.85)
        for key in ('rotate', 'memory', 'new_period', 'reset'):
            out[src + '_' + key] = h[key] * reliability
        out[src + '_validated'] = np.minimum(h.pre, h.replicate_pre)
        out[src + '_started'] = h.start
    return pd.DataFrame(out, index=a.index).clip(-35, 70)

def main():
    _, ids, tids, y, wt, wn, miss = wl.load()
    for part, ix in (('train', ids), ('test', tids)):
        a = pd.read_parquet(OUT / f'seasonal_memory_{part}.parquet').loc[ix]
        d = derive(a)
        d.to_parquet(OUT / f'repeat_compact_{part}.parquet')
        if part != 'train':
            continue
        for src in ('level', 'innovation', 'amplitude', 'short_memory'):
            h = a.filter(like=src + '_3_64_').copy()
            h.columns = h.columns.str.removeprefix(src + '_3_64_')
            for name, mask in (('all nonbreaks', y == 0), ('white nonbreaks', wn & (y == 0)), ('821', miss)):
                selected = (h.pre > 3) & mask
                repeat = (h.replicate_pre > 1) & selected
                print(src, name, 'n', mask.sum(), 'pre log BF>3', selected.sum(), 'same lag held out log BF>1', repeat.sum())
if __name__ == '__main__':
    main()
