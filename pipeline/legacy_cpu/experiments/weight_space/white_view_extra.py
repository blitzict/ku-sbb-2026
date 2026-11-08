# CPU port; original experiments/weight_space/white_view_extra.py, SHA256 ef1f44f833224d100ff3ed09675f13384c269add2557725c6a7602afa618b4ee
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Run additional label-free physical tests on Claude's verified real views."""
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
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from partition_evidence import OUT
import white_volatility_law
import white_native_law
import white_walk

def run(view):
    root = OUT / f'view_{view}'
    assert (root / 'STATES_READY').exists()
    for p in ('train', 'test'):
        with patch.object(white_volatility_law, 'OUT', root):
            white_volatility_law.extract(p)
        with patch.multiple(white_walk, ROOT=root, OUT=root):
            white_walk.extract(p)
        with patch.object(white_native_law, 'OUT', root):
            white_native_law.extract(p, 'state')
        for name in ('volatility_law', 'walk_jump', 'native_law_state'):
            v = pd.read_parquet(root / f'{name}_{p}.parquet')
            orig = pd.read_parquet(OUT / f'{name}_{p}.parquet')
            assert orig.columns.equals(v.columns) and orig.index.equals(v.index) and np.isfinite(v).all().all()
    (root / 'EXTRA_READY').write_text('Verified volatility, walk, and native noise-law tables on both partitions.')
    print('EXTRA_READY', view, flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('views', nargs='+')
    with torch.inference_mode(), threadpool_limits(limits=1):
        torch.set_num_threads(1)
        for v in p.parse_args().views:
            run(v)
