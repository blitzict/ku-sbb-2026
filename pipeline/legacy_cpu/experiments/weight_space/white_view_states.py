# CPU port; original experiments/weight_space/white_view_states.py, SHA256 877e0130e9d54ac32e303cffa8cc474d240d9a92ef905349ad9fd67072b2cb54
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Recompute the five jump-law tables from the raw, view-specific state fits."""
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
import state_filter
import white_alt_innov
import white_jump_priors
from partition_evidence import OUT

def build(view):
    folder = OUT / f'view_{view}'
    assert (folder / 'READY').exists()
    sources = ('pulse', 'state', 'nullstate', 'prestate', 'smoothvol')
    for part in ('train', 'test'):
        with patch.object(state_filter, 'OUT', folder):
            if not (folder / f'state_streams_{part}.npz').exists():
                state_filter.run(part)
        with patch.object(white_alt_innov, 'OUT', folder):
            white_alt_innov.prepare(part)
        with patch.object(white_jump_priors, 'OUT', folder):
            for stream in sources:
                white_jump_priors.extract(part, stream)
                data = pd.read_parquet(folder / f'jump_law_{stream}_{part}.parquet')
                old = pd.read_parquet(OUT / f'jump_law_{stream}_{part}.parquet')
                assert data.index.equals(old.index) and data.columns.equals(old.columns) and np.isfinite(data).all().all()
    (folder / 'STATES_READY').write_text('Five jump-law evidence tables are refitted on this raw view and verified for both partitions.')
    print('STATES_READY', view, flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('views', nargs='+', choices=('c1', 'c2', 'c3', 'c4', 'rev'))
    with torch.inference_mode(), threadpool_limits(limits=1):
        torch.set_num_threads(1)
        for view in p.parse_args().views:
            build(view)
