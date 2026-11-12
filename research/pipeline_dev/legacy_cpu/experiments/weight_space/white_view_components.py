# CPU port; original experiments/weight_space/white_view_components.py, SHA256 4f872981e943e64a02b26fee8955126b423ebe174bcdfea9376af6f0790d8b1f
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Measure individual noise-component moves and latent levels on genuine views."""
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
import white_component_moves
import white_latent_level

def run(view):
    root = OUT / f'view_{view}'
    assert (root / 'STATES_READY').exists()
    for part in ('train', 'test'):
        with patch.object(white_component_moves, 'OUT', root):
            white_component_moves.extract(part, 'state')
        with patch.multiple(white_latent_level, ROOT=root, OUT=root):
            white_latent_level.extract('pulse', part)
        for name in ('component_moves_state', 'latent_level_pulse'):
            a, b = [pd.read_parquet(d / f'{name}_{part}.parquet') for d in (OUT, root)]
            assert a.index.equals(b.index) and a.columns.equals(b.columns) and np.isfinite(b).all().all()
    (root / 'COMPONENTS_READY').write_text('Exact component and latent-level tables verified on both partitions.')
    print('COMPONENTS_READY', view, flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('views', nargs='+')
    with threadpool_limits(limits=1), torch.inference_mode():
        torch.set_num_threads(1)
        for v in p.parse_args().views:
            run(v)
