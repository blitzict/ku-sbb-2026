# CPU port; original experiments/weight_space/white_family_ablation.py, SHA256 ebd25c840e1efe589240590034647f9f4b19e43843c09e44533d259a8bed854b
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Refit the accepted pairwise readout while omitting one detector family at a time."""
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
from pathlib import Path
from threadpoolctl import threadpool_limits
from white_score_scale import main
from white_opinion_spectrum import design
FAMILY = ('original_means', 'original_variants', 'multiview_trees', 'physical_trees', 'mechanism_nets')

def ablate(packs, reg, tr, va, z, v):
    charts = {}
    for j, name in enumerate(FAMILY):
        bank = list(packs)
        bank[j] = bank[j][:, :0]
        charts[name] = design(bank, reg, tr, va, z, v)[0]
    return charts
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main(ablate, 'local_rank_without_', Path(__file__).resolve())
