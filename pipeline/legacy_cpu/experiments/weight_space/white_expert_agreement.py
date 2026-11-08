# CPU port; original experiments/weight_space/white_expert_agreement.py, SHA256 3a56f16d8f2ad17dfac761fd4eeb4159e2fad11701bff2ad452a002df221f599
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Fit the rank correction to agreements between independently built break detectors."""
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
from pathlib import Path
from unittest.mock import patch
import numpy as np
from threadpoolctl import threadpool_limits
from white_joint import read_member
import white_local_rank_weights as rank
GROUPS = {'crop_trees': ('claude_cropvolbag', 'claude_revlow', 'claude_revorig', 'lawcrop_c1_c2_rev'), 'rank_trees': ('lawrank_original', 'lawrank_crops'), 'noise_law_trees': ('claude_law', 'claude_jump', 'lawtree_both', 'lawvoltree'), 'likelihood': ('lawshared', 'lawconditional', 'lawpredictive', 'lawfourier'), 'sequence': ('lawblock', 'lawforecast', 'lawretrieval', 'net')}
BASE = rank.inputs

def constructor(extras):

    def inputs(ids, tids, models, physics):
        calibrate, paths = BASE(ids, tids, models, physics)
        families = dict(GROUPS)
        if extras:
            families['additional_density'] = extras
        panels = []
        for names in families.values():
            scores = []
            for name in names:
                arr, pp = read_member(name, ids, tids)
                paths += pp
                scores.append(arr[:, 0])
            panels.append(np.column_stack(scores))

        def make(ix, z):
            ff = calibrate(ix, z)
            groups = []
            uncertainty = []
            for panel in panels:
                d = (panel[ix] - z[:, None]).clip(-6, 6)
                groups.append(d.mean(1))
                uncertainty.append(d.std(1))
            a = np.column_stack(groups)
            s = np.column_stack(uncertainty)
            terms = [ff, 0.68 * a, 0.63 * s]
            consensus = a.mean(1, keepdims=True)
            b = np.tanh((a - consensus) / 1.4)
            i, j = np.triu_indices(b.shape[1])
            products = b[:, i] * b[:, j]
            terms.extend([0.8 * products, 0.85 * b, 0.7 * b * s, 0.45 * s * abs(a), 0.55 * products * np.tanh((z[:, None] + 0.6) / 1.35)])
            for c, width in ((-1.7, 0.85), (-0.35, 0.72), (0.85, 0.98)):
                window = np.exp(-0.5 * ((z[:, None] - c) / width) ** 2)
                terms.extend([0.65 * window * a, 0.55 * window * s, 0.55 * window * products])
            return np.ascontiguousarray(np.column_stack(terms))
        return (make, [*dict.fromkeys(paths), Path(__file__).resolve()])
    return inputs
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('parent')
    p.add_argument('tag')
    p.add_argument('--new', nargs='*', default=[])
    a = p.parse_args()
    models = [*GROUPS['crop_trees'][:3], *GROUPS['rank_trees'], 'lawforecast', 'lawretrieval', *a.new]
    with threadpool_limits(limits=1), patch.object(rank, 'inputs', constructor(a.new)):
        rank.run(a.parent, a.tag, models, True)
