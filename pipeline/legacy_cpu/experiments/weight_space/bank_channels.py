# CPU port; original experiments/weight_space/bank_channels.py, SHA256 0480d326d9c3f890d20373fae8bf2e2511af7765d06e461a259acc423b0a33eb
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Connect canonical OOF physical-evidence experts to the locked readout."""
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
from scipy.special import expit

def tags(mode):
    names = ['base', 'adjusted'] if mode == 'both' else [mode]
    return ['shape_bank_' + m for m in names] + ['curve', 'periodic', 'innov_v2', 'mixture_law']

def design(mode, t, z, ix):
    x = np.column_stack([t[p].iloc[ix].score for p in tags(mode) if p.startswith('shape_bank_')])
    delta = (x - z[:, None]).clip(-9, 9)
    per = t['periodic'].iloc[ix]
    season = (per.gate * expit((per.period - 9) / 1.5) * expit((38 - per.period) / 2)).to_numpy()
    vals = [delta, np.tanh(z[:, None] / 2) * delta]
    for gate in [season, t['curve'].iloc[ix].gate, t['innov_v2'].iloc[ix].pre_walk, t['mixture_law'].iloc[ix].gate]:
        vals.append(np.asarray(gate)[:, None] * delta)
    return np.column_stack(vals)
