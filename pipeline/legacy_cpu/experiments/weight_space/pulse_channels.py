# CPU port; original experiments/weight_space/pulse_channels.py, SHA256 dc347ca90530770530a1f345acf1f848311683d1d85f991b422aa317a9ac6dc7
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Project nuisance-cleaned network scores into the existing ensemble."""
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
    return ['pulse', 'pulse_seq', 'curve_seq', 'periodic_seq', 'periodic', 'curve', 'innov_v2']

def design(mode, t, z, ix):
    p = t['pulse'].iloc[ix]
    g = p.gate.to_numpy()
    season = t['periodic'].iloc[ix]
    wave = (season.gate * expit((season.period - 9) / 1.5) * expit((38 - season.period) / 2)).to_numpy()
    curve = t['curve'].iloc[ix].gate.to_numpy()
    walk = t['innov_v2'].iloc[ix].pre_walk.to_numpy()
    seq = lambda tag: [q.mean(1) for q in np.split(t[tag].iloc[ix].to_numpy(), 3, axis=1)]
    orig, clean, physical = seq('pulse_seq')
    _, _, detrend = seq('curve_seq')
    _, _, dewave = seq('periodic_seq')
    before = wave * dewave + (1 - wave) * detrend
    delta = np.column_stack([clean - orig, physical - before, clean - z, physical - z]).clip(-9, 9)
    f = [g[:, None] * np.column_stack([np.ones(len(z)), np.tanh(z / 2), np.clip(z, -6, 9), delta, delta * np.tanh(z[:, None] / 2), np.maximum(delta - 1, 0), np.clip(p.step - p.raw_step, -9, 9), np.log1p(p.step.clip(0, 35)), np.log1p(p.amplitude.clip(0, 60)), p.width / 3, abs(p.center) / 3]), delta[:, :2], (g * walk)[:, None] * delta]
    if mode == 'joint':
        diff = (physical - z).clip(-8, 8)
        f.extend([curve * diff, wave * diff, walk * diff])
    return np.column_stack(f)
