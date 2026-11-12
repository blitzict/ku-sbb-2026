# CPU port; original experiments/weight_space/clock_channels.py, SHA256 427c3f53493a56f18d580c1a9469a12ff343219204639b2f11d66c4829c65217
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Use a physically constrained trend clock and one-sided extrapolations."""
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

def tags(mode):
    return ['clock'] + (['clock_seq', 'curve_seq'] if mode == 'net' else [])

def design(mode, t, z, ix):
    c = t['clock'].iloc[ix]
    gate = c.gate.to_numpy()
    likelihood = c[['restart', 'restart_smooth', 'freeze', 'off', 'reverse', 'reverse_smooth', 'on', 'start', 'change']].to_numpy().clip(-16, 40)
    rows = [np.ones(len(z)), np.tanh(z / 2), np.arcsinh(likelihood), likelihood / 7, np.clip(c.change - z, -8, 10), c.before, c.after, np.abs(c.before - c.after), np.arcsinh(c.trend.clip(-8, 40))]
    if mode == 'net':
        orig, pre, post = (q.mean(1) for q in np.split(t['clock_seq'].iloc[ix].to_numpy(), 3, axis=1))
        mean = t['curve_seq'].iloc[ix].to_numpy()[:, -5:].mean(1)
        d = np.column_stack([pre - z, post - z, pre - orig, post - orig, pre - mean, post - mean]).clip(-9, 9)
        rows.extend([d, d * np.tanh(z[:, None] / 2), d[:, :3] * c.before.to_numpy()[:, None], d[:, 3:] * c.after.to_numpy()[:, None]])
    return gate[:, None] * np.column_stack(rows)
