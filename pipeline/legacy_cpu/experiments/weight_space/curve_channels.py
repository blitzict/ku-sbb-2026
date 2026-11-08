# CPU port; original experiments/weight_space/curve_channels.py, SHA256 f2998adfa3e51a3921674702e3e1752b0aca3fdfe6892d99a0f3a545fb6ad386
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Fixed readout rows for intervention evidence after marginalizing smooth drift."""
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
from scipy.special import logsumexp

def tags(mode):
    return ['curve', 'ar_law', 'trend'] + (['curve_seq'] if 'net' in mode else [])

def design(mode, t, z, ix):
    c, a, d = (t[k].iloc[ix] for k in ('curve', 'ar_law', 'trend'))
    gate = np.maximum(c.gate, d.gate).to_numpy()
    factors = c[['obs_step', 'slope_step', 'global_linear', 'global_quad', 'local_quad', 'combo']].to_numpy().clip(-10, 30)
    raw = a.level_bf.to_numpy().clip(-12, 30)
    corrected = c.mixture.to_numpy().clip(-12, 30)
    other = logsumexp(np.column_stack([a.scale_bf, a.tail_bf, a.memory_bf, np.zeros(len(z))]).clip(-12, 30), axis=1)
    delta = np.logaddexp(corrected, other) - np.logaddexp(raw, other)
    vals = [np.ones(len(z)), np.tanh(z / 2), np.clip(z, -5, 10), factors / 7, np.log1p(np.maximum(0, factors)), np.clip(corrected - z, -6, 12), delta.clip(-12, 12), np.clip(corrected - raw, -12, 12), c.null_trend, c.jump_trend, c.curvature.abs().clip(0, 8), c.slope.abs().clip(0, 8), np.log1p(c.adjusted_scale.clip(0, 40)), np.log1p(c.adjusted_level.clip(0, 40))]
    if 'net' in mode:
        mat = t['curve_seq'].iloc[ix].to_numpy()
        orig, shared, sep = (p.mean(axis=1) for p in np.split(mat, 3, axis=1))
        diff = np.column_stack([shared - z, sep - z, shared - orig, sep - orig]).clip(-8, 8)
        vals.extend([diff, diff * np.tanh(z[:, None] / 2), np.maximum(diff - 1, 0)])
    return gate[:, None] * np.column_stack(vals)
