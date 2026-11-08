# CPU port; original experiments/weight_space/head_channels.py, SHA256 8b4aa1b5dc52141ceefee78c80c0d2f07c52327608f984a696b10dc5bdff9a44
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Preserve the distinct output directions of the two stored neural architectures."""
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
    return ['periodic_seq', 'innov_v2_seq', 'curve_seq', 'pulse_seq', 'periodic', 'innov_v2', 'curve', 'pulse', 'state']

def design(mode, t, z, ix):
    s = t['periodic'].iloc[ix]
    wave = (s.gate * expit((s.period - 9) / 1.5) * expit((38 - s.period) / 2)).to_numpy()
    c = t['curve'].iloc[ix].gate.to_numpy()
    p = t['pulse'].iloc[ix].gate.to_numpy()
    h = t['state'].iloc[ix]
    memory = np.maximum(h.walk, np.maximum(h.order2, np.minimum(1, abs(h.ar1) * 3))).to_numpy()

    def architecture(mat):
        return mat[:, :3].mean(1) - mat[:, 3:].mean(1)
    fields = []
    for name, gate in [('periodic', wave), ('innov_v2', (1 - wave) * memory), ('curve', c), ('pulse', p)]:
        orig, shared, separate = np.split(t[name + '_seq'].iloc[ix].to_numpy(), 3, axis=1)
        if not fields:
            orig_diff = architecture(orig).clip(-5, 5)
            fields.extend([orig_diff, orig_diff * np.tanh(z / 2)])
        delta = np.column_stack([architecture(shared - orig), architecture(separate - orig), architecture(shared), architecture(separate)]).clip(-5, 5)
        fields.extend([gate[:, None] * delta, gate[:, None] * delta * np.tanh(z[:, None] / 2)])
    if mode == 'family':
        for gate in (wave, c, p, h.pre2, h.pre_cluster, h.walk):
            g = np.asarray(gate)
            fields.extend([g, g * np.tanh(z / 2), g * np.exp(-0.5 * ((z - 0.7) / 1.1) ** 2)])
    return np.column_stack(fields)
