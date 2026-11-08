# CPU port; original experiments/weight_space/state_channels.py, SHA256 d6c3ba9141a03cecbe29bbac1726d190d1698fda182e32e34e49c0113e104869
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Route variance-normalized and autoregressive innovation evidence."""
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
    return ['state', 'state_seq', 'pulse_seq', 'innov_v2_seq']

def design(mode, t, z, ix):
    h = t['state'].iloc[ix]
    seq = lambda tag: [q.mean(1) for q in np.split(t[tag].iloc[ix].to_numpy(), 3, axis=1)]
    _, shared, sep = seq('state_seq')
    _, _, clean = seq('pulse_seq')
    _, old_c, old_s = seq('innov_v2_seq')
    diff = np.column_stack([shared - z, sep - z, shared - old_c, sep - old_s, shared - clean, sep - clean]).clip(-8, 8)
    vol = np.maximum(h.clustering, 0.7 * np.maximum(h.pre_cluster, h.post_cluster))
    ar = np.maximum(h.order2, 0.65 * np.maximum(h.pre2, h.post2))
    tests = h[['ar_change', 'vol_bf', 'vol_split_bf', 'scale_z', 'change_scale', 'change_level']].to_numpy().clip(-15, 50)
    parts = [diff, np.tanh(z[:, None] / 2) * diff]
    for gate in (vol, ar, h.walk):
        parts.append(np.asarray(gate)[:, None] * np.column_stack([np.ones(len(z)), np.tanh(z / 2), diff, np.arcsinh(tests)]))
    return np.column_stack(parts)
