# CPU port; original experiments/weight_space/other_transition_catalog.py, SHA256 a2b2017a5930584d2faa823456d44c81aac22966599d6251139bfffaa2b987db
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Integrate observed likelihoods for interpretable transitions between signed memory states."""
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
import json
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from threadpoolctl import threadpool_limits
from white_memory_law import bank, row, D, S, P, IDS
from white_jump_priors import tensor
from partition_evidence import OUT
PHI = np.r_[np.linspace(-0.96, 0.96, 33), 1.0]

def dictionary():
    spacing = np.gradient(PHI)
    atom = lambda k: (np.arange(len(PHI)) == k).astype(float)
    start = {'zero': atom(16), 'unit': atom(len(PHI) - 1)}
    for name, center, width in (('negative', -0.53, 0.27), ('near_zero', 0.0, 0.17), ('positive', 0.5, 0.25), ('persistent', 0.88, 0.095)):
        a = spacing * np.exp(-0.5 * ((PHI - center) / width) ** 2)
        start[name] = a / a.sum()
    delta = PHI[None, :] - PHI[:, None]
    more = lambda shift, sd: spacing[None, :] * np.exp(-0.5 * ((delta - shift) / sd) ** 2)
    transitions = {'same': np.eye(len(PHI)), 'local': more(0.0, 0.13), 'increase': more(0.3, 0.17), 'decrease': more(-0.3, 0.17), 'lose_memory': np.tile(atom(16), (len(PHI), 1)), 'to_unit': np.tile(atom(len(PHI) - 1), (len(PHI), 1)), 'new_slope': np.tile(spacing, (len(PHI), 1))}
    labels = []
    blocks = []
    for a, p in start.items():
        for b, kernel in transitions.items():
            if (a, b) in (('zero', 'lose_memory'), ('unit', 'to_unit')):
                continue
            w = p[:, None] * kernel / kernel.sum(1, keepdims=True)
            labels.append(a + '_' + b)
            blocks.append(w.ravel())
    h = np.stack(blocks)
    np.testing.assert_allclose(h.sum(1), 1.0, atol=2e-12)
    return (labels, h)

def evidence(vals, ref, scale, kernels):
    B = len(ref)
    delta = PHI[None, :] - ref[:, None]
    i = (np.searchsorted(D, delta) - 1).clip(0, len(D) - 2)
    frac = tensor(((delta - D[i]) / (D[i + 1] - D[i])).clip(0, 1))[:, None, None, :, None]
    index = torch.as_tensor(i, device=vals.device)[:, None, None, :, None].expand(-1, 2, len(IDS), -1, len(S))
    ll = vals.gather(3, index) * (1 - frac) + vals.gather(3, index + 1) * frac
    left, right = ll.unbind(1)
    a = left.amax(2, keepdim=True)
    b = right.amax(2, keepdim=True)
    s = np.log(scale[:, None]) + S
    v = 0.9 * expit((0.71 - abs(s)) / 0.05) + 0.1 * np.exp(-abs(s))
    prior = tensor(np.log(P))[None, :, None] + tensor(np.log(v / v.sum(1, keepdims=True)))[:, None, :]
    logn = a[:, :, 0] + b[:, :, 0] + prior
    norm = logn.amax((1, 2), keepdim=True)
    la = (left - a).clamp_min(-480).double().exp()
    lb = (right - b).clamp_min(-480).double().exp()
    post = (logn - norm).clamp_min(-480).double().exp()
    joint = torch.einsum('bkis,bkjs,bks->bij', la, lb, post)
    outputs = (joint.reshape(B, -1) @ kernels.T).clamp_min(1e-260).log()
    outputs -= outputs.amax(1, keepdim=True)
    return outputs.clamp_min(-140).float().cpu().numpy()

def run(part, grid, names, kernels):
    with np.load(OUT / f'pulse_streams_{part}.npz') as a:
        ids, x, n, b = [a[k] for k in ('ids', 'separate', 'n', 'b')]
    phi = pd.read_parquet(OUT / f'state_{part}.parquet').loc[ids, ['ar1', 'ar2']].to_numpy()
    blocks = []
    for start in range(0, len(n), 60):
        h, scale, r = ([], [], [])
        for v, j, m, p in zip(x[start:start + 60], b[start:start + 60], n[start:start + 60], phi[start:start + 60]):
            mat, sd, ref = row(v[:m].astype(float), j, p, 1)
            h.append(mat)
            scale.append(sd)
            r.append(ref)
        vals = (tensor(np.asarray(h)).reshape(2 * len(h), -1) @ grid).reshape(len(h), 2, len(IDS), len(D), len(S))
        vals -= vals.amax((2, 3, 4), keepdim=True)
        blocks.append(evidence(vals, np.array(r), np.array(scale), kernels))
    df = pd.DataFrame(np.vstack(blocks), index=ids, columns=names)
    assert np.isfinite(df).all().all()
    df.to_parquet(OUT / f'transition_catalog_{part}.parquet')
    print('transition likelihood', part, df.shape, flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1), torch.inference_mode():
        torch.set_num_threads(1)
        names, k = dictionary()
        kernel = torch.as_tensor(k, device='cpu', dtype=torch.float64)
        grid = bank()
        for part in ('train', 'test'):
            run(part, grid, names, kernel)
