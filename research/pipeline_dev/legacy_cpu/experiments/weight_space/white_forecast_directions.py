# CPU port; original experiments/weight_space/white_forecast_directions.py, SHA256 b70e876f4919c80b1b27e3a60b247d27e85327a5672a6a3f1bf491f07fa9fb52
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Identify recurrent directions of forecast-weight change from independent labelled sources."""
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
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from partition_evidence import OUT
from white_forecast_weight_change import run

def subspace(x, positive, negative):

    def cross(ix):
        a, b = (x[ix, 0], x[ix, 1])
        radius = np.sqrt((a * a + b * b).mean(1) / 2)
        scale = np.maximum(1.0, radius / 1.8)[:, None]
        return (a / scale).T @ (b / scale) / len(ix)
    cross_delta = cross(positive) - cross(negative)
    val, vec = np.linalg.eigh((cross_delta + cross_delta.T) * 0.5)
    vec = vec[:, ::-1]
    h = x @ vec[:, :8]
    return {f'top{k}_repeat': (h[:, 0, :k] * h[:, 1, :k]).mean(1) for k in (1, 3, 8)}

def directions(scores, counts, core, labels, ids):
    n = len(scores)
    sizes = np.sqrt(counts[:, :2] * counts[:, 2:] / (counts[:, :2] + counts[:, 2:]))
    d = (scores[:, 2:] - scores[:, :2]) * sizes[:, :, None, None, None]
    gate = pd.read_parquet(OUT / 'innov_v2_train.parquet').loc[ids, 'pre_white'].to_numpy()
    regions = {'all': core, 'white': core[gate[core] > 0.65]}
    out = {}
    for population, rows in regions.items():
        pos, neg = (rows[labels[rows] == 1], rows[labels[rows] == 0])
        assert min(len(pos), len(neg)) > 100
        blocks = []
        for j, label in enumerate(('centers', 'widths', 'mix')):
            h = d[:, :, :, 5 * j:5 * j + 5].reshape(n, 2, -1)
            reference = h[neg].reshape(-1, h.shape[-1])
            eigen, basis = np.linalg.eigh(np.cov(reference, rowvar=False))
            keep = np.arange(len(eigen) - 1, max(-1, len(eigen) - 33), -1)
            norm = basis[:, keep] / np.sqrt(np.maximum(eigen[keep], 0) + 0.05 * eigen.mean() + 1e-07)
            x = (h - reference.mean(0)) @ norm
            blocks.append(x)
            for k, v in subspace(x, pos, neg).items():
                out[f'{population}_{label}_{k}'] = v
        for k, v in subspace(np.concatenate(blocks, axis=2), pos, neg).items():
            out[f'{population}_combined_{k}'] = v
    return pd.DataFrame(out)
if __name__ == '__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        run(learned=True)
