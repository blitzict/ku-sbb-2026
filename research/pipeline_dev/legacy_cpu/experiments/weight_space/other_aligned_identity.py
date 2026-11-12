# CPU port; original experiments/weight_space/other_aligned_identity.py, SHA256 95ac323acbc370f1c1a31d8d3ca0f009876c604fe73159c5c491c45a95041c19
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Measure the trained source scores after held-back calibration of each model."""
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
import numpy as np
from threadpoolctl import threadpool_limits
from other_response_gate import run
from partition_evidence import OUT
from white_local_rank_weights import weights, centered

def factory(ids, tids, y):
    models = []
    files = []
    for s in (7, 29):
        f = OUT / f'process_identity_s{s}_aligned.npz'
        files.append(f)
        with np.load(f) as a:
            assert np.array_equal(a['ids'], ids) and np.array_equal(a['tids'], tids)
            models.append(np.vstack([a['oof'], a['test']]))
    raw = np.mean(models, axis=0)

    def fit(tr, va, z, v, pr, pv):

        def build(ix, base):
            dx = np.arcsinh(raw[ix, 0] - base)
            change = np.arcsinh(raw[ix, 1] - raw[ix, 0])
            risk = np.tanh((base + 0.3) / 1.5)
            return np.column_stack([dx, change, 0.5 * dx * risk, 0.5 * change * risk])
        h, ref = centered(build(tr, z), z)
        test, _ = centered(build(va, v), v, ref)
        w = 0.75 * weights(h, z, y[tr])
        return (v + test @ w, test, w)
    return (fit, [*files, Path(__file__).resolve()])
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run(factory, 'local_rank_calibrated_identity')
