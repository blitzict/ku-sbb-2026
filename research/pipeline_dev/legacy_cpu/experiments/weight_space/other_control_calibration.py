# CPU port; original experiments/weight_space/other_control_calibration.py, SHA256 d9770de16a82da784a251ed335bc9a08d6fb36e5c0f4e7d9a480920a6b93def3
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Update serially dependent series while anchoring the no-break score distribution."""
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
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from other_response_gate import run
from partition_evidence import ROOT
from other_transition_prior import factory as transition
from other_reversible_pairs import factory as contested

def factory(ids, tids, y, kind):
    get = transition if kind == 'transition' else contested
    correction, paths = get(ids, tids, y)
    path = ROOT / 'tune_out/generator_segments.parquet'
    d = pd.read_parquet(path, columns=['id', 'part', 'p_bic', 'phi1_ar1'])
    d = d[d.part == 'pre'].set_index('id').loc[ids.append(tids)]
    serial = (d.p_bic.to_numpy() != 0) | (d.phi1_ar1.to_numpy() > 0.95)

    def fit(tr, va, z, v, prev, pv):
        tentative = correction(tr, va, z, v, prev, pv)[0]
        cross = np.empty(len(tr))
        for a, b in StratifiedKFold(3, shuffle=True, random_state=3023).split(tr, y[tr]):
            cross[b] = correction(tr[a], tr[b], z[a], z[b], prev[a], prev[b])[0]
        controls = serial[tr] & (y[tr] == 0)
        grid = np.r_[0.002, 0.006, 0.012, np.linspace(0.025, 0.975, 51), 0.988, 0.994, 0.998]
        old = np.quantile(z[controls], grid)
        new = np.quantile(cross[controls], grid)
        offset = gaussian_filter1d(old - new, 1.25)
        adj = tentative - v + np.interp(tentative, new, offset)
        adj = np.where(serial[va], adj, 0.0)
        return (v + adj, adj[:, None], np.ones(1))
    return (fit, [*paths, path, Path(__file__).resolve()])
if __name__ == '__main__':
    kind = sys.argv[1]
    assert kind in ('transition', 'contested')
    with threadpool_limits(limits=1):
        run(lambda ids, tids, y: factory(ids, tids, y, kind), 'local_rank_serial_calibrated_' + kind)
