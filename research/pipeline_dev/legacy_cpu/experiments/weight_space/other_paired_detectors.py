# CPU port; original experiments/weight_space/other_paired_detectors.py, SHA256 008ad53d4060303cf2057e75a9112b1e8ea33bf1f4e27f34a6246856e2dc83b5
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Refit the output weights on explicit changes between paired source models."""
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
from white_joint import read_member
from white_local_rank_weights import weights
from white_opinion_spectrum import inputs
from white_shared_residual import MEMBERS, chart
PAIRS = (('lawgrowth_c1_c2_c3', 'lawgrowth_c1_c2_c3_control'), ('law_symmetric_pairs', 'law_symmetric_binary'), ('law_linked_rank_s29', 'law_independent_rank_s29'), ('lawrank_crops', 'lawrank_original'), ('lawtree_both', 'lawtree_pulse'))

def paired(ids, tids, y):
    bank, reg, files = inputs(ids, tids)

    def member(name):
        v, p = read_member(name, ids, tids)
        files.extend(p)
        return v[:, 0]
    values = [member(m) for m in MEMBERS]
    for left, right in PAIRS:
        values.extend((member(left), member(right)))
    src = []
    for s in (7, 29):
        path = OUT / f'measured_source_s{s}.npz'
        files.append(path)
        with np.load(path) as a:
            assert np.array_equal(a['ids'], ids) and np.array_equal(a['tids'], tids)
            src.append(np.vstack([a['oof'], a['test']]))
    control, change = np.mean(src, axis=0).T
    values.extend((change, control))
    diff = np.column_stack(values)

    def fit(tr, va, z, v, previous, prior):
        h = chart(bank, reg, diff, tr, va, previous, prior)['response']
        w = 0.5 * weights(h[:len(tr)], previous, y[tr])
        matrix = np.column_stack([h[len(tr):], prior - v])
        w = np.r_[w, 1.0]
        return (v + matrix @ w, matrix, w)
    return (fit, [*files, Path(__file__).resolve()])
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run(paired, 'local_rank_explicit_detector_changes')
