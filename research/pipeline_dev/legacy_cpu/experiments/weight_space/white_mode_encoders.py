# CPU port; original experiments/weight_space/white_mode_encoders.py, SHA256 d8c65e3681740e0ea9e3df79d766193ec84f0d301b8520a7ae4c0079e6188c41
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Compare the two archived sequence architectures on separated noise peaks."""
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
from partition_evidence import OUT

def run(part):
    modes = pd.read_parquet(OUT / f'mode_residual_{part}.parquet')
    new = pd.read_parquet(OUT / f'mode_residual_seq_{part}.parquet').loc[modes.index]
    old = pd.read_parquet(OUT / f'pulse_seq_{part}.parquet').loc[modes.index]
    prior = pd.read_parquet(OUT / f'reliable_signals_{part}.parquet').loc[modes.index].filter(regex='^mode_')
    gate = ((modes.gate - 0.5) / 0.45).clip(0, 1)
    basis = dict(prior.items())
    for view in ('common', 'separate'):

        def effect(runs):
            return np.column_stack([(new[view + '_' + s] - old['separate_' + s]).clip(-8, 8) for s in runs]).mean(1)
        quantiles = effect(['W_SEQ_q', 'W_SEQ_q_s1', 'W_SEQ_q_s2'])
        histograms = effect(['W_SEQ_qh', 'W_SEQ_qh_s1'])
        difference = quantiles - histograms
        basis[view + '_architecture'] = gate * difference
        basis[view + '_asymmetry'] = gate * (modes.post_gate - modes.pre_gate) * (quantiles + histograms) / 2
    table = pd.DataFrame(basis, index=modes.index)
    assert np.isfinite(table).all().all()
    table.to_parquet(OUT / f'mode_encoders_{part}.parquet')
if __name__ == '__main__':
    for part in ('train', 'test'):
        run(part)
