# CPU port; original experiments/weight_space/direction_channels.py, SHA256 a8f3d07f08e7f68336225320da4ae3ff3c7612a0e3d484fa1dc1c15db0eec3b9
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Keep the forward/reverse evidence and its uncertainty from saved neural heads."""
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
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
from partition_evidence import ROOT, OUT
MODEL = ['W_SEQ_q', 'W_SEQ_q_s1', 'W_SEQ_q_s2', 'W_SEQ_qh', 'W_SEQ_qh_s1']

def prepare():
    ids, tids = (pd.read_parquet(ROOT / f'clean/features_{p}.parquet', columns=[]).index for p in ('train', 'test'))
    y = pd.read_csv(ROOT / 'data/train_labels.csv', dtype={'id': str}).set_index('id').loc[ids, 'label']
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(ids, y))
    alltr = []
    allte = []
    for model in MODEL:
        tr = np.full((len(ids), 4), np.nan)
        te = np.zeros((len(tids), 4))
        for f, (_, va) in enumerate(folds):
            with np.load(ROOT / f'tune_out/seqpart/{model}_f{f}.npz') as q:
                assert np.array_equal(va, q['va'])
                tr[va] = q['pv'].T
                te += q['pt'].T.astype(float) / len(folds)
        alltr.append(tr)
        allte.append(te)
    cols = [f'{m}_v{i}' for m in MODEL for i in range(4)]
    for p, index, ds in (('train', ids, alltr), ('test', tids, allte)):
        x = np.column_stack(ds)
        assert np.isfinite(x).all()
        pd.DataFrame(x, index=index, columns=cols).to_parquet(OUT / f'directions_{p}.parquet')

def tags(mode):
    return ['directions', 'state', 'curve', 'pulse', 'periodic']

def design(mode, t, z, ix):
    a = t['directions'].iloc[ix].to_numpy().reshape(-1, 5, 4)
    dirs = []
    for mat in (a[:, :3].mean(1), a[:, 3:].mean(1)):
        before = (mat[:, 0] + mat[:, 1]) / 2
        after = (mat[:, 2] + mat[:, 3]) / 2
        gap = (before - after).clip(-7, 7)
        sign = (abs(mat[:, 0] - mat[:, 1]) + abs(mat[:, 2] - mat[:, 3])).clip(0, 10) / 2
        d = (mat.mean(1) - z).clip(-8, 8)
        dirs.extend([gap, abs(gap), gap * np.tanh(z / 2), sign, sign * np.tanh(z / 2), d])
    base = np.column_stack(dirs)
    s = t['periodic'].iloc[ix]
    wave = s.gate * expit((s.period - 9) / 1.5) * expit((38 - s.period) / 2)
    h = t['state'].iloc[ix]
    vals = [base]
    for g in (wave, t['curve'].iloc[ix].gate, t['pulse'].iloc[ix].gate, h.walk, h.pre_cluster):
        vals.append(np.array(g)[:, None] * base[:, [0, 1, 5, 6, 7, 11]])
    return np.column_stack(vals)
if __name__ == '__main__':
    prepare()
