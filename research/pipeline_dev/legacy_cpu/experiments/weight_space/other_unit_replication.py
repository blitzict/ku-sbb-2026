# CPU port; original experiments/weight_space/other_unit_replication.py, SHA256 6582aafa606f27ce5bca71101ec2b18a1e2afe8d14d3d06cd52f15f3564b12eb
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Check the fixed unit-start direction in broader misses outside the discovery pool."""
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
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
from partition_evidence import OUT
from other_disagreement_signals import matching

def main():
    ids, a, b, meta = matching(any_vote=False)
    studied = pd.read_csv(OUT / 'other_disagreement_pairs.csv')
    used = set(studied.case_id) | set(studied.control_id)
    assert used.isdisjoint(ids[a]) and used.isdisjoint(ids[b])
    u = pd.read_parquet(OUT / 'memory_law1_train.parquet').loc[ids, 'unit_start'].to_numpy()
    rank = rankdata(np.r_[u[a], u[b]]) / (2 * len(a))
    delta = rank[:len(a)] - rank[len(a):]
    rng = np.random.default_rng(177232)
    boot = np.concatenate([rng.choice(delta, (128, len(a))).mean(1) for _ in range(96)])
    flips = np.concatenate([(rng.integers(2, size=(128, len(a)), dtype=np.int8) * 2 - 1) @ delta / len(a) for _ in range(128)])
    report = dict(**meta, win=float(np.mean((delta > 0) + 0.5 * (delta == 0))), rank_difference=float(delta.mean()), paired_interval=np.quantile(boot, [0.025, 0.975]).tolist(), one_sided_p=float((1 + sum(flips >= delta.mean())) / (1 + len(flips))), scope='All six detector families below their own 20% false-positive cutoff. No source in common with the original disagreement pairs.')
    (OUT / 'other_unit_disjoint_replication.json').write_text(json.dumps(report, indent=2))
    print(report, flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
