# CPU port; original experiments/weight_space/white_source_bag.py, SHA256 52470c7f943d0cf1d68f58e725e4a3f0a958fa9fece8c7baf2db011f13183382
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Average source-aligned out-of-fold logits from distinct base training partitions."""
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
import argparse
import numpy as np
from scipy.special import expit, logit
from partition_evidence import OUT

def combine(name, members):
    sums = {}
    for i, member in enumerate(members):
        with np.load(OUT / (member + '.npz')) as a:
            if i == 0:
                ids, tids = (a['ids'].copy(), a['tids'].copy())
            assert np.array_equal(ids, a['ids']) and np.array_equal(tids, a['tids'])
            for key in ('oof', 'test'):
                p = a[key]
                assert np.all((p > 0) & (p < 1))
                sums[key] = sums.get(key, 0.0) + logit(p)
    path = OUT / (name + '.npz')
    data = dict(ids=ids, tids=tids, members=np.asarray(members), **{key: expit(v / len(members)) for key, v in sums.items()})
    if path.exists():
        with np.load(path) as old:
            assert all((np.array_equal(old[k], v) for k, v in data.items()))
    else:
        np.savez_compressed(path, **data)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('members', nargs='+')
    args = parser.parse_args()
    combine(args.name, args.members)
