# CPU port; original experiments/weight_space/white_import_tree.py, SHA256 6b6682fe78e0a5624fc19fd56d1033aca45d7cac5e406ea5fa273dfbd5b56737
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Align Claude's completed prediction files to the locked source identifiers."""
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
import numpy as np
from pathlib import Path

def align(tag, name):
    out = Path(__file__).resolve().parents[2] / 'tune_out'
    with np.load(out / 'weight_space/white_loop_parent.npz') as parent, np.load(out / 'var/mat_ns3_plain.npz', allow_pickle=True) as order:
        ids, tids = (parent['ids'], parent['tids'])
        assert np.array_equal(order['ids'], ids) and np.array_equal(order['tid'], tids)
    a, b = [np.load(out / f'{part}_{tag}.npy') for part in ('oof', 'test')]
    assert a.shape == ids.shape and b.shape == tids.shape
    assert all((np.all((x > 0) & (x < 1)) for x in (a, b)))
    dest = out / 'weight_space' / (name + '.npz')
    if dest.exists():
        with np.load(dest) as old:
            assert all((np.array_equal(old[k], v) for k, v in (('ids', ids), ('tids', tids), ('oof', a), ('test', b))))
    else:
        np.savez_compressed(dest, ids=ids, tids=tids, oof=a, test=b)
    print(tag, name, len(a), len(b), flush=True)
if __name__ == '__main__':
    align(*sys.argv[1:])
