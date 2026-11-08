# CPU port; original experiments/weight_space/white_freeze_first_layer.py, SHA256 f557cab62adf297adb8cb21a4d60032e8c8860882ee36e9e3a032d86ef439efd
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Reconstruct out-of-training parent scores for follow-up edits to the overall weights."""
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
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from white_joint import read_member
from white_crop_replication import BASE, PREVIOUS
from white_paired_branches import test_scores
from white_checkpoint import SEEDS
from partition_evidence import ROOT, OUT
import white_loop as wl
NAME = 'local_rank_first_layer_overall'

def main():
    q, ids, tids, y, *_ = wl.load()
    folder = ROOT / f'new weights/wn_{NAME}_candidate'
    digest = wl.ef.sha(folder / 'weights.json')
    recipe = json.loads((ROOT / f'new weights/wn_{BASE}_candidate/weights.json').read_text())
    experts = np.column_stack([read_member(name, ids, tids)[0][:, 0] for name in recipe['members']])
    white = pd.read_parquet(OUT / 'innov_v2_train.parquet').loc[ids, 'pre_white'].to_numpy()
    dest = OUT / ('frozen_' + NAME)
    dest.mkdir(exist_ok=True)
    old = np.load(OUT / (BASE + '_oof.npz'))
    full = np.load(OUT / (NAME + '_oof.npz'))
    up = np.load(OUT / 'local_rank_upstream_joint_oof.npz')
    sha = wl.ef.sha(OUT / 'local_rank_upstream_joint_oof.npz')
    for a in (old, up, full):
        assert np.array_equal(a['ids'], ids)
    for s in SEEDS:
        for f in range(5):
            k = f's{s}_f{f}_'
            tr, va = (q[k + 'tr'], q[k + 'va'])
            with np.load(OUT / f'frozen_{PREVIOUS}/s{s}_f{f}.npz') as a:
                assert np.array_equal(a['train_ids'], ids[tr]) and np.array_equal(a['validation_ids'], ids[va])
                z, v = (a['oof_train'], a['valid'])
            with np.load(OUT / f'upstream_inner/s{s}_f{f}.npz') as a:
                assert str(a['branch_sha']) == sha and np.array_equal(a['ids'], ids[tr])
                u = a['branch']
            lo, med, hi = np.quantile(experts[tr], [0.25, 0.5, 0.75], axis=0)
            l, c, r = np.quantile(z, [0.25, 0.5, 0.75])

            def base(i, p):
                e = (experts[i] - med) * ((r - l) / (hi - lo)) + c
                g = 1 - 0.85 * white[i] * np.exp(-0.5 * ((p + 1.1) / 1.25) ** 2)
                return p + (e - p[:, None]) * g[:, None] @ recipe['weights']
            assert np.max(abs(base(va, v) - old[f's{s}'][va])) < 1e-08
            v = 0.75 * base(va, v) + 0.25 * up[f'branch_{s}'][va]
            assert np.max(abs(v - full[f's{s}'][va])) < 1e-08
            np.savez_compressed(dest / f's{s}_f{f}.npz', train_ids=ids[tr].to_numpy(dtype=str), validation_ids=ids[va].to_numpy(dtype=str), oof_train=0.75 * base(tr, z) + 0.25 * u, valid=v, parent_sha=digest)
    np.savez_compressed(dest / 'full.npz', train_ids=ids.to_numpy(dtype=str), test_ids=tids.to_numpy(dtype=str), oof_train=full['s7'], test=test_scores(NAME, tids), parent_sha=digest)
    print('Reconstructed all 30 validation folds and 4,000 predictions', flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
