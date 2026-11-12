# CPU port; original experiments/weight_space/white_freeze_opinion.py, SHA256 edccb20e58c0bfabe261273300dd7deafe7247793d3ee28c8f1bb1e56a38720b
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Rebuild the six partitions of the incumbent and crossfit its readout for new edits."""
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
import argparse
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from white_opinion_spectrum import NAME, inputs, design
from white_nested import fit
from white_local_rank_weights import weights
from white_checkpoint import SEEDS
from white_paired_branches import test_scores
from partition_evidence import ROOT, OUT
import white_loop as wl
TAG = 'local_rank_opinion_linear'

def step(packs, reg, tr, va, z, v, y, ranked=False):
    h, _ = design(packs, reg, tr, va, z, v)
    w = weights(h[:len(tr)], z, y[tr]) if ranked else fit(h[:len(tr)], z, y[tr], True)
    return v + 0.5 * h[len(tr):] @ w

def main(tag=TAG):
    q, ids, tids, y, *_ = wl.load()
    packs, reg, _ = inputs(ids, tids)
    ranked = tag != TAG
    if tag == 'local_rank_without_multiview_trees':
        packs[2] = packs[2][:, :0]
    origin = OUT / ('frozen_' + NAME)
    dst = OUT / ('frozen_' + tag)
    dst.mkdir(exist_ok=True)
    sha = wl.ef.sha(ROOT / f'new weights/wn_{tag}_candidate/weights.json')
    with np.load(OUT / (tag + '_oof.npz')) as archive:
        assert np.array_equal(ids, archive['ids'])
        for s in SEEDS:
            for f in range(5):
                key = f's{s}_f{f}_'
                tr, va = (q[key + 'tr'], q[key + 'va'])
                file = dst / f's{s}_f{f}.npz'
                if file.exists():
                    with np.load(file) as a:
                        assert str(a['parent_sha']) == sha and np.array_equal(a['validation_ids'], ids[va])
                    continue
                with np.load(origin / file.name) as a:
                    assert np.array_equal(ids[tr], a['train_ids']) and np.array_equal(ids[va], a['validation_ids'])
                    z, v = (a['oof_train'], a['valid'])
                v = step(packs, reg, tr, va, z, v, y, ranked)
                assert max(abs(v - archive[f's{s}'][va])) < 1e-06
                inner = np.empty(len(tr))
                for a, b in StratifiedKFold(3, shuffle=True, random_state=831 + f).split(tr, y[tr]):
                    inner[b] = step(packs, reg, tr[a], tr[b], z[a], z[b], y, ranked)
                np.savez_compressed(file, train_ids=ids[tr].to_numpy(dtype=str), validation_ids=ids[va].to_numpy(dtype=str), oof_train=inner, valid=v, parent_sha=sha)
            print('reconstructed', s, flush=True)
        np.savez_compressed(dst / 'full.npz', train_ids=ids.to_numpy(dtype=str), test_ids=tids.to_numpy(dtype=str), oof_train=archive['s7'], test=test_scores(tag, tids), parent_sha=sha)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('tag', nargs='?', default=TAG, choices=(TAG, 'local_rank_family_near_pairs', 'local_rank_without_multiview_trees'))
    with threadpool_limits(limits=2):
        main(p.parse_args().tag)
