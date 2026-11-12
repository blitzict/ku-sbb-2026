# CPU port; original experiments/weight_space/other_freeze_current.py, SHA256 ab9aba4f03fae8c0796c4e6156044bdffbf9dc2044831608909935a9e07c5ee3
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Reconstruct out-of-training scores for evaluating new dynamics measurements."""
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
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from other_missed_breaks import CURRENT
from partition_evidence import ROOT, OUT
from white_checkpoint import SEEDS
from white_joint import read_member
from white_local_rank_weights import weights
from white_opinion_spectrum import inputs, NAME
from white_paired_branches import test_scores
from white_shared_residual import MEMBERS, chart
import white_loop as wl

def main():
    q, ids, tids, y, *_ = wl.load()
    bank, reg, _ = inputs(ids, tids)
    extra = np.column_stack([read_member(m, ids, tids)[0][:, 0] for m in MEMBERS])
    folder = OUT / ('frozen_' + CURRENT)
    folder.mkdir(exist_ok=True)
    digest = wl.ef.sha(ROOT / f'new weights/wn_{CURRENT}_candidate/weights.json')

    def step(tr, va, z, v):
        h = chart(bank, reg, extra, tr, va, z, v)['response']
        return v + h[len(tr):] @ (0.5 * weights(h[:len(tr)], z, y[tr]))
    with np.load(OUT / (CURRENT + '_oof.npz')) as known:
        assert np.array_equal(known['ids'], ids)
        for s in [*SEEDS, 61817]:
            for f, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=s).split(ids, y)):
                file = folder / f's{s}_f{f}.npz'
                if file.exists():
                    with np.load(file) as a:
                        assert str(a['parent_sha']) == digest and np.array_equal(a['validation_ids'], ids[va])
                    continue
                with np.load(OUT / f'frozen_{NAME}/s{s}_f{f}.npz') as a:
                    assert np.array_equal(a['train_ids'], ids[tr]) and np.array_equal(a['validation_ids'], ids[va])
                    z, v = (a['oof_train'], a['valid'])
                    locked = q[f's{s}_f{f}_zv'] if s in SEEDS else a['locked']
                new = step(tr, va, z, v)
                np.testing.assert_allclose(new, known[f's{s}'][va], atol=2e-07)
                inner = np.empty(len(tr))
                for aa, bb in StratifiedKFold(3, shuffle=True, random_state=831 + f).split(tr, y[tr]):
                    inner[bb] = step(tr[aa], tr[bb], z[aa], z[bb])
                np.savez_compressed(file, train_ids=ids[tr].to_numpy(dtype=str), validation_ids=ids[va].to_numpy(dtype=str), oof_train=inner, valid=new, parent_sha=digest, locked=locked)
            print('frozen', s, flush=True)
        np.savez_compressed(folder / 'full.npz', train_ids=ids.to_numpy(dtype=str), test_ids=tids.to_numpy(dtype=str), oof_train=known['s7'], test=test_scores(CURRENT, tids), parent_sha=digest)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
