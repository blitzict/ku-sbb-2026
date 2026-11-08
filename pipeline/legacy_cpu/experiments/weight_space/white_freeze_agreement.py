# CPU port; original experiments/weight_space/white_freeze_agreement.py, SHA256 63db42399784faf061bc3e13c3d07ecb7205ced1269cf6f18fd893c7ea92358e
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Cross-fit the accepted expert agreement readout within each meta-training fold."""
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
import json
import numpy as np
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from white_checkpoint import SEEDS
from white_expert_agreement import constructor
from white_local_rank_weights import centered, weights
from partition_evidence import OUT, ROOT
import white_loop as wl

def main(name):
    pack = ROOT / f'new weights/wn_{name}_candidate'
    spec = json.loads((pack / 'weights.json').read_text())
    digest = wl.ef.sha(pack / 'weights.json')
    q, ids, tids, y, *_ = wl.load()
    if name == 'local_rank_family_agreement':
        builder = constructor([])
    else:
        from white_baseline_opinions import inputs as builder
    make, _ = builder(ids, tids, spec['models'], spec['physics'])
    origin = OUT / ('frozen_stage_' + spec['parent'])
    dest = OUT / ('frozen_' + name)
    dest.mkdir(exist_ok=True)
    with np.load(OUT / (name + '_oof.npz')) as a:
        assert np.array_equal(a['ids'], ids)
        original = {s: a[f's{s}'] for s in SEEDS}
    for s in SEEDS:
        for f in range(5):
            k = f's{s}_f{f}_'
            tr, va = (q[k + key] for key in ('tr', 'va'))
            dst = dest / f's{s}_f{f}.npz'
            if dst.exists():
                with np.load(dst) as a:
                    assert str(a['parent_sha']) == digest
                continue
            with np.load(origin / f's{s}_f{f}.npz') as a:
                assert np.array_equal(a['train_ids'], ids[tr]) and np.array_equal(a['validation_ids'], ids[va])
                z = a['oof_train']
                v = a['valid']
            raw = make(tr, z)
            xx, ref = centered(raw, z)
            vv, _ = centered(make(va, v), v, ref)
            w = weights(xx, z, y[tr])
            pv = v + vv @ w
            assert np.max(abs(original[s][va] - pv)) < 1e-06
            inner = np.empty(len(tr))
            for a, b in StratifiedKFold(3, shuffle=True, random_state=1103 + f).split(tr, y[tr]):
                x, mean = centered(raw[a], z[a])
                h, _ = centered(raw[b], z[b], mean)
                inner[b] = z[b] + h @ weights(x, z[a], y[tr[a]])
            np.savez_compressed(dst, train_ids=ids[tr].to_numpy(dtype=str), validation_ids=ids[va].to_numpy(dtype=str), oof_train=inner, valid=pv, parent_sha=digest)
        print('nested expert agreement', s, flush=True)
    with np.load(pack / 'replay_inputs.npz') as a:
        p = a['parent_logits'] + a['features'] @ np.array(spec['weights'])
        assert np.array_equal(a['ids'], tids)
    import pandas as pd
    ref = pd.read_csv(pack / 'submission.csv', dtype={'id': str}).set_index('id').loc[tids, 'break_probability']
    assert max(abs(expit(p) - ref)) < 1e-12
    np.savez_compressed(dest / 'full.npz', train_ids=ids.to_numpy(dtype=str), test_ids=tids.to_numpy(dtype=str), oof_train=original[7], test=p, parent_sha=digest)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('name', nargs='?', choices=('local_rank_family_agreement', 'local_rank_early_opinions'), default='local_rank_family_agreement')
    with threadpool_limits(limits=1):
        main(p.parse_args().name)
