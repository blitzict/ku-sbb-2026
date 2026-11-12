# CPU port; original experiments/weight_space/white_holdout_pair_head.py, SHA256 c0d1c64713837e704dd8083707a3f34dcf376001f3f495ea8e374c20efedc0f6
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Refit the pair-trained heads on the independently reconstructed output partition."""
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
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from white_checkpoint import SEEDS
from white_family_ablation import FAMILY
from white_local_rank_weights import weights
from white_opinion_spectrum import design, inputs, NAME
from white_score_scale import REFERENCE
from white_nested import audit
from partition_evidence import ROOT, OUT
import white_loop as wl

def main(seed):
    assert seed not in SEEDS
    q, ids, tids, y, wt, wn, miss = wl.load()
    packs, gates, _ = inputs(ids, tids)
    cache = OUT / ('frozen_' + NAME)
    candidates = [(REFERENCE, None)]
    for j, group in enumerate(FAMILY):
        name = 'local_rank_without_' + group
        if (ROOT / f'new weights/wn_{name}_candidate/weights.json').exists():
            candidates.append((name, j))
    with np.load(OUT / (REFERENCE + '_oof.npz')) as a:
        assert np.array_equal(a['ids'], ids)
    results = {k: {} for k, _ in candidates}
    errors = {}
    for s in (7, seed):
        pred = {k: np.empty(len(ids)) for k, _ in candidates}
        locked = np.empty(len(y))
        for f, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=s).split(ids, y)):
            with np.load(cache / f's{s}_f{f}.npz') as a:
                assert np.array_equal(a['train_ids'], ids[tr]) and np.array_equal(a['validation_ids'], ids[va])
                z, v = (a['oof_train'], a['valid'])
                locked[va] = q[f's7_f{f}_zv'] if s == 7 else a['locked']
            for k, j in candidates:
                bank = list(packs)
                if j is not None:
                    bank[j] = bank[j][:, :0]
                h, _ = design(bank, gates, tr, va, z, v)
                w = 0.5 * weights(h[:len(tr)], z, y[tr])
                pred[k][va] = v + h[len(tr):] @ w
        for name, p in pred.items():
            if s == 7:
                with np.load(OUT / (name + '_oof.npz')) as a:
                    assert np.array_equal(a['ids'], ids)
                    errors[name] = float(max(abs(p - a['s7'])))
                    assert errors[name] < 2e-06
            results[name][s] = dict(locked=wl.score(y, locked, wt, wn, miss), candidate=wl.score(y, p, wt, wn, miss), recovery=audit(y, miss, locked, p))
            print(name, s, results[name][s]['candidate'], flush=True)
    for name, score in results.items():
        folder = ROOT / f'new weights/wn_{name}_candidate'
        (folder / f'additional_partition_{seed}.json').write_text(json.dumps(dict(recipe_sha256=wl.ef.sha(folder / 'weights.json'), primary_max_logit_error=errors[name], scores=score, scope='Same series and base-model OOF inputs. Intermediate output corrections and source priors were refitted to this additional partition, and the original primary scores were reproduced first.'), indent=2))
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main(61817)
