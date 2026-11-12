# CPU port; original experiments/weight_space/verify_target.py, SHA256 02091c5efdbf21f02b2208e6a350ddc3ef33d291a2942b4bec8e47bde31d19eb
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Evaluate the locked phase/AR recipe on additional, fixed meta splits."""
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
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
import expert_fusion as ef
import innovation_head as ih
import mechanism_edit as me
import seasonal_head as sh
DEST = ef.PACKAGE / 'mechanism_phase_ar_candidate'

def run():
    data = ef.read_data()
    item, ids, tids, y, wt, X, V, VT, labs, groups, E, D, nc = data
    locked = json.loads((DEST / 'weights.json').read_text())
    parent = json.loads(ih.PARENT.read_text())
    mode, penalty = (locked['mode'], locked['selected'])
    assert mode == 'phase_ar' and penalty == 40.0
    tables = me.tables(data, me.inputs(mode))
    _, features, gate, H = sh.read_tables(data, True)
    A = features[parent['cols']].to_numpy()
    old = np.load(ef.OUT / 'mechanism_phase_ar_oof.npz')
    assert np.array_equal(ids, old['ids'])
    extra = {}
    for seed in (0, 123, 2024, 61543):
        pred, orig = (np.empty(len(y)), np.empty(len(y)))
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            theta = ef.fit_views(V[tr], y[tr], groups, ef.logistic_fit(X[tr], y[tr]), 100.0)
            zr, zv = (V[i] @ theta[:-1] + theta[-1] for i in (tr, va))
            q = ef.distribution_maps(D[tr])
            r, v = (ef.design('regime', E[i], ef.ranks(D[i], q), z, nc) for i, z in ((tr, zr), (va, zv)))
            w = ef.fit_head(r, zr, y[tr])
            zr, zv = (zr + r @ w[:-1] + w[-1], zv + v @ w[:-1] + w[-1])
            orig[va] = zv
            hr, hv = sh.harmonic_parent(H, tr, va, zr, zv, y)
            zr, zv = (sh.base(parent['mode'], zr, hr, gate[tr]), sh.base(parent['mode'], zv, hv, gate[va]))
            cq = sh.lookup(A, tr, gate)
            r, v = (sh.features(parent['mode'], A[i], gate[i], z, cq) for i, z in ((tr, zr), (va, zv)))
            w = sh.fit(r, zr, y[tr])
            zr, zv = (zr + r @ w, zv + v @ w)
            r, v = (me.encode(mode, tables, z, gate, i) for i, z in ((tr, zr), (va, zv)))
            w = sh.fit(r, zr, y[tr], penalty=penalty)
            pred[va] = zv + v @ w
        if seed == 0:
            assert np.max(abs(pred - old['s0_40.0'])) < 1e-07
        extra[seed] = {'fusion': ef.score(y, orig, wt), 'candidate': ef.score(y, pred, wt)}
        print(seed, extra[seed], flush=True)
    official = pd.read_csv(ef.ROOT / 'data/sample_submission.csv', dtype={'id': str})
    sub = pd.read_csv(DEST / 'submission.csv', dtype={'id': str})
    assert list(official.columns) == list(sub.columns) and np.array_equal(official.id, sub.id)
    assert sub.id.is_unique and len(sub) == 4000 and np.isfinite(sub.break_probability).all() and sub.break_probability.between(0, 1).all()
    assert ef.sha(DEST / 'submission.csv') == ef.sha(DEST / 'submission_replay.csv')
    p = pd.read_csv(ih.PARENT.parent / 'submission.csv', dtype={'id': str}).set_index('id').loc[tids, 'break_probability']
    base_logits = logit(p.to_numpy())
    design = me.encode(mode, tables, base_logits, gate, np.arange(len(y), len(gate)))
    order = pd.Index(tids).get_indexer(official.id)
    np.savez_compressed(DEST / 'replay_inputs.npz', ids=tids.to_numpy(dtype=str), order=order, parent_logits=base_logits, features=design)
    report = {'validation': 'Five-fold meta validation on 24,000 existing out-of-fold base predictions', 'initial_splits': {k: s[str(penalty)] for k, s in locked['scores'].items()}, 'additional_splits': extra, 'recipe_sha256': ef.sha(DEST / 'weights.json'), 'submission_sha256': ef.sha(DEST / 'submission.csv'), 'rows': len(sub), 'paired_original_sha256': ef.sha(ef.BASE)}
    (DEST / 'validation.json').write_text(json.dumps(report, indent=2))
if __name__ == '__main__':
    with threadpool_limits(limits=2):
        run()
