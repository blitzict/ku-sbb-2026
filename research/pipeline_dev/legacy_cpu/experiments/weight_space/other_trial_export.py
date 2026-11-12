# CPU port; original experiments/weight_space/other_trial_export.py, SHA256 05bf906a9a944a8f988f0253248cd1be8ebebf1f808bed883fb17431bd028adb
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Save and verify the serial-only trial separately from the confirmed weights."""
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
import shutil
from pathlib import Path
import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits
from other_quiet_reordering import setup
from other_missed_breaks import CURRENT
from white_opinion_spectrum import NAME
from partition_evidence import OUT, ROOT
import white_loop as wl
TAG = 'local_rank_quiet_reordering_serial'

def main():
    q, ids, tids, y, _, _, _ = wl.load()
    parent = ROOT / f'new weights/wn_{CURRENT}_candidate'
    trial = ROOT / 'new weights/quiet_serial_trial'
    trial.mkdir(exist_ok=True)
    fit, files = setup(ids, tids, y, True, False)
    inputs = OUT / ('frozen_' + CURRENT) / 'full.npz'
    before = OUT / ('frozen_' + NAME) / 'full.npz'
    with np.load(inputs) as a, np.load(before) as b:
        assert np.array_equal(ids, a['train_ids']) and np.array_equal(a['test_ids'], tids)
        assert np.array_equal(ids, b['train_ids']) and np.array_equal(b['test_ids'], tids)
        assert str(a['parent_sha']) == wl.ef.sha(parent / 'weights.json')
        v = a['test']
        new, h, w = fit(np.arange(len(y)), len(y) + np.arange(len(tids)), a['oof_train'], v, b['oof_train'], b['test'])
    assert np.isfinite(new).all() and np.allclose(new, v + h @ w, rtol=0, atol=1e-14)
    stats = json.loads((OUT / (TAG + '.json')).read_text())
    assert not stats['accepted'] and sum((d['new']['auc'] > d['previous']['auc'] for d in stats['scores'].values())) == 6
    sub = wl.ef.write_submission(expit(new), tids, trial / 'submission.csv')
    rec = trial / 'weights.json'
    deps = [Path(__file__).resolve(), parent / 'weights.json', inputs, before, *files]
    rec.write_text(json.dumps(dict(weights=w.tolist(), learned=fit.parameters, hashes={str(p.relative_to(ROOT)): wl.ef.sha(p) for p in deps}), indent=2))
    np.savez_compressed(trial / 'replay_inputs.npz', ids=tids.to_numpy(dtype=str), order=tids.get_indexer(sub.id), parent_logits=v, features=h)
    (trial / 'validation.json').write_text(json.dumps(dict(**stats, recipe_sha256=wl.ef.sha(rec), submission_sha256=wl.ef.sha(trial / 'submission.csv'), status='trial, one partition below incumbent'), indent=2))
    shutil.copyfile(parent / 'replay.py', trial / 'replay.py')
    print('changed test scores', int(sum(abs(new - v) > 1e-10)), 'of', len(tids), flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
