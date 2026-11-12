# CPU port; original experiments/weight_space/other_rank_policy.py, SHA256 0d490c378fa17ecaca2daeea07c3605df6ef43942e36a824265987dd59dc03ff
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Choose a specialist by its net positive-negative rank wins inside real source folds."""
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
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from other_archived_votes import load
from other_miss_witnesses import make
from other_response_gate import run
from white_family_calibration import groups
from partition_evidence import OUT

def factory(ids, tids, y):
    banks, paths = make(ids, tids)
    older, files = load(ids, tids)
    paths.extend(files)
    family, shape, files = groups(ids, tids)
    paths.extend(files)
    models = [b.mean(1) for b in banks]
    models.extend([np.median(older, axis=1), older.max(1)])
    calibrated = []
    for s in (7, 29):
        p = OUT / f'process_identity_s{s}_aligned.npz'
        paths.append(p)
        with np.load(p) as a:
            assert np.array_equal(ids, a['ids']) and np.array_equal(tids, a['tids'])
            calibrated.append(np.vstack([a['oof'], a['test']]))
    models.extend(np.mean(calibrated, axis=0).T)
    E = np.column_stack(models)

    def fit(tr, va, z, v, prev, pv):
        ix = np.r_[tr, va]
        base = np.r_[z, v]
        n = len(tr)
        yt = y[tr]
        out = np.zeros((len(ix), 3))
        for fold, (learn, check) in enumerate(StratifiedKFold(3, shuffle=True, random_state=50873).split(tr, yt)):
            p = np.array([0.15, 0.5, 0.85])
            a, b, c = np.quantile(E[tr[learn]], p, axis=0)
            l, m, r = np.quantile(z[learn], p)
            aligned = (E[ix] - b) * (r - l) / (c - a).clip(0.1) + m
            edits = 0.34 * np.tanh((aligned - base[:, None]) / 1.2)
            edits = np.column_stack([edits, 0.32 * np.tanh((base - np.r_[prev, pv]) / 0.22)])
            flags = base < np.quantile(z[learn][yt[learn] == 0], 0.8)
            change = np.empty((len(learn), edits.shape[1]))
            for label in (0, 1):
                ref = np.sort(z[learn][yt[learn] != label])
                sel = yt[learn] == label
                zz = z[learn[sel]]
                counts = np.searchsorted(ref, zz[:, None] + edits[learn[sel]]) - np.searchsorted(ref, zz)[:, None]
                change[sel] = counts * (2 * label - 1) / len(ref)
            cellsets = [family[ix], shape[ix]]
            proposals = []
            for cell in cellsets:
                edit = np.zeros(len(base))
                for code in np.unique(cell):
                    eligible = flags & (cell == code)
                    sample = change[eligible[learn]]
                    if len(sample) < 175:
                        continue
                    avg = sample.mean(0)
                    lower = avg - 1.2 * sample.std(0) / np.sqrt(len(sample))
                    k = int(lower.argmax())
                    if lower[k] > 0:
                        edit[eligible] = edits[eligible, k] * np.clip(lower[k] / max(avg[k], 1e-07), 0, 1)
                proposals.append(edit)
            edit = np.mean(proposals, axis=0)
            gain = roc_auc_score(yt[check], base[check] + edit[check]) - roc_auc_score(yt[check], base[check])
            if gain > 0:
                out[:, fold] = edit * min(1, gain / 0.00018)
        w = np.full(3, 1 / 3.0)
        return (v + out[n:] @ w, out[n:], w)
    return (fit, [*paths, Path(__file__).resolve()])
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run(factory, 'local_rank_net_wins_policy')
