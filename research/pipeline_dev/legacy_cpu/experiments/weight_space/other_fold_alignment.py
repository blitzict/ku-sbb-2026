# CPU port; original experiments/weight_space/other_fold_alignment.py, SHA256 a999a99bddeccfa576ada89d9192550245fc570e4cb610312a6f147a5bbfe8e4
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Check cross-fold score scales and calibrate a new detector on its reserved inner rows."""
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
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits
from other_process_identity import blocks, Identify
from other_missed_breaks import CURRENT
from white_block_kernel import phys
from partition_evidence import ROOT, OUT
from white_checkpoint import SEEDS
import white_loop as wl

def aucs(y, z, parts):
    local = [roc_auc_score(y[v], z[v]) for t, v in parts]
    return dict(pooled=float(roc_auc_score(y, z)), within=float(np.mean(local)), fold_auc=local)

def calibration(x, y):

    def loss(p):
        z = x * p[1] + p[0]
        r = expit(z) - y
        return (np.sum(np.logaddexp(0, z) - y * z) + 25 * p[0] ** 2 + 50 * (p[1] - 1) ** 2, np.array([r.sum() + 50 * p[0], r @ x + 100 * (p[1] - 1)]))
    m = minimize(loss, [0.0, 1.0], jac=True, bounds=[(-1.5, 1.5), (0.55, 1.6)], method='L-BFGS-B')
    assert m.success
    return m.x

def main():
    _, ids, tids, y, wt, wn, miss = wl.load()
    results = {}
    with np.load(OUT / (CURRENT + '_oof.npz')) as a:
        assert np.array_equal(a['ids'], ids)
        for s in [*SEEDS, 61817]:
            parts = list(StratifiedKFold(5, shuffle=True, random_state=s).split(ids, y))
            results[str(s)] = aucs(y, a[f's{s}'], parts)
    print('incumbent primary', results['7'], flush=True)
    rid, x = blocks('train')
    assert np.array_equal(rid, ids)
    c = phys('train', ids)
    rows = []
    for s in (7, 29):
        name = f'process_identity_s{s}'
        folder = OUT / (name + '_models')
        with np.load(OUT / (name + '.npz')) as a:
            original = a['oof'][:, :2]
        parts = list(StratifiedKFold(5, shuffle=True, random_state=s).split(ids, y))
        aligned = np.empty_like(original)
        tests = []
        fits = []
        for fold, (tr, va) in enumerate(parts):
            core, inner = train_test_split(tr, stratify=y[tr], test_size=0.14, random_state=2923 + fold)
            with np.load(folder / f'fold{fold}.npz') as a:
                assert np.array_equal(a['valid_ids'], ids[va]) and np.array_equal(a['train_ids'], ids[tr])
                assert np.array_equal(a['tids'], tids)
                mid, sd, cm, cs, pt = [a[k] for k in ('mid', 'sd', 'cm', 'cs', 'pt')]
            net = Identify(c.shape[1]).to('cpu')
            net.load_state_dict(torch.load(folder / f'fold{fold}.pt', map_location='cpu', weights_only=True))
            net.eval()

            def predict(ix):
                with torch.no_grad():
                    return np.vstack([net(torch.tensor(((x[j] - mid) / sd).clip(-5, 5), device='cpu'), torch.tensor(((c[j] - cm) / cs).clip(-6, 6), device='cpu'))[0].cpu().numpy() for j in np.array_split(ix, max(1, len(ix) // 320))])
            np.testing.assert_allclose(predict(va[:160]), original[va[:160]], atol=5e-05)
            p = predict(inner)
            co = np.stack([calibration(p[:, k], y[inner]) for k in range(2)], axis=1)
            aligned[va] = co[0] + co[1] * original[va]
            tests.append(co[0] + co[1] * pt[:, :2])
            fits.append(co.tolist())
        last = {}
        for k in (0, 1):
            last[str(k)] = {label: aucs(y, v[:, k], parts) for label, v in (('original', original), ('aligned', aligned))}
        results[name] = dict(quality=last, fold_parameters=fits)
        print(name, last, flush=True)
        np.savez_compressed(OUT / (name + '_aligned.npz'), ids=ids.to_numpy(dtype=str), tids=tids.to_numpy(dtype=str), oof=aligned, test=np.mean(tests, 0))
    (OUT / 'other_fold_alignment.json').write_text(json.dumps(results, indent=2))
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        torch.set_num_threads(1)
        main()
