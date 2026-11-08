# CPU port; original experiments/weight_space/periodic_replay.py, SHA256 a067fd6bb8cc6674bd1bd554d128bc8e5c1c621a1336707dfcba7b35a2211a07
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Run verified sequence checkpoints on the original and de-seasonalized series."""
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
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from sequence_edits import ROOT, OUT, seq

@torch.inference_mode()
def predict(models, x, n, b):
    order = torch.argsort(n)
    x, n, b = (x[order], n[order], b[order])
    blocks = []
    for lo in range(0, len(n), 512):
        sl = slice(lo, lo + 512)
        views = []
        for rv, fl in ((False, False), (False, True), (True, False), (True, True)):
            masks = [torch.full_like(n[sl], v, dtype=torch.bool) for v in (rv, fl)]
            xv, nv, bv = seq.view(x[sl], n[sl], b[sl], *masks)
            xc, bef, aft, q = seq.channels(xv, nv, bv)
            with _cpu_context.nullcontext():
                logits = []
                for hq, model in models:
                    seq.a.hq = hq
                    logits.append(model(xc, bef, aft, xv[:, :0], q).float())
                views.append(torch.stack(logits))
        blocks.append(torch.stack(views, dim=1).cpu())
    return torch.cat(blocks, dim=2)[:, :, torch.argsort(order).cpu()].numpy()

def streams(part, prefix):
    index = pd.read_parquet(ROOT / f'clean/features_{part}.parquet', columns=[]).index
    z = np.load(ROOT / f'tune_out/seq_cache_{part}.npz', allow_pickle=True)
    out = np.load(OUT / f'{prefix}_streams_{part}.npz')
    assert np.array_equal(index, z['ids']) and np.array_equal(index, out['ids'])
    assert np.array_equal(z['n'], out['n']) and np.array_equal(z['b'], out['b'])
    gpu = [torch.as_tensor(arr, device='cpu') for arr in (z['x'], out['common'], out['separate'], z['n'], z['b'])]
    gate = pd.read_parquet(OUT / f'periodic_{part}.parquet', columns=['gate']).loc[index, 'gate'].to_numpy() if prefix == 'periodic' else np.ones(len(index))
    return (index, gpu, gate)

def run(prefix):
    parts = [streams(k, prefix) for k in ('train', 'test')]
    ids, tids = (p[0] for p in parts)
    y = pd.read_csv(ROOT / 'data/train_labels.csv', dtype={'id': str}).set_index('id').loc[ids, 'label']
    manifest = json.loads((ROOT / 'new weights/model_manifest.json').read_text())
    hashes = {Path(row['file']): row['sha256'] for row in manifest['files']}
    tags = list(seq.RUNS)
    cols = [prefix + '_' + t for prefix in ('original', 'common', 'separate') for t in tags]
    archives = OUT / f'{prefix}_replay'
    archives.mkdir(exist_ok=True)
    train = np.full((len(ids), len(cols)), np.nan)
    test = np.zeros((len(tids), len(cols)))
    folds = StratifiedKFold(5, shuffle=True, random_state=0).split(ids, y)
    for fold, (_, va) in enumerate(folds):
        path = archives / f'fold{fold}.npz'
        if path.exists():
            with np.load(path) as a:
                assert np.array_equal(a['ids'], ids[va]) and np.array_equal(a['tids'], tids)
                train[va], test = (a['val'], test + a['test'] / 5)
            continue
        models, refs, ref_tests = ([], [], [])
        for tag, (hq, seed) in seq.RUNS.items():
            seq.a.hq = hq
            model = seq.Net().eval().to('cpu')
            rel = Path('models') / tag / f'fold{fold}_seed{seed}.pt'
            ck = ROOT / 'new weights' / rel
            assert hashlib.sha256(ck.read_bytes()).hexdigest() == hashes[rel]
            model.load_state_dict(torch.load(ck, weights_only=True, map_location='cpu'))
            models.append((hq, model))
            with np.load(ROOT / f'tune_out/seqpart/{tag}_f{fold}.npz') as a:
                assert np.array_equal(va, a['va'])
                refs.append(a['pv'])
                ref_tests.append(a['pt'])
        outputs = []
        for j, (_, gpu, gate) in enumerate(parts):
            row = va if j == 0 else np.arange(len(tids))
            orig = np.stack(refs if j == 0 else ref_tests)
            selected = torch.as_tensor(row, device='cpu')
            before = orig.mean(1).T
            scores = [before]
            for i, x in enumerate(gpu[:3]):
                pred = predict(models, x[selected], gpu[3][selected], gpu[4][selected])
                if i == 0:
                    err = abs(pred - orig)
                    assert err.mean() < 0.002 and err.max() < 0.05, (err.mean(), err.max())
                else:
                    result = before.copy()
                    active = gate[row] > 0.075
                    result[active] = pred.mean(1).T[active]
                    scores.append(result)
            outputs.append(np.column_stack(scores))
        train[va] = outputs[0]
        test += outputs[1] / 5
        np.savez(path, ids=ids[va].to_numpy(dtype=str), tids=tids.to_numpy(dtype=str), val=outputs[0], test=outputs[1])
        print('seasonal sequence replay fold', fold, flush=True)
    assert np.isfinite(train).all() and np.isfinite(test).all()
    for name, tab, index in (('train', train, ids), ('test', test, tids)):
        pd.DataFrame(tab, index=index, columns=cols).to_parquet(OUT / f'{prefix}_seq_{name}.parquet')
if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('--prefix', choices=('periodic', 'innov_v2', 'curve', 'pulse', 'state'), default='periodic')
    with threadpool_limits(limits=2):
        run(arg.parse_args().prefix)
