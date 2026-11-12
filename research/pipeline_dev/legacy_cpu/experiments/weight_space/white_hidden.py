# CPU port; original experiments/weight_space/white_hidden.py, SHA256 c7e6999f6756f16929da5c14b20db0d04c00556ba0abd71b67d4805f13c18357
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Refit frozen sequence readouts on real, nuisance-corrected inputs."""
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits
from sequence_edits import ROOT, OUT, seq
from physical_870 import fit
import white_loop as wl

def read_streams(part):
    raw = np.load(ROOT / f'tune_out/seq_cache_{part}.npz', allow_pickle=True)
    ids = raw['ids'].astype(str)
    arrays = [raw['x']]
    for name in ('pulse', 'state'):
        with np.load(OUT / f'{name}_streams_{part}.npz') as p:
            assert np.array_equal(p['ids'], ids)
            arrays.append(p['separate'])
    return (ids, [torch.as_tensor(v, device='cpu') for v in [*arrays, raw['n'], raw['b']]])

def extract(tag):
    torch.set_num_threads(1)
    q, ids, tids, y, *_ = wl.load()
    train, test = (read_streams('train'), read_streams('test'))
    assert np.array_equal(train[0], ids) and np.array_equal(test[0], tids)
    src = ROOT / 'new weights'
    hashes = {Path(r['file']): r['sha256'] for r in json.loads((src / 'model_manifest.json').read_text())['files']}
    folder = OUT / f'hidden_{tag}'
    folder.mkdir(exist_ok=True)
    hq, seed = seq.RUNS[tag]
    seq.a.hq = hq
    for i, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(ids, y)):
        dst = folder / f'fold{i}.npz'
        if dst.exists():
            continue
        _, inner = train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=1000 * i + seed)
        checkpoint = Path('models') / tag / f'fold{i}_seed{seed}.pt'
        assert hashlib.sha256((src / checkpoint).read_bytes()).hexdigest() == hashes[checkpoint]
        model = seq.Net().to('cpu').eval()
        model.load_state_dict(torch.load(src / checkpoint, map_location='cpu', weights_only=True))
        hook = model.head[6].register_forward_pre_hook(lambda m, inputs: seq.CAP.append(inputs[0].float()))
        saved = dict(va=va, inner=inner)
        for part, vectors, ix in (('inner', train[1], inner), ('va', train[1], va), ('test', test[1], np.arange(len(tids)))):
            idx = torch.as_tensor(ix, device='cpu')
            n, b = (a[idx] for a in vectors[-2:])
            for j, v in enumerate(vectors[:-2]):
                x = v[idx]
                logits, hidden = seq.predict(model, x, n, b, x[:, :0])
                if part in ('va', 'test') and j == 0:
                    with np.load(ROOT / f'tune_out/seqpart/{tag}_f{i}.npz') as ref:
                        diff = abs(logits - ref['pv' if part == 'va' else 'pt'])
                        assert diff.mean() < 0.002 and diff.max() < 0.05
                saved[f'{part}_{j}'] = hidden.mean(0).cpu().numpy()
            print(tag, i, part, flush=True)
        hook.remove()
        np.savez_compressed(dst, **saved)

def evaluate(tag):
    q, ids, tids, y, *_ = wl.load()
    oof = np.zeros((len(ids), 3))
    test = np.zeros((len(tids), 3))
    paths = []
    for i in range(5):
        path = OUT / f'hidden_{tag}/fold{i}.npz'
        paths.append(path)
        with np.load(path) as a:
            inner, va = (a['inner'], a['va'])
            k = f's0_f{i}_'
            assert np.array_equal(va, q[k + 'va'])
            z = pd.Series(q[k + 'zr'], index=q[k + 'tr']).loc[inner].to_numpy()
            for j in range(3):
                inp = a[f'inner_{j}'].astype(float)
                center = inp.mean(0)
                scale = inp.std(0).clip(0.08)

                def design(h):
                    return np.column_stack([((h - center) / scale).clip(-8, 8), np.ones(len(h))])
                w = fit(design(inp), z, y[inner], 180.0)
                oof[va, j] = design(a[f'va_{j}']) @ w
                test[:, j] += design(a[f'test_{j}']) @ w / 5
    f = np.vstack([oof, test])

    def transform(ix, z):
        return np.column_stack([f[ix], f[ix] * np.tanh(z[:, None] / 2)])
    wl.run('hidden_' + tag, transform, 60.0, dependencies=[Path(__file__).resolve(), *paths])
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--tag', default='W_SEQ_q')
    a = p.parse_args()
    with threadpool_limits(limits=1):
        extract(a.tag)
        evaluate(a.tag)
