# CPU port; original experiments/weight_space/boundary_probe.py, SHA256 268cb22ad7a88cd69830df716cf593bcd7c786272f8ad14738923a900592d758
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Run shared copies of the packaged sequence detector at five boundary locations."""
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
from sequence_edits import ROOT, OUT, seq
BRANCHES = ('given', 'earlier', 'later', 'before_only', 'after_only')

def branches(x, n, b):
    shift = (n + 4) // 8
    yield (x, n, b)
    yield (x, n, b - shift)
    yield (x, n, b + shift)
    yield (x, b, b // 2)
    offset = b[:, None] + torch.arange(x.shape[1], device=x.device)[None, :]
    after = x.gather(1, offset.clamp_max(x.shape[1] - 1))
    yield (after, n - b, (n - b) // 2)

@torch.inference_mode()
def predict(models, x, n, b):
    branches_out = []
    for bx, bn, bb in branches(x, n, b):
        assert bool(torch.all((bb > seq.P) & (bn - bb > seq.P)))
        order = torch.argsort(bn)
        bx, bn, bb = (bx[order], bn[order], bb[order])
        chunks = []
        for start in range(0, len(bn), 512):
            sl = slice(start, start + 512)
            views = []
            for reverse, flip in ((False, False), (False, True), (True, False), (True, True)):
                masks = [torch.full_like(bn[sl], flag, dtype=torch.bool) for flag in (reverse, flip)]
                xv, nv, bv = seq.view(bx[sl], bn[sl], bb[sl], *masks)
                xc, bef, aft, q = seq.channels(xv, nv, bv)
                outputs = []
                with _cpu_context.nullcontext():
                    for hq, model in models:
                        seq.a.hq = hq
                        outputs.append(model(xc, bef, aft, xv[:, :0], q).float())
                views.append(torch.stack(outputs))
            chunks.append(torch.stack(views, dim=1).cpu())
        branches_out.append(torch.cat(chunks, dim=-1)[:, :, torch.argsort(order).cpu()])
    return torch.stack(branches_out).numpy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', choices=['train', 'test'], default='train')
    args = parser.parse_args()
    part = args.part
    OUT.mkdir(exist_ok=True)
    data = np.load(ROOT / f'tune_out/seq_cache_{part}.npz', allow_pickle=True)
    ids = pd.read_parquet(ROOT / f'clean/features_{part}.parquet', columns=[]).index
    assert np.array_equal(data['ids'], ids)
    gpu = [torch.as_tensor(data[k], device='cpu') for k in ('x', 'n', 'b')]
    train_ids = pd.read_parquet(ROOT / 'clean/features_train.parquet', columns=[]).index
    labels = pd.read_csv(ROOT / 'data/train_labels.csv', dtype={'id': str}).set_index('id').loc[train_ids, 'label']
    manifest = json.loads((ROOT / 'new weights/model_manifest.json').read_text())
    hashes = {Path(row['file']): row['sha256'] for row in manifest['files']}
    for fold, (_, valid) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(train_ids, labels)):
        models, refs = ([], [])
        for tag, (hq, seed) in seq.RUNS.items():
            seq.a.hq = hq
            model = seq.Net().eval().to('cpu')
            path = Path('models') / tag / f'fold{fold}_seed{seed}.pt'
            checkpoint = ROOT / 'new weights' / path
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == hashes[path]
            model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cpu'))
            models.append((hq, model))
            with np.load(ROOT / f'tune_out/seqpart/{tag}_f{fold}.npz') as archive:
                assert np.array_equal(archive['va'], valid)
                refs.append(archive['pv' if part == 'train' else 'pt'])
        rows = [a[torch.as_tensor(valid, device='cpu')] for a in gpu] if part == 'train' else gpu
        pred = predict(models, *rows)
        err = np.abs(pred[0] - np.stack(refs))
        assert np.isfinite(pred).all() and err.max() < 0.05 and (err.mean() < 0.002), (err.max(), err.mean())
        fold_ids = ids[valid] if part == 'train' else ids
        np.savez(OUT / f'boundary_{part}_f{fold}.npz', logits=pred.mean(axis=2), ids=fold_ids.to_numpy(dtype=str), branches=BRANCHES, tags=list(seq.RUNS), max_replay_error=err.max())
        print(f'{part} fold {fold}: five probes complete, max replay error {err.max():.5g}', flush=True)
if __name__ == '__main__':
    main()
