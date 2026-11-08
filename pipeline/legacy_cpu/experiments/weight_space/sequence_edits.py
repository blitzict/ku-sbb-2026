# CPU port; original experiments/weight_space/sequence_edits.py, SHA256 b371244b0cf9eba794d05da6f13bbfccc8ddb20d777076d35fade626d9a2b4f9
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Run the frozen and directly edited sequence heads on held-out series."""
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
import copy
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.nn import functional as F
ROOT = _cpu_workspace()
sys.path.insert(0, str(ROOT / 'experiments/follow_gpt'))
import seq_head_edit as seq
OUT = ROOT / 'tune_out/weight_space'
VARIANTS = ('identity', 'smooth', 'common', 'both')

@torch.no_grad()
def make_edits(model):
    quantiles = copy.deepcopy(model.qn)
    weight = quantiles.weight.reshape(64, 6, 16)
    padded = F.pad(weight, (1, 1), mode='replicate')
    smooth = 0.25 * padded[..., :-2] + 0.5 * weight + 0.25 * padded[..., 2:]
    torch.testing.assert_close(smooth.sum(-1), weight.sum(-1), atol=2e-06, rtol=1e-05)
    weight.copy_(smooth)
    head = copy.deepcopy(model.head)
    n = (head[0].in_features - 66) // 4
    assert n in (192, 640) and head[0].in_features == 4 * n + 66
    before, after = (head[0].weight[:, :n], head[0].weight[:, n:2 * n])
    common = (before + after) / 2
    contrast = after - before
    before.sub_(0.25 * common)
    after.sub_(0.25 * common)
    torch.testing.assert_close(after - before, contrast)
    return (quantiles, head)

@torch.inference_mode()
def predict(model, rows):
    quantiles, head = make_edits(model)
    captured = {}

    def capture(name):

        def hook(module, args):
            captured[name] = args[0]
        return hook
    handles = [model.qn.register_forward_pre_hook(capture('q')), model.head.register_forward_pre_hook(capture('h'))]
    x, n, b = rows
    order = torch.argsort(n)
    x, n, b = (x[order], n[order], b[order])
    chunks = []
    for start in range(0, len(n), 512):
        sl = slice(start, start + 512)
        aux = x[sl, :0]
        views = []
        for reverse, flip in ((False, False), (False, True), (True, False), (True, True)):
            masks = [torch.full_like(n[sl], flag, dtype=torch.bool) for flag in (reverse, flip)]
            original = seq.run(model, x[sl], n[sl], b[sl], aux, *masks)
            h = captured['h']
            with _cpu_context.nullcontext():
                edited_h = torch.cat([h[:, :-64], F.gelu(quantiles(captured['q']))], 1)
                values = [original, model.head(edited_h)[:, 0], head(h)[:, 0], head(edited_h)[:, 0]]
            views.append(torch.stack(values).float())
        chunks.append(torch.stack(views, dim=1).cpu())
    for handle in handles:
        handle.remove()
    return torch.cat(chunks, dim=2)[:, :, torch.argsort(order).cpu()].numpy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='also replay edits on test rows')
    args = parser.parse_args()
    part = 'test' if args.test else 'train'
    OUT.mkdir(exist_ok=True)
    data = np.load(ROOT / f'tune_out/seq_cache_{part}.npz', allow_pickle=True)
    ids = pd.read_parquet(ROOT / f'clean/features_{part}.parquet', columns=[]).index
    assert np.array_equal(ids, data['ids'])
    data_gpu = [torch.as_tensor(data[k], device='cpu') for k in ('x', 'n', 'b')]
    labels = pd.read_csv(ROOT / 'data/train_labels.csv', dtype={'id': str}).set_index('id')
    train_ids = pd.read_parquet(ROOT / 'clean/features_train.parquet', columns=[]).index
    folds = StratifiedKFold(5, shuffle=True, random_state=0).split(train_ids, labels.loc[train_ids, 'label'])
    manifest = json.loads((ROOT / 'new weights/model_manifest.json').read_text())
    hashes = {entry['file'].replace(chr(92), '/'): entry['sha256'] for entry in manifest['files']}
    import hashlib
    for fold, (_, valid) in enumerate(folds):
        for tag, (hq, seed) in seq.RUNS.items():
            relative = f'models/{tag}/fold{fold}_seed{seed}.pt'
            checkpoint = ROOT / 'new weights' / relative
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == hashes[relative]
            seq.a.hq = hq
            model = seq.Net().eval().to('cpu')
            model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cpu'))
            reference = np.load(ROOT / f'tune_out/seqpart/{tag}_f{fold}.npz')
            assert np.array_equal(reference['va'], valid)
            rows = data_gpu if args.test else [v[torch.as_tensor(valid, device='cpu')] for v in data_gpu]
            predictions = predict(model, rows)
            err = np.abs(predictions[0] - reference['pt' if args.test else 'pv'])
            assert np.isfinite(predictions).all() and err.mean() < 0.002 and (err.max() < 0.05), (err.max(), err.mean())
            target = OUT / f'{tag}_f{fold}_{part}.npz'
            np.savez(target, names=VARIANTS, predictions=predictions, ids=np.array(ids if args.test else ids[valid], dtype=str))
            print(f'{target.name} identity max logit error {err.max():.6g}', flush=True)
if __name__ == '__main__':
    main()
