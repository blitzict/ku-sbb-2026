# CPU port; original experiments/weight_space/export_candidate.py, SHA256 72e67a7f04f098a2bcd06ba5692a43bd83dc86abe5ddda5327201be87234315a
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Save the common-mode parameter replacements and replay all 25 patched nets."""
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
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits
from sequence_edits import ROOT, OUT, make_edits, seq
from evaluate import sequence_logits, inputs
from finalize import write_submission
DEST = ROOT / 'new weights/common_mode_candidate'
PATCH = DEST / 'head_replacements.pt'
META = DEST / 'manifest.json'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def path_for(tag, fold, seed):
    return f'models/{tag}/fold{fold}_seed{seed}.pt'

def export():
    DEST.mkdir(exist_ok=True)
    saved, hashes = ({}, {})
    for tag, (hq, seed) in seq.RUNS.items():
        seq.a.hq = hq
        for fold in range(5):
            key = path_for(tag, fold, seed)
            path = ROOT / 'new weights' / key
            state = torch.load(path, weights_only=True, map_location='cpu')
            model = seq.Net().eval().to('cpu')
            model.load_state_dict(state)
            _, head = make_edits(model)
            original = state['head.0.weight']
            width = 2 * ((original.shape[1] - 66) // 4)
            edited = head[0].weight.detach().cpu()
            assert torch.equal(edited[:, width:], original[:, width:]) and (not torch.equal(edited[:, :width], original[:, :width]))
            saved[key] = edited[:, :width].contiguous()
            hashes[key] = digest(path)
    torch.save(saved, PATCH)
    meta = {'tensor': 'head.0.weight', 'slice': 'all rows, leftmost columns up to the width of each replacement', 'base_sha256': hashes, 'patch_sha256': digest(PATCH), 'ensemble_sha256': digest(ROOT / 'new weights/ensemble.json'), 'evaluation': json.loads((OUT / 'sequence_results.json').read_text()), 'promoted': False}
    META.write_text(json.dumps(meta, indent=2))

@torch.inference_mode()
def replay(output, compare=False):
    meta = json.loads(META.read_text())
    assert digest(PATCH) == meta['patch_sha256']
    patches = torch.load(PATCH, weights_only=True, map_location='cpu')
    base = ROOT / 'new weights/ensemble.json'
    assert digest(base) == meta['ensemble_sha256']
    item = json.loads(base.read_text())
    for name, expected in item['input_hashes'].items():
        assert digest(ROOT / name) == expected, name
    _, _, ids, _, _, _, _, _, VT, names, _, _ = inputs(True)
    data = np.load(ROOT / 'tune_out/seq_cache_test.npz', allow_pickle=True)
    assert np.array_equal(ids, data['ids'])
    rows = [torch.as_tensor(data[k], device='cpu') for k in ('x', 'n', 'b')]
    result = []
    for tag, (hq, seed) in seq.RUNS.items():
        seq.a.hq = hq
        folds = []
        for fold in range(5):
            key = path_for(tag, fold, seed)
            path = ROOT / 'new weights' / key
            assert digest(path) == meta['base_sha256'][key]
            state = torch.load(path, weights_only=True, map_location='cpu')
            replacement = patches[key]
            state['head.0.weight'][:, :replacement.shape[1]].copy_(replacement)
            model = seq.Net().eval().to('cpu')
            model.load_state_dict(state, strict=True)
            hook = model.head[6].register_forward_pre_hook(lambda layer, args: seq.CAP.append(args[0].float()))
            pred, _ = seq.predict(model, *rows, rows[0][:, :0])
            hook.remove()
            if compare:
                with np.load(OUT / f'{tag}_f{fold}_test.npz') as data_ref:
                    assert np.array_equal(data_ref['ids'], ids)
                    assert np.array_equal(pred, data_ref['predictions'][2])
            folds.append(pred.mean(axis=0))
        result.append(logit(np.clip(expit(np.mean(folds, axis=0).astype(float)), 1e-06, 1 - 1e-06)))
        print(f'Replayed {tag}', flush=True)
    V = VT[:, [names.index(k) for k in item['feature_order']]].copy()
    th = np.array(item['coefficients_and_intercept'])
    control = pd.read_csv(ROOT / 'new weights/submission.csv', dtype={'id': str}).set_index('id').loc[ids]
    assert np.max(np.abs(expit(V @ th[:-1] + th[-1]) - control.break_probability.to_numpy())) < 1e-13
    j = item['feature_order'].index('W_SEQ_qb5')
    if compare:
        all_variants = sequence_logits(ids, 'test')
        assert np.max(np.abs(V[:, j] - all_variants[0])) < 2e-05
        V[:, j] = all_variants[2]
        write_submission(expit(V @ th[:-1] + th[-1]), ids, DEST / 'submission.csv')
    V[:, j] = np.mean(result, axis=0)
    write_submission(expit(V @ th[:-1] + th[-1]), ids, output)
    if compare:
        assert digest(output) == digest(DEST / 'submission.csv')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay', type=Path, help='rebuild a CSV directly from the saved patch')
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        if args.replay:
            replay(args.replay)
        else:
            export()
            replay(DEST / 'submission_replay.csv', compare=True)
