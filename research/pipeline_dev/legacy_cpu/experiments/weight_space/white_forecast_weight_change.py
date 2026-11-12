# CPU port; original experiments/weight_space/white_forecast_weight_change.py, SHA256 e13eccbe9903494c295b3e316fa8df77b53dacef9577d339e30dc7aee8fd2607
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Compare which forecast weights the observed data push on across the boundary."""
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits
from partition_evidence import OUT, ROOT
from white_forecast_density import Dynamics, prepare, timing
from white_signature_screen import pairs

def weight_scores(model, x, context, lengths, cuts, check=False):
    time = torch.arange(x.shape[2], device=x.device)[None].expand(len(x), -1)
    side, own, _ = timing(lengths, cuts, time)
    parity = 1 - own % 2
    capture = []
    hook = model.output.register_forward_hook(lambda layer, args, output: capture.append((args[0], output)))
    try:
        likelihood = model(x, context, 1 - parity)
    finally:
        hook.remove()
    hidden, logits = capture[0]
    gradient, = torch.autograd.grad(likelihood.sum(), logits, retain_graph=bool(check))
    with torch.no_grad():
        hidden = torch.cat([torch.ones_like(hidden[..., :1]), hidden], -1)
        block = 2 * side + parity
        within = torch.where(side.bool(), time - cuts[:, None], time)
        valid = (time < lengths[:, None]) & (within >= 16)
        gradients, counts = ([], [])
        for group in range(4):
            mask = valid & (block == group)
            count = mask.sum(1)
            assert (count > 0).all()
            means = torch.einsum('bctd,bcth->bcdh', gradient * mask[:, None, :, None], hidden)
            gradients.append(means[:, :, :15] / count[:, None, None, None])
            counts.append(count)
    if check:
        chosen = valid & (block == 0)
        numerator = (likelihood * chosen[:, None]).sum()
        direct, = torch.autograd.grad(numerator, model.output.weight)
        reconstructed = (gradients[0] * counts[0][:, None, None, None]).sum((0, 1))[..., 1:]
        torch.testing.assert_close(reconstructed, direct[:15], rtol=0.003, atol=0.002)
    return (torch.stack(gradients, 1).detach().cpu().numpy(), torch.stack(counts, 1).cpu().numpy())

def features(scores, counts, null):
    pre, post = (scores[:, :2], scores[:, 2:])
    size = np.sqrt(counts[:, :2] * counts[:, 2:] / (counts[:, :2] + counts[:, 2:]))
    out = {}
    for channel, sensor in enumerate(('pulse', 'state')):
        for g, meaning in enumerate(('centers', 'widths', 'occupancy')):
            sl = slice(5 * g, 5 * g + 5)
            grad = (post - pre)[:, :, channel, sl]
            for layer, weights in (('bias', grad[..., :1]), ('conditioning', grad[..., 1:])):
                x = weights.reshape(len(grad), 2, -1) * size[:, :, None]
                reference = x[null].reshape(-1, x.shape[-1])
                mean = reference.mean(0)
                cov = np.cov(reference, rowvar=False)
                d, v = np.linalg.eigh(cov)
                whitening = v / np.sqrt(d.clip(0) + 0.06 * np.mean(d) + 1e-08)
                z = (x - mean) @ whitening
                tag = f'{sensor}_{meaning}_{layer}'
                out[tag + '_repeat'] = np.mean(z[:, 0] * z[:, 1], axis=1)
                out[tag + '_energy'] = np.mean(z.mean(1) ** 2, axis=1)
    return pd.DataFrame(out)

def run(learned=False):
    a, b = (prepare('train'), prepare('test'))
    ids, tids = (a[0], b[0])
    n = len(ids)
    x, c, N, B = [torch.as_tensor(np.concatenate([a[k], b[k]]), device='cpu') for k in (3, 4, 1, 2)]
    del a, b
    labels = pd.read_csv(ROOT / 'data/train_labels.csv').set_index('id').loc[ids, 'label'].to_numpy()
    name = 'forecast_break_directions' if learned else 'forecast_weight_change'
    dst = OUT / ('forecast_directions_models' if learned else 'forecast_weight_models')
    dst.mkdir(exist_ok=True)
    trains, tests = ([], [])
    for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(ids, labels)):
        path = dst / f'fold{fold}.parquet'
        core, _ = train_test_split(tr, test_size=0.12, stratify=labels[tr], random_state=485 + fold)
        if not path.exists():
            source = OUT / 'forecast_density_models'
            with np.load(source / f'fold{fold}.npz') as archive:
                assert np.array_equal(archive['validation_ids'], ids[va])
            model = Dynamics(c.shape[-1] - 3).to('cpu').eval()
            model.load_state_dict(torch.load(source / f'forecast{fold}.pt', map_location='cpu', weights_only=True))
            model.requires_grad_(False)
            model.output.requires_grad_(True)
            pieces, sizes = ([], [])
            for j in np.array_split(np.arange(len(x)), (len(x) + 63) // 64):
                width = int(N[j].max())
                scores, counts = weight_scores(model, x[j, :, :width].float(), c[j], N[j], B[j], check=j[0] == 0)
                pieces.append(scores)
                sizes.append(counts)
            if learned:
                from white_forecast_directions import directions
                frame = directions(np.vstack(pieces), np.vstack(sizes), core, labels, ids)
            else:
                frame = features(np.vstack(pieces), np.vstack(sizes), core[labels[core] == 0])
            frame.index = np.concatenate([ids, tids])
            assert np.isfinite(frame).all().all()
            frame.loc[np.r_[ids[va], tids]].to_parquet(path)
        frame = pd.read_parquet(path)
        assert np.array_equal(frame.index, np.r_[ids[va], tids])
        trains.append(frame.loc[ids[va]])
        tests.append(frame.loc[tids])
        print('forecast weight gradients', fold, flush=True)
    train = pd.concat(trains).loc[ids]
    train.to_parquet(OUT / f'{name}_train.parquet')
    (sum(tests) / len(tests)).to_parquet(OUT / f'{name}_test.parquet')
    ix, h, controls, _ = pairs()
    repeat = train.loc[ix].filter(like='_repeat').mean(axis=1).to_numpy()
    delta = repeat[h] - repeat[controls]
    rng = np.random.default_rng(28891)
    draws = np.concatenate([np.mean(rng.choice(delta, (512, len(h))), axis=1) for _ in range(32)])
    report = dict(n=len(h), average_break=float(repeat[h].mean()), average_control=float(repeat[controls].mean()), gap=float(delta.mean()), paired_win=float(np.mean(delta > 0)), paired_95_interval=np.quantile(draws, [0.025, 0.975]).tolist(), scope='Fixed average repeated gradient change on the previously matched 808 pairs.')
    (OUT / f'{name}_matched.json').write_text(json.dumps(report, indent=2))
    print(report, flush=True)
if __name__ == '__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        run()
