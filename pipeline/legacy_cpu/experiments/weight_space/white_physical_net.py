# CPU port; original experiments/weight_space/white_physical_net.py, SHA256 d6c3fe57a10aa17e1413fd0d96c11b9ce801491e51e07da4809c3b8f75897144
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Adapt real-series sequence checkpoints to physical levels and innovations."""
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
import json
import hashlib
import math
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits
from sequence_edits import ROOT, OUT, seq
import white_loop as wl

def signals(part):
    with np.load(OUT / f'pulse_streams_{part}.npz') as a:
        ids = a['ids']
        x = a['separate']
        n = a['n']
        b = a['b']
    with np.load(OUT / f'state_streams_{part}.npz') as a:
        assert np.array_equal(ids, a['ids'])
        e = a['common']
    return (ids, np.stack([x, e], axis=1), n, b)

def prepare_input(x, n, b, rv, fl):
    t = torch.arange(x.shape[-1], device=x.device)[None, :]
    idx = torch.where(rv[:, None], n[:, None] - 1 - t, t).clamp(min=0)
    s = torch.gather(x, 2, idx[:, None, :].expand(-1, 2, -1))
    s = torch.where(fl[:, None, None], -s, s)
    b = torch.where(rv, n - b, b)
    width = int((int(n.max()) + 63) // 64 * 64)
    t = t[:, :width]
    s = s[:, :, :width]
    pre = (t < b[:, None]).float()
    valid = (t < n[:, None]).float()
    post = valid - pre
    mean = (s * pre[:, None, :]).sum(2) / b[:, None]
    s = s - mean[:, :, None]
    sd = ((s * s * pre[:, None, :]).sum(2) / b[:, None]).sqrt().clamp_min(1e-05)
    s = (s / sd[:, :, None]).clamp(-10, 10) * valid[:, None, :]
    z, e = s.unbind(1)
    ch = torch.stack([z, e, e.abs(), (e * e).clamp(max=25) / 5, post, t / n[:, None].float() * valid], 1)
    qs = torch.cat([seq.mquant(s, v[:, None, :], 16).flatten(1) for v in (pre, post)], 1)
    return (ch, pre, post, qs)

class Detector(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.sequence = seq.Net()
        self.evidence = nn.Sequential(nn.Linear(dim, 48), nn.GELU(), nn.Dropout(0.15), nn.Linear(48, 2))
        nn.init.zeros_(self.evidence[-1].weight)
        nn.init.zeros_(self.evidence[-1].bias)

    def forward(self, raw, n, b, a, rv, fl):
        x, pre, post, q = prepare_input(raw, n, b, rv, fl)
        with _cpu_context.nullcontext():
            old = self.sequence(x, pre, post, x[:, 0, :0], q).float()
            intercept, slope = self.evidence(a).float().unbind(1)
        return old * (1 + 0.5 * torch.tanh(slope)) + intercept

@torch.no_grad()
def predict(model, gpu, aux, indices, views):
    model.eval()
    x, n, b = gpu
    idx = torch.as_tensor(indices, device='cpu')
    order = n[idx].argsort()
    idx = idx[order]
    parts = []
    for sub in idx.split(192):
        vals = []
        for rv, fl in views:
            r, f = (torch.full_like(n[sub], p, dtype=torch.bool) for p in (rv, fl))
            vals.append(model(x[sub], n[sub], b[sub], aux[sub], r, f))
        parts.append(torch.stack(vals).mean(0))
    return torch.cat(parts)[torch.argsort(order)].cpu().numpy()

def run(epochs):
    seq.a.hq = 0
    torch.set_num_threads(1)
    q, ids, tids, y, *_ = wl.load()
    ti, x, n, b = signals('train')
    ei, xe, ne, be = signals('test')
    assert np.array_equal(ti, ids) and np.array_equal(ei, tids)
    num = len(y)
    te = num + np.arange(len(tids))
    gpu = [torch.as_tensor(v, device='cpu') for v in (np.concatenate([x, xe]), np.r_[n, ne], np.r_[b, be])]
    del x, xe
    yt = torch.as_tensor(y, dtype=torch.float32, device='cpu')
    tags = [('state', ['pre2', 'pre_ar1', 'pre_cluster', 'post_cluster', 'walk']), ('curve', ['gate']), ('periodic', ['gate']), ('pulse', ['gate']), ('innov_v2', ['white', 'pre_white'])]
    fields = []
    for name, cols in tags:
        fields.append(np.vstack([pd.read_parquet(OUT / f'{name}_{p}.parquet', columns=cols).loc[ix].to_numpy() for p, ix in (('train', ids), ('test', tids))]))
    field = np.column_stack(fields)
    dst = OUT / f'physical_ft_{epochs}'
    dst.mkdir(exist_ok=True)
    hashes = {Path(a['file']): a['sha256'] for a in json.loads((ROOT / 'new weights/model_manifest.json').read_text())['files']}
    all_views = [(False, False), (True, True), (True, False), (False, True)]
    oof = np.zeros(num)
    tests = []
    paths = []
    for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(ids, y)):
        finished = dst / f'fold{fold}.npz'
        paths.extend([finished, dst / f'fold{fold}.pt'])
        if finished.exists():
            with np.load(finished) as a:
                assert np.array_equal(a['va'], va)
                oof[va] = a['pv']
                tests.append(a['pt'])
            continue
        prefix = OUT / 'law_outer0'
        with np.load(prefix / f'fold{fold}_rows.npz') as a:
            assert np.array_equal(a['validation_ids'], ids[va])
        law = np.vstack([pd.read_parquet(prefix / f'fold{fold}_{p}.parquet').loc[ix].to_numpy() for p, ix in (('train', ids), ('test', tids))])
        features = np.column_stack([np.arcsinh(law), field])
        core, inner = train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=1000 * fold)
        center, sd = (features[core].mean(0), features[core].std(0).clip(0.12))
        aux = torch.as_tensor(((features - center) / sd).clip(-6, 6), dtype=torch.float32, device='cpu')
        torch.manual_seed(167 + fold)
        checkpoint = Path('models/W_SEQ_q') / f'fold{fold}_seed0.pt'
        path = ROOT / 'new weights' / checkpoint
        assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes[checkpoint]
        net = Detector(aux.shape[1]).to('cpu')
        net.sequence.load_state_dict(torch.load(path, weights_only=True, map_location='cpu'))
        for m in (net.sequence.stem, *net.sequence.blocks[:4]):
            m.requires_grad_(False)
        opt = torch.optim.AdamW([{'params': [p for p in net.sequence.parameters() if p.requires_grad], 'lr': 0.0001}, {'params': net.evidence.parameters(), 'lr': 0.0013}], weight_decay=0.008)
        rng = np.random.default_rng(893 + fold)
        best = -1.0
        bestep = None
        for ep in range(epochs):
            net.train()
            for m in (net.sequence.stem, *net.sequence.blocks[:4]):
                m.eval()
            order = rng.permutation(core)
            for start in range(0, len(core), 128):
                ix = torch.as_tensor(order[start:start + 128], device='cpu')
                count = len(ix)
                rv, fl = (torch.rand(count, device='cpu') < 0.5 for _ in range(2))
                opt.zero_grad(set_to_none=True)
                score = net(gpu[0][ix], gpu[1][ix], gpu[2][ix], aux[ix], rv, fl)
                loss = F.binary_cross_entropy_with_logits(score, yt[ix])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                opt.step()
            result = predict(net, gpu, aux, inner, all_views[:2])
            auc = roc_auc_score(y[inner], result)
            if auc > best + 1e-05:
                best, bestep = (auc, ep)
                torch.save(net.state_dict(), dst / f'fold{fold}.pt')
            if ep in (0, epochs - 1) or ep - bestep >= 3:
                print('physical net', fold, ep, round(auc, 5), 'best', round(best, 5), flush=True)
            if ep - bestep >= 3:
                break
        net.load_state_dict(torch.load(dst / f'fold{fold}.pt', map_location='cpu', weights_only=True))
        pv = predict(net, gpu, aux, va, all_views)
        pt = predict(net, gpu, aux, te, all_views)
        np.savez_compressed(finished, va=va, pv=pv, pt=pt, center=center, sd=sd, bestep=bestep)
        print('physical net validation', fold, roc_auc_score(y[va], pv), flush=True)
        oof[va] = pv
        tests.append(pt)
    logits = np.r_[oof, np.mean(tests, axis=0)]

    def design(ix, z):
        d = (logits[ix] - z).clip(-8, 8)
        return np.column_stack([d, d * np.tanh(z / 2), d * field[ix, 2], d * field[ix, -1], d * field[ix, 4]])
    wl.run('physical_ft' + str(epochs), design, 70.0, dependencies=[Path(__file__).resolve(), *paths])
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=8)
    with threadpool_limits(limits=1):
        run(parser.parse_args().epochs)
