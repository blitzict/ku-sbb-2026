# CPU port; original experiments/weight_space/other_mechanism_readout.py, SHA256 c8a9f64a907d155f5ba289eb8c306740175fa422f237767545acd26c533529d6
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Train a small additive network over complete measurement families on the latest scores."""
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
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits
from other_missed_breaks import CURRENT, rankings
from other_miss_witnesses import make
from hard_pair_budget import cohorts
from white_checkpoint import SEEDS
from white_nested import audit
from white_signature_screen import MEASURES
from partition_evidence import ROOT, OUT
import white_loop as wl
TABLES = (*MEASURES, 'memory_law1', 'memory_law2', 'strict_unit_memory', 'transition_catalog', 'other_memory_horizons', 'phase_coupling', 'shock_ringing', 'context_changes', 'time_arrow', 'run_age', 'memory_events', 'mode_switching', 'value_bands', 'conditional_noise_law', 'segment_population', 'reliable_signals')
TAG = 'local_rank_mechanism_additive'

def inputs(ids, tids):
    groups = []
    files = []
    for name in TABLES:
        blocks = []
        for part, ix in (('train', ids), ('test', tids)):
            file = OUT / f'{name}_{part}.parquet'
            blocks.append(pd.read_parquet(file).loc[ix])
            files.append(file)
        assert blocks[0].columns.equals(blocks[1].columns)
        v = pd.concat(blocks).to_numpy(dtype=np.float32)
        assert np.isfinite(v).all(), name
        groups.append(np.arcsinh(v).astype(np.float32))
    banks, paths = make(ids, tids)
    groups.append(np.column_stack([v.mean(1) for v in banks]).astype(np.float32))
    return (groups, [*files, *paths])

def compress(groups, core):
    coordinates = []
    for v in groups:
        mu = v[core].mean(0)
        sd = v[core].std(0).clip(0.06)
        z = ((v - mu) / sd).clip(-5, 5)
        cov = z[core].T @ z[core] / len(core)
        d, b = np.linalg.eigh(cov.astype(float))
        ix = np.flatnonzero(d > 0.015)[-8:][::-1]
        pc = np.zeros((len(z), 8), np.float32)
        if len(ix):
            pc[:, :len(ix)] = (z @ (b[:, ix] / np.sqrt(d[ix] + 0.25)).astype(np.float32)).clip(-5, 5)
        coordinates.append(pc)
    return torch.tensor(np.stack(coordinates, axis=1), device='cpu')

class Changes(nn.Module):

    def __init__(self, g):
        super().__init__()
        self.direction = nn.Parameter(torch.randn(g, 8, 6) * 0.25)
        self.bias = nn.Parameter(torch.zeros(g, 6))
        self.position = nn.Parameter(torch.randn(g, 6) * 0.06)
        self.shared = nn.Sequential(nn.Linear(13, 20), nn.SiLU(), nn.Dropout(0.2), nn.Linear(20, 2))
        self.direct = nn.Linear(8 * g, 1, bias=False)
        nn.init.zeros_(self.shared[-1].weight)
        nn.init.zeros_(self.shared[-1].bias)
        nn.init.zeros_(self.direct.weight)

    def forward(self, x, base):
        h = torch.einsum('bgd,gdh->bgh', x, self.direction) + self.bias
        context = torch.tanh(base[:, None, None] / 2).expand(-1, h.shape[1], 1)
        inp = torch.cat([torch.tanh(h), self.position.expand(len(h), -1, -1), context], dim=2)
        change, attention = self.shared(inp).unbind(2)
        vote = (change * torch.softmax(attention, dim=1)).sum(1)
        if self.training:
            x = x * (torch.rand((*x.shape[:2], 1), device=x.device) > 0.2) / 0.8
        return 1.4 * torch.tanh((vote + self.direct(x.flatten(1))[:, 0]) / 1.4)

def fit(groups, tr, va, z, v, y, test, n, seed, fold):
    core, inner = train_test_split(np.arange(len(tr)), test_size=0.17, stratify=y[tr], random_state=3209 + fold)
    x = compress(groups, tr[core])
    yy = torch.as_tensor(y[tr], device='cpu', dtype=torch.float32)
    base = torch.as_tensor(z, device='cpu', dtype=torch.float32)
    torch.manual_seed(27881 + fold)
    rng = np.random.default_rng(27881 + fold)
    net = Changes(x.shape[1]).to('cpu')
    opt = torch.optim.AdamW(net.parameters(), lr=0.00055, weight_decay=0.12)

    def evaluate(rows, score):
        net.eval()
        with torch.no_grad():
            val = torch.as_tensor(score.copy(), device='cpu', dtype=torch.float32)
            chunks = np.array_split(np.arange(len(rows)), max(1, (len(rows) + 511) // 512))
            return np.concatenate([net(x[rows[j]], val[j]).cpu().numpy() for j in chunks])
    best = roc_auc_score(y[tr[inner]], z[inner])
    epoch_best = -1
    weights = None
    for ep in range(22):
        net.train()
        for j in np.array_split(rng.permutation(core), max(1, len(core) // 256)):
            opt.zero_grad(set_to_none=True)
            edit = net(x[tr[j]], base[j])
            score = base[j] + edit
            pos = torch.as_tensor(np.flatnonzero(y[tr[j]] == 1), device='cpu')
            neg = torch.as_tensor(np.flatnonzero(y[tr[j]] == 0), device='cpu')
            gap = score[neg][None, :] - score[pos][:, None]
            local = (base[j][neg][None, :] - base[j][pos][:, None]).abs()
            pairs = torch.exp(-0.5 * (local / 0.8) ** 2)
            rank = (F.softplus(gap / 0.55) * pairs).sum() / pairs.sum().clamp_min(0.1)
            loss = 0.56 * F.binary_cross_entropy_with_logits(score, yy[j]) + 0.44 * 0.55 * rank + 0.1 * edit.square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.5)
            opt.step()
        correction = evaluate(tr[inner], z[inner])
        quality = roc_auc_score(y[tr[inner]], z[inner] + 0.55 * correction)
        if quality > best + 2.5e-05:
            best = quality
            epoch_best = ep
            weights = {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}
        if ep in (6, 13):
            for g in opt.param_groups:
                g['lr'] *= 0.5
        if ep - max(epoch_best, 2) >= 7:
            break
    if weights is None:
        return (np.zeros(len(va)), np.zeros(len(test)), -1)
    net.load_state_dict(weights)
    return (0.55 * evaluate(va, v), 0.55 * evaluate(n + np.arange(len(test)), test), epoch_best)

def main():
    q, ids, tids, y, wt, wn, white = wl.load()
    other = cohorts(ids, y)[0] & ~white
    groups, files = inputs(ids, tids)
    n = len(y)
    cache = OUT / ('frozen_' + CURRENT)
    dst = OUT / (TAG + '_models')
    dst.mkdir(exist_ok=True)
    parent = ROOT / f'new weights/wn_{CURRENT}_candidate'
    sha = wl.ef.sha(parent / 'weights.json')
    with np.load(cache / 'full.npz') as a:
        assert np.array_equal(tids, a['test_ids']) and str(a['parent_sha']) == sha
        test = a['test']
    scores, audits, preds = ({}, {}, {})
    seeds = [*SEEDS, 61817] if len(sys.argv) == 1 else [int(v) for v in sys.argv[1:]]
    for seed in seeds:
        latest, new, locked = [np.empty(n) for _ in range(3)]
        if seed == 7:
            tests = []
        for f, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(ids, y)):
            with np.load(cache / f's{seed}_f{f}.npz') as a:
                assert np.array_equal(ids[tr], a['train_ids']) and np.array_equal(ids[va], a['validation_ids'])
                z, v = (a['oof_train'], a['valid'])
                locked[va] = a['locked']
                latest[va] = v
            path = dst / f's{seed}_f{f}.npz'
            if not path.exists():
                d, t, ep = fit(groups, tr, va, z, v, y, test, n, seed, f)
                np.savez_compressed(path, ids=ids[va].to_numpy(dtype=str), delta=d, test=t, epoch=ep, tids=tids.to_numpy(dtype=str), parent_sha=sha)
            with np.load(path) as a:
                assert np.array_equal(a['ids'], ids[va]) and np.array_equal(a['tids'], tids) and (str(a['parent_sha']) == sha)
                new[va] = v + a['delta']
                if seed == 7:
                    tests.append(a['test'])
                print('mechanism readout', seed, f, int(a['epoch']), flush=True)
        row = {}
        for k, p in (('locked', locked), ('previous', latest), ('new', new)):
            r = wl.score(y, p, wt, wn, white)
            rk = rankings(y, p)
            r['other_historical_auc'] = float(rk[other].mean() / sum(y == 0))
            r['other_historical_caught_20'] = int(sum(rk[other] >= 0.8 * sum(y == 0)))
            row[k] = r
        scores[seed] = row
        audits[seed] = audit(y, white, latest, new)
        preds[f's{seed}'] = new
        print(seed, row['new'], flush=True)
    ok = len(seeds) == 7 and all((r['new']['auc'] > r['previous']['auc'] and all((r['new'][k] >= r['locked'][k] for k in ('target_vs_neg', 'remaining'))) for r in scores.values()))
    report = dict(accepted=ok, scores=scores, audits=audits)
    (OUT / (TAG + '.json')).write_text(json.dumps(report, indent=2))
    np.savez_compressed(OUT / (TAG + '_oof.npz'), ids=ids.to_numpy(dtype=str), **preds)
    if not ok:
        return
    pack = ROOT / f'new weights/wn_{TAG}_candidate'
    pack.mkdir(exist_ok=True)
    h = np.column_stack(tests).astype(float)
    w = np.full(h.shape[1], 1 / h.shape[1])
    sub = wl.ef.write_submission(expit(test + h @ w), tids, pack / 'submission.csv')
    deps = [parent / 'weights.json', Path(__file__).resolve(), *files, *[dst / f's7_f{k}.npz' for k in range(5)]]
    rec = pack / 'weights.json'
    rec.write_text(json.dumps(dict(weights=w.tolist(), hashes={str(p.relative_to(ROOT)): wl.ef.sha(p) for p in deps}), indent=2))
    np.savez_compressed(pack / 'replay_inputs.npz', ids=tids.to_numpy(dtype=str), order=tids.get_indexer(sub.id), parent_logits=test, features=h)
    (pack / 'validation.json').write_text(json.dumps(dict(**report, recipe_sha256=wl.ef.sha(rec), submission_sha256=wl.ef.sha(pack / 'submission.csv')), indent=2))
    shutil.copyfile(parent / 'replay.py', pack / 'replay.py')
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        torch.set_num_threads(1)
        main()
