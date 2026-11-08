# CPU port; original experiments/weight_space/white_restart_density.py, SHA256 bc98c8ad2b82637fdb3b332644416aec6bdc040188bab68c52d5c93abb0eeb36
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Learn how genuine source starts and unbroken continuations differ at their first samples."""
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
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits
from white_block_kernel import phys
from partition_evidence import OUT, ROOT
T = 12

def measure(left, right):
    future = right[T:T + 40]
    assert len(future) == 40
    m = np.median(future)
    z = future - m
    dif = np.diff(future)
    s = max(1.4826 * np.median(abs(dif - np.median(dif))) / np.sqrt(2), 0.17 * dif.std(), 0.01)
    q = z / s
    prior = (left[-8:] - m) / s if left is not None else np.zeros(8)
    assert len(prior) == 8
    trend = [q[:k].mean() for k in (3, 8, 16)]
    c = np.r_[np.arcsinh(m), np.log(s), np.log1p(q.std()), np.mean(q[1:] * q[:-1]) / (np.mean(q * q) + 0.01), np.arcsinh(q[:5]), np.arcsinh(trend), np.arcsinh(np.quantile(q, [0.05, 0.2, 0.35, 0.65, 0.8, 0.95])), np.mean(np.cos(2.8 * (dif - np.median(dif)) / s)), np.arcsinh(np.mean(dif) / s)]
    return (c, np.arcsinh(prior), ((right[:T] - m) / s).clip(-140, 140))

def make(part):
    path = OUT / f'restart_records_{part}.npz'
    if not path.exists():
        with np.load(OUT / f'pulse_streams_{part}.npz') as a:
            ids, x, n, b = (a[k] for k in ('ids', 'separate', 'n', 'b'))
        with np.load(ROOT / f'tune_out/seq_cache_{part}.npz', allow_pickle=True) as a:
            assert np.array_equal(a['ids'].astype(str), ids)
            raw = a['x']
        rows = []
        for r, p, end, j in zip(raw, x, n, b):
            channels = []
            for v in (r[:end].astype(float), p[:end].astype(float)):
                a, d = (v[:j], v[j:])
                lo = (len(a) - T - 40) // 2
                hi = (len(d) - T - 40) // 2
                cuts = ((None, a), (a[:lo], a[lo:]), (d[:hi], d[hi:]), (a, d), (d[::-1], a[::-1]), (None, d[::-1]))
                channels.append([np.concatenate(measure(l, w)) for l, w in cuts])
            rows.append(channels)
        h = np.array(rows, np.float32)
        assert np.isfinite(h).all()
        np.savez_compressed(path, ids=ids, h=h)
    with np.load(path) as a:
        return (a['ids'], a['h'])

class Densities(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.mode = nn.Embedding(5, 8)
        self.context = nn.Sequential(nn.Linear(dim + 8 + 8, 64), nn.LayerNorm(64), nn.SiLU(), nn.Linear(64, 48), nn.Tanh())
        self.dynamics = nn.GRU(2, 48, batch_first=True)
        self.head = nn.Linear(48, 15)
        self.register_buffer('anchor', torch.tensor([-1.6, -0.65, 0.0, 0.65, 1.6]))

    def forward(self, events, mode):
        v = events.reshape(-1, events.shape[-1])
        y = v[:, -T:]
        past = v[:, -T - 8:-T]
        history = past if mode in (2, 3, 4) else torch.zeros_like(past)
        idx = torch.full((len(v),), mode, device=v.device, dtype=torch.long)
        initial = self.context(torch.cat([v[:, :-T - 8], history, self.mode(idx)], 1))
        z = torch.cat([history[:, -1:], torch.asinh(y[:, :-1])], 1)
        clock = (torch.arange(T, device=v.device) / T).expand_as(z)
        states, _ = self.dynamics(torch.stack([z, clock], 2), initial[None, :, :])
        w, m, s = self.head(states).chunk(3, 2)
        m = 3 * m + self.anchor
        s = s.clamp(-2.6, 3.0).exp()
        m = m + torch.sinh(z.clamp(-5, 5))[:, :, None] * torch.sigmoid(initial[:, :1])[:, :, None]
        ll = F.log_softmax(w, 2) - s.log() - 2.5 * torch.log1p(((y[:, :, None] - m) / s).square() / 4)
        return ll.logsumexp(2).reshape(*events.shape[:-1], T)

def chart(net, h, ix):
    output = []
    with torch.no_grad():
        for j in np.array_split(ix, max(1, len(ix) // 160)):
            event = h[j]
            ll = torch.stack([net(event, k) for k in range(5)], -1)
            cols = []
            for a, b in ((0, 1), (0, 2), (1, 2), (4, 3)):
                gap = (ll[..., a] - ll[..., b]).clamp(-15, 15).cumsum(3)
                for end in (0, 2, 5, 11):
                    g = gap[:, :, :, end]
                    boundary = g[:, :, 3]
                    interior = (g[:, :, 1] + g[:, :, 2]) / 2
                    cols.extend([boundary, g[:, :, 4], boundary - interior, boundary - g[:, :, 0]])
            output.append(torch.cat(cols, 1).cpu().numpy())
    return np.concatenate(output)

def run():
    a, b = (make('train'), make('test'))
    ids, tids = (a[0], b[0])
    n = len(ids)
    te = n + np.arange(len(tids))
    H = torch.as_tensor(np.concatenate([a[1], b[1]]), device='cpu')
    del a, b
    y = pd.read_csv(ROOT / 'data/train_labels.csv', dtype={'id': str}).set_index('id').loc[ids, 'label'].to_numpy().copy()
    yy = torch.tensor(y, dtype=torch.float32, device='cpu')
    aux = np.vstack([phys('train', ids), phys('test', tids)])
    pv = np.empty(n)
    pt = []
    dest = OUT / 'restart_density_models'
    dest.mkdir(exist_ok=True)
    for f, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(ids, y)):
        core, inner = train_test_split(tr, test_size=0.12, stratify=y[tr], random_state=511 + f)
        torch.manual_seed(736 + f)
        rng = np.random.default_rng(736 + f)
        net = Densities(H.shape[-1] - T - 8).to('cpu')
        opt = torch.optim.AdamW(net.parameters(), lr=0.0011, weight_decay=0.018)
        for ep in range(18):
            for j in np.array_split(rng.permutation(core), max(1, len(core) // 192)):
                v = H[j]
                k = 1 + ep % 2
                ll = [net(v[:, :, 0], 0).mean(), net(v[:, :, k], 1).mean(), net(v[:, :, k], 2).mean(), net(v[y[j] == 0, :, 3], 3).mean(), net(v[y[j] == 1, :, 3], 4).mean()]
                opt.zero_grad(set_to_none=True)
                (-torch.stack(ll).mean()).backward()
                nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                opt.step()
            if ep in (8, 14):
                for g in opt.param_groups:
                    g['lr'] *= 0.5
        torch.save(net.state_dict(), dest / f'density{f}.pt')
        net.eval()
        feat = chart(net, H, np.arange(n + len(tids)))
        xx = np.column_stack([aux, np.arcsinh(feat)]).astype(np.float32)
        mu = xx[core].mean(0)
        sd = xx[core].std(0).clip(0.08)
        x = torch.tensor(((xx - mu) / sd).clip(-7, 7), device='cpu')
        head = nn.Sequential(nn.Linear(x.shape[1], 144), nn.LayerNorm(144), nn.SiLU(), nn.Dropout(0.22), nn.Linear(144, 80), nn.LayerNorm(80), nn.SiLU(), nn.Dropout(0.18), nn.Linear(80, 1)).to('cpu')
        op = torch.optim.AdamW(head.parameters(), lr=0.00065, weight_decay=0.025)
        best, bep = (-1, 0)
        ck = dest / f'head{f}.pt'
        for ep in range(32):
            head.train()
            for j in np.array_split(rng.permutation(core), max(1, len(core) // 220)):
                op.zero_grad(set_to_none=True)
                F.binary_cross_entropy_with_logits(head(x[j])[:, 0], yy[j]).backward()
                op.step()
            head.eval()
            with torch.no_grad():
                sc = roc_auc_score(y[inner], head(x[inner])[:, 0].cpu().numpy())
            if sc > best + 5e-05:
                best, bep = (sc, ep)
                torch.save(head.state_dict(), ck)
            if ep in (9, 17):
                for g in op.param_groups:
                    g['lr'] *= 0.45
            if ep - bep >= 6:
                break
        head.load_state_dict(torch.load(ck, map_location='cpu', weights_only=True))
        head.eval()
        with torch.no_grad():
            pv[va] = head(x[va])[:, 0].sigmoid().cpu().numpy()
            pt.append(head(x[te])[:, 0].sigmoid().cpu().numpy())
        np.savez_compressed(dest / f'fold{f}.npz', validation_ids=ids[va], center=mu, scale=sd, evidence=feat[va], test_evidence=feat[te])
        print('first-sample density', f, bep, roc_auc_score(y[va], pv[va]), flush=True)
    np.savez_compressed(OUT / 'lawrestart.npz', ids=ids, tids=tids, oof=pv, test=np.mean(pt, 0))
    print('restart OOF', roc_auc_score(y, pv), flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        torch.set_num_threads(1)
        run()
