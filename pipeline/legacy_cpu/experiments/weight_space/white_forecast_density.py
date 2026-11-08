# CPU port; original experiments/weight_space/white_forecast_density.py, SHA256 108959134b6d12743eec18e8caec3806855dab10055378e5e680add429c1b27f
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Predict genuine withheld observations under the dynamics of each side."""
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
from scipy.signal import lfilter
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits
from white_block_kernel import phys
from partition_evidence import OUT, ROOT

def context(obs, side, parity, source_center, source_scale):
    mask = side & parity
    valid = mask.copy()
    valid[:2] = False
    valid[2:] &= mask[1:-1] & mask[:-2]
    x = obs[valid]
    mu = np.median(x[:, 0])
    y = x[:, 0] - mu
    lag = x[:, 1:3] - mu
    robust = max(np.median(abs(y)) * 1.4826, 0.2 * y.std(), 0.001)
    X = lag.clip(-5 * robust, 5 * robust)
    z = y.clip(-5 * robust, 5 * robust)
    G = X.T @ X + np.eye(2) * max(0.001 * robust ** 2, 1e-06)
    v = X.T @ z
    phi = np.linalg.solve(G, v)
    phi[1] = phi[1].clip(-0.9, 0.9)
    phi[0] = (phi[0] / (1 - phi[1])).clip(-0.97, 0.97) * (1 - phi[1])
    choices = np.array([[0.0, 0.0], [(v[0] / G[0, 0]).clip(-0.97, 0.97), 0.0], phi, [1.0, 0.0]])
    loss = np.mean((z[:, None] - X @ choices.T) ** 2, 0).clip(1e-07)
    w = softmax(-0.5 * len(z) * np.log(loss) - np.array([0.0, 0.5, 1.0, 0.0]) * np.log(len(z)) + np.log([0.53, 0.29, 0.1, 0.08]))
    ar = w @ choices
    r = x[:, 0] - x[:, 1:3] @ ar
    inter = np.median(r)
    sd = max(np.median(abs(r - inter)) * 1.4826, 0.2 * r.std(), 0.002)
    e = (r - inter) / sd
    q = np.quantile(e, [0.01, 0.05, 0.15, 0.25, 0.75, 0.85, 0.95, 0.99]).clip(-12, 12)
    cu = (x[:, 1] - x[:, 2]) / sd
    return np.array([inter, *ar, sd, mu, np.log(len(z)), np.arcsinh(source_center + source_scale * mu), np.log(sd * source_scale), *w, *q, np.mean(np.abs(e)), np.mean(np.cos(2.8 * e)), np.mean(np.cos(4.3 * e)), np.mean(np.tanh(e) * np.tanh(cu)), np.mean(np.tanh(abs(e) - 0.7) * np.tanh(abs(cu) - 0.7))], np.float32)

def prepare(part):
    path = OUT / f'ar_predictive_{part}.npz'
    if not path.exists():
        samples = []
        contexts = []
        for tag in ('pulse', 'state'):
            with np.load(OUT / f'{tag}_streams_{part}.npz') as a:
                ids, x, n, b = (a[k] for k in ('ids', 'separate', 'n', 'b'))
            hist = np.zeros((*x.shape, 9), np.float16)
            C = []
            for i, (end, j) in enumerate(zip(n, b)):
                v = x[i, :end].astype(float)
                m = np.median(v)
                s = max(np.median(abs(v - m)) * 1.4826, 0.18 * v.std(), 0.015)
                z = (v - m) / s
                lag = lambda k: np.r_[np.repeat(z[0], k), z[:-k]]
                increments = np.r_[0.0, abs(np.diff(z))]
                smooth = [lfilter([1 - d], [1, -d], np.r_[increments[0], increments[:-1]]) for d in (0.5, 0.86, 0.967)]
                h = np.column_stack([z, lag(1), lag(2), lag(4), lag(8), lag(16), *smooth])
                sb = np.arange(end) >= j
                par = np.r_[np.arange(j) * 4 // j, np.arange(end - j) * 4 // (end - j)] % 2
                C.append(np.stack([context(h, sb == k, par == v, m, s) for k in (0, 1) for v in (0, 1)]))
                hist[i, :end] = h
            samples.append(hist)
            contexts.append(np.stack(C))
        np.savez(path, ids=ids, n=n, b=b, x=np.stack(samples, 1), c=np.stack(contexts, 1))
        print('observed forecast records', part, len(ids), flush=True)
    with np.load(path) as a:
        return [a[k] for k in ('ids', 'n', 'b', 'x', 'c')]

class Dynamics(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.encode = nn.Sequential(nn.Linear(dim, 48), nn.LayerNorm(48), nn.SiLU(), nn.Linear(48, 32), nn.SiLU())
        self.shape = nn.Linear(32, 16)
        self.hist = nn.Sequential(nn.Linear(7, 32), nn.SiLU(), nn.Linear(32, 16))
        self.couple = nn.Linear(32, 16)
        self.output = nn.Linear(16, 16)
        self.register_buffer('anchors', torch.tensor([0.0, -0.85, 0.85, -1.65, 1.65]))
        self.register_buffer('spread', torch.tensor([0.9, 0.55, 0.55, 1.55, 1.55]))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, c, ref):
        ctx = c[..., 5:].clone()
        ctx[..., 0] -= 4.8
        ctx = torch.cat([ctx, c[..., 1:3]], -1)
        gains = self.encode(ctx)
        ref = ref[:, None, :].expand(-1, x.shape[1], -1)
        gather = lambda z: z.gather(2, ref[:, :, :, None].expand(-1, -1, -1, z.shape[3]))
        dyn = gather(c[..., :5])
        inter, p1, p2, sd, mu = dyn.unbind(3)
        err = (x[:, :, :, 0] - inter - p1 * x[:, :, :, 1] - p2 * x[:, :, :, 2]) / sd
        lag = x[:, :, :, 1:6]
        hist = torch.cat([((lag[..., :1] - mu[:, :, :, None]) / sd[:, :, :, None]).clamp(-8, 8), ((lag[..., 1:4] - lag[..., :1]) / sd[:, :, :, None]).clamp(-8, 8), torch.log1p(x[:, :, :, 6:9] / sd[:, :, :, None]).clamp(0, 5)], -1)
        h = gather(self.shape(gains)) + self.hist(hist) * torch.sigmoid(gather(self.couple(gains)))
        delta = self.output(F.silu(h))
        m = self.anchors + 1.35 * torch.tanh(delta[..., :5])
        ss = self.spread * torch.exp(delta[..., 5:10].clamp(-2, 2))
        w = F.log_softmax(delta[..., 10:15], -1)
        r = (err[..., None] - m) / ss
        ll = -2.5 * torch.log1p(r.square() / 4) - ss.log() - 0.980829
        return torch.logsumexp(w + ll, -1) - sd.log()

def timing(n, b, t):
    side = (t >= b[:, None]).long()
    par = torch.where(side.bool(), 4 * (t - b[:, None]) // (n - b)[:, None], 4 * t // b[:, None]) % 2
    return (side, side * 2 + 1 - par, (1 - side) * 2 + 1 - par)

def chart(model, gpu, ix):
    x, c, n, b = gpu
    rows = []
    with torch.no_grad():
        for j in np.array_split(ix, max(1, len(ix) // 75)):
            nn0, bb = (n[j], b[j])
            T = int(nn0.max())
            h = x[j, :, :T].float()
            t = torch.arange(T, device='cpu')[None, :].expand(len(j), -1)
            side, own, other = timing(nn0, bb, t)
            truth = t < nn0[:, None]
            a = model(h, c[j], own)
            d = a - model(h, c[j], other)
            f = []
            for k in (0, 1):
                ok = (side == k) & truth & (t > 1)
                m = ok[:, None, :]
                N = m.sum(2)
                for v in (a, d, d.clamp(-2, 2), torch.tanh(d * 3), d.abs(), d.square().clamp(0, 30)):
                    z = (v * m).sum(2) / N
                    f.extend([z, z * N.sqrt(), (v * m).sum(2) / ((v * v * m).sum(2) + 0.1).sqrt()])
                middle = torch.where(side.bool(), (t - bb[:, None]) / (nn0 - bb)[:, None], (bb[:, None] - 1 - t) / bb[:, None])
                for edge in (0.125, 0.35, 0.65):
                    w = m & (middle[:, None, :] < edge)
                    f.append((d.clamp(-6, 6) * w).sum(2) / w.sum(2).clamp_min(1).sqrt())
            rows.append(torch.cat(f, 1).cpu().numpy())
    return np.vstack(rows)

def run():
    a, b = (prepare('train'), prepare('test'))
    ids, tids = (a[0], b[0])
    n = len(ids)
    te = n + np.arange(len(tids))
    Z, C, N, B = [torch.tensor(np.concatenate([a[k], b[k]]), device='cpu') for k in (3, 4, 1, 2)]
    gpu = (Z, C, N, B)
    del a, b
    y = pd.read_csv(ROOT / 'data/train_labels.csv', dtype={'id': str}).set_index('id').loc[ids, 'label'].to_numpy()
    yy = torch.tensor(y, device='cpu', dtype=torch.float32)
    aux = np.vstack([phys('train', ids), phys('test', tids)])
    pred = np.empty(n)
    test = []
    folder = OUT / 'forecast_density_models'
    folder.mkdir(exist_ok=True)
    for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(ids, y)):
        core, inner = train_test_split(tr, test_size=0.12, stratify=y[tr], random_state=485 + fold)
        rng = np.random.default_rng(624 + fold)
        torch.manual_seed(624 + fold)
        model = Dynamics(C.shape[-1] - 3).to('cpu')
        opt = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=0.015)
        for ep in range(15):
            model.train()
            for j in np.array_split(rng.permutation(core), max(1, len(core) // 160)):
                ns = N[j]
                bs = B[j]
                count = 48
                t = (2 + torch.rand(len(j), count, device='cpu') * (ns[:, None] - 2)).long()
                h = Z[j].gather(2, t[:, None, :, None].expand(-1, Z.shape[1], -1, Z.shape[3])).float()
                _, own, _ = timing(ns, bs, t)
                ll = model(h, C[j], own)
                opt.zero_grad(set_to_none=True)
                (-ll.mean()).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
            if ep in (6, 11):
                for group in opt.param_groups:
                    group['lr'] *= 0.5
        model.eval()
        torch.save(model.state_dict(), folder / f'forecast{fold}.pt')
        H = chart(model, gpu, np.arange(n + len(tids)))
        x = np.column_stack([aux, np.arcsinh(H)]).astype(np.float32)
        mu = x[core].mean(0)
        sd = x[core].std(0).clip(0.07)
        xx = torch.tensor(((x - mu) / sd).clip(-7, 7), device='cpu')
        head = nn.Sequential(nn.Linear(x.shape[1], 128), nn.LayerNorm(128), nn.SiLU(), nn.Dropout(0.21), nn.Linear(128, 80), nn.LayerNorm(80), nn.SiLU(), nn.Dropout(0.18), nn.Linear(80, 1)).to('cpu')
        op = torch.optim.AdamW(head.parameters(), lr=0.00072, weight_decay=0.025)
        best, bep = (-1, 0)
        ck = folder / f'head{fold}.pt'
        for ep in range(38):
            head.train()
            for j in np.array_split(rng.permutation(core), max(1, len(core) // 250)):
                op.zero_grad(set_to_none=True)
                F.binary_cross_entropy_with_logits(head(xx[j])[:, 0], yy[j]).backward()
                op.step()
            head.eval()
            with torch.no_grad():
                sc = roc_auc_score(y[inner], head(xx[inner])[:, 0].cpu().numpy())
            if sc > best + 4e-05:
                best, bep = (sc, ep)
                torch.save(head.state_dict(), ck)
            if ep in (9, 17):
                for group in op.param_groups:
                    group['lr'] *= 0.5
            if ep - bep >= 6:
                break
        head.load_state_dict(torch.load(ck, map_location='cpu', weights_only=True))
        head.eval()
        with torch.no_grad():
            pred[va] = head(xx[va])[:, 0].sigmoid().cpu().numpy()
            test.append(head(xx[te])[:, 0].sigmoid().cpu().numpy())
        np.savez_compressed(folder / f'fold{fold}.npz', validation_ids=ids[va], mu=mu, sd=sd, H=H[va], Ht=H[te])
        print('forecast likelihood', fold, bep, roc_auc_score(y[va], pred[va]), flush=True)
    np.savez_compressed(OUT / 'lawforecast.npz', ids=ids, tids=tids, oof=pred, test=np.mean(test, 0))
    print('forecast OOF', roc_auc_score(y, pred), flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        torch.set_num_threads(1)
        run()
