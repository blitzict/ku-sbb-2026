# CPU port; original experiments/weight_space/white_rank_likelihood.py, SHA256 89b0285775958bcde376626d218501c7b97e61e803e225c70cb641da9ca2c081
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Learn alternatives to exchangeable segment membership across pooled sample ranks."""
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
from scipy.special import ndtri
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits
from white_block_kernel import phys
from partition_evidence import OUT, ROOT
NB = 32
U = (np.arange(NB) + 0.5) / NB

def normalized(x):
    q = np.quantile(x, [0.25, 0.75])
    m = q.mean()
    return (x - m) / max((q[1] - q[0]) / 1.349, 0.17 * x.std(), 0.002)

def draws(x, j):
    u = rankdata(x)
    idx = ((u - 0.5) * NB / len(x)).astype(int).clip(0, NB - 1)
    block = np.r_[np.arange(j) * 4 // j, 4 + np.arange(len(x) - j) * 4 // (len(x) - j)]
    count = np.bincount(block * NB + idx, minlength=8 * NB).reshape(8, NB)
    quant = np.quantile(normalized(x), U).clip(-8, 8)
    return (count, quant)

def prepare(part):
    path = OUT / f'rank_likelihood_{part}.npz'
    if not path.exists():
        raw = []
        for src in ('pulse', 'state'):
            with np.load(OUT / f'{src}_streams_{part}.npz') as a:
                ids, x, n, b = (a[k] for k in ('ids', 'separate', 'n', 'b'))
                raw.append(x)
        counts = []
        q = []
        for v, e, end, j in zip(*raw, n, b):
            v = v[:end].astype(float)
            e = e[:end].astype(float)
            z = normalized(e)
            lag = np.r_[z[0], z[:-1]]
            sv = np.r_[normalized(e[:j]), normalized(e[j:])]
            xs = (v, e, sv, abs(z), z * lag, abs(z) * abs(lag))
            cc, qq = zip(*(draws(x, j) for x in xs))
            counts.append(cc)
            q.append(qq)
        np.savez_compressed(path, ids=ids, counts=np.asarray(counts, np.int16), q=np.asarray(q, np.float16))
    with np.load(path) as d:
        return (d['ids'], d['counts'], d['q'])

class RankLaws(nn.Module):

    def __init__(self, dim, streams):
        super().__init__()
        z = ndtri(U)
        base = np.stack([z, z * z - 1, np.tanh(3 * z), z * np.exp(-z * z), np.cos(2.5 * z), np.sin(3 * z), np.cos(4.1 * z), np.exp(-((z - 1.35) / 0.55) ** 2), np.exp(-((z + 1.35) / 0.55) ** 2)])
        self.K = len(base)
        base = base - base.mean(1, keepdims=True)
        base /= base.std(1, keepdims=True)
        self.register_buffer('base', torch.tensor(base, dtype=torch.float32))
        self.register_buffer('basis', torch.tensor(np.stack([np.cos(np.pi * j * U) for j in range(1, 10)]), dtype=torch.float32))
        self.register_buffer('size', torch.tensor([-0.95, -0.39, -0.14, 0.14, 0.39, 0.95]))
        self.context = nn.Sequential(nn.Linear(NB + 3, 48), nn.LayerNorm(48), nn.SiLU(), nn.Linear(48, 40), nn.SiLU())
        self.correction = nn.Linear(40, self.K * 9)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)
        self.weight = nn.Linear(40, self.K)
        self.effective = nn.Linear(40, 1)
        self.physics = nn.Sequential(nn.Linear(dim, 96), nn.LayerNorm(96), nn.SiLU(), nn.Dropout(0.15))
        self.head = nn.Sequential(nn.Linear(streams * (self.K * 3 + 3) + 96, 144), nn.LayerNorm(144), nn.SiLU(), nn.Dropout(0.21), nn.Linear(144, 64), nn.LayerNorm(64), nn.SiLU(), nn.Dropout(0.14), nn.Linear(64, 1))

    def forward(self, count, q, aux):
        n = count.sum(2)
        right = count[:, :, 4:].sum(2)
        N = n.sum(2)
        b = right.sum(2)
        frac = (b / N).clamp(0.05, 0.95)
        within = count / count.sum(3, keepdim=True).clamp_min(1)
        noise = (within[:, :, 1::2] - within[:, :, ::2]).square().mean((2, 3)) * N
        h = self.context(torch.cat([q, torch.stack([N.log() - 6.2, frac - 0.5, noise.log1p()], 2)], 2))
        delta = self.correction(h).reshape(*h.shape[:2], self.K, 9)
        t = self.base + 0.35 * (delta.tanh() @ self.basis)
        t = t - t.mean(3, keepdim=True)
        t = t / t.square().mean(3, keepdim=True).clamp_min(0.12).sqrt()
        law = t[:, :, :, None, :] * self.size[None, None, None, :, None]
        ne = n[:, :, None, None, :]
        alpha = torch.logit(frac)[:, :, None, None, None]
        c = alpha
        for _ in range(3):
            p = (c + law).sigmoid()
            c = c + ((b[:, :, None, None, None] - (p * ne).sum(4, keepdim=True)) / (ne * p * (1 - p)).sum(4, keepdim=True).clamp_min(0.01)).clamp(-2.0, 2.0)
        ll = (right[:, :, None, None, :] * law).sum(4) + b[:, :, None, None] * (c - alpha).squeeze(4)
        ll = ll - (ne * (F.softplus(law + c) - F.softplus(alpha))).sum(4)
        var = (ne * (c + law).sigmoid() * (1 - (c + law).sigmoid())).sum(4)
        nullvar = (N * frac * (1 - frac))[:, :, None, None]
        ll = ll + 0.5 * (var / nullvar).clamp_min(0.01).log()
        block = within @ t.transpose(2, 3)
        vv = (block[:, :, 1::2] - block[:, :, ::2]).square().mean(2) * N[:, :, None] / 16
        eff = (0.5 + F.softplus(self.effective(h))) / (1 + 0.36 * vv)
        fit = (ll * eff[:, :, :, None].clamp(0.12, 1.6)).clamp(-28, 60)
        weak = fit[:, :, :, 2:4].logsumexp(3) - np.log(2)
        broad = fit.logsumexp(3) - np.log(6)
        direction = fit[:, :, :, 3:].logsumexp(3) - fit[:, :, :, :3].logsumexp(3)
        mixture = (broad + F.log_softmax(self.weight(h), 2)).logsumexp(2)
        allf = torch.cat([torch.asinh(weak), torch.asinh(broad), torch.asinh(direction), torch.asinh(mixture)[:, :, None], torch.log1p(vv.mean(2))[:, :, None], noise.log1p()[:, :, None]], 2)
        return self.head(torch.cat([allf.flatten(1), self.physics(aux)], 1))[:, 0]

def run():
    a, b = (prepare('train'), prepare('test'))
    ids, tids = (a[0], b[0])
    n = len(ids)
    te = n + np.arange(len(tids))
    cnt, q = [torch.tensor(np.concatenate([a[i], b[i]]), device='cpu') for i in (1, 2)]
    del a, b
    y = pd.read_csv(ROOT / 'data/train_labels.csv', dtype={'id': str}).set_index('id').loc[ids, 'label'].to_numpy()
    yy = torch.as_tensor(y, dtype=torch.float32, device='cpu')
    aux = np.vstack([phys('train', ids), phys('test', tids)])
    pred = np.empty(n)
    test = []
    dest = OUT / 'rank_likelihood_models'
    dest.mkdir(exist_ok=True)
    for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(ids, y)):
        core, inner = train_test_split(tr, test_size=0.12, stratify=y[tr], random_state=582 + fold)
        torch.manual_seed(334 + fold)
        rng = np.random.default_rng(334 + fold)
        mu = aux[core].mean(0)
        sd = aux[core].std(0).clip(0.09)
        x = torch.as_tensor(((aux - mu) / sd).clip(-7, 7), dtype=torch.float32, device='cpu')
        net = RankLaws(x.shape[1], cnt.shape[1]).to('cpu')
        opt = torch.optim.AdamW(net.parameters(), lr=0.00063, weight_decay=0.032)
        best, bep = (-1, 0)
        ck = dest / f'fold{fold}.pt'

        def predict(rows):
            net.eval()
            with torch.no_grad():
                return np.concatenate([net(cnt[j].float(), q[j].float(), x[j]).sigmoid().cpu().numpy() for j in np.array_split(rows, max(1, len(rows) // 130))])
        for ep in range(32):
            net.train()
            for j in np.array_split(rng.permutation(core), max(1, len(core) // 160)):
                opt.zero_grad(set_to_none=True)
                logits = net(cnt[j].float(), q[j].float(), x[j])
                F.binary_cross_entropy_with_logits(logits, yy[j]).backward()
                nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                opt.step()
            score = roc_auc_score(y[inner], predict(inner))
            if score > best + 6e-05:
                best, bep = (score, ep)
                torch.save(net.state_dict(), ck)
            if ep in (9, 16):
                for group in opt.param_groups:
                    group['lr'] *= 0.5
            if ep - bep >= 6:
                break
        net.load_state_dict(torch.load(ck, map_location='cpu', weights_only=True))
        pred[va] = predict(va)
        test.append(predict(te))
        np.savez_compressed(dest / f'fold{fold}.npz', core_ids=ids[core], validation_ids=ids[va], center=mu, scale=sd)
        print('rank conditional likelihood', fold, bep, roc_auc_score(y[va], pred[va]), flush=True)
    np.savez_compressed(OUT / 'lawrankdensity.npz', ids=ids, tids=tids, oof=pred, test=np.mean(test, 0))
    print('rank conditional likelihood OOF', roc_auc_score(y, pred), flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        torch.set_num_threads(1)
        run()
