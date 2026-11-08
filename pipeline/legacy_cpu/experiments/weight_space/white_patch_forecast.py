# CPU port; original experiments/weight_space/white_patch_forecast.py, SHA256 a81b1b2090ad2aabfd887c999e29dc44e3e910667ec72950b1e37b99df262362
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Compare nearby and opposite-side forecasts for real withheld series patches."""
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
PATCHES = 12

def signal(v):
    med = np.median(v)
    scale = max(1.4826 * np.median(abs(v - med)), 0.2 * v.std(), 0.01)
    z = ((v - med) / scale).clip(-16, 16)
    lag = np.r_[z[0], z[:-1]]
    lag2 = np.r_[z[0], z[0], z[:-2]]
    return np.stack([z, np.abs(z), np.minimum(z * z, 20), np.tanh(1.6 * z), np.cos(2.7 * z), np.sin(2.7 * z), np.cos(4.5 * z), np.sin(4.5 * z), np.tanh(z) * np.tanh(lag), np.tanh(z) * np.tanh(lag2), np.tanh(abs(z) - 0.7) * np.tanh(abs(lag) - 0.7), np.log1p(abs(z - lag))], 1)

def make(part):
    path = OUT / f'forecast_patches_{part}.npz'
    if not path.exists():
        with np.load(OUT / f'pulse_streams_{part}.npz') as a:
            ids, x, n, b = (a[k] for k in ('ids', 'separate', 'n', 'b'))
        with np.load(OUT / f'state_streams_{part}.npz') as a:
            assert np.array_equal(a['ids'], ids)
            e = a['separate']
        allrows = []
        for row, innov, end, j in zip(x, e, n, b):
            raw = row[:end].astype(float)
            res = innov[:end].astype(float)
            ss = np.column_stack([signal(v) for v in (raw, res, np.r_[0.0, np.diff(raw)])])
            seg = np.r_[np.arange(j) * PATCHES // j, PATCHES + np.arange(end - j) * PATCHES // (end - j)]
            count = np.bincount(seg, minlength=2 * PATCHES)
            sums = np.zeros((2 * PATCHES, ss.shape[1]))
            np.add.at(sums, seg, ss)
            means = sums / count[:, None]
            patch = []
            for k in range(2 * PATCHES):
                sel = ss[seg == k][:, [0, 12, 24]]
                patch.append(np.r_[means[k], np.quantile(sel, [0.15, 0.35, 0.5, 0.65, 0.85], axis=0).ravel(), np.log(count[k])])
            allrows.append(np.stack(patch).reshape(2, PATCHES, -1))
        A = np.asarray(allrows, np.float16)
        assert np.isfinite(A).all()
        np.savez_compressed(path, ids=ids, patch=A)
    with np.load(path) as a:
        return (a['ids'], a['patch'])

class Predictor(nn.Module):

    def __init__(self, dim, aux):
        super().__init__()
        H = 64
        self.dim = dim
        self.embed = nn.Sequential(nn.Linear(dim, H), nn.LayerNorm(H), nn.SiLU(), nn.Linear(H, H))
        self.pos = nn.Linear(4, H)
        self.attn = nn.MultiheadAttention(H, 4, batch_first=True)
        self.norm = nn.LayerNorm(H)
        self.mlp = nn.Sequential(nn.Linear(H, H), nn.SiLU(), nn.Linear(H, H))
        self.dec = nn.Sequential(nn.Linear(H * 2, H), nn.SiLU(), nn.Linear(H, 2 * (dim - 1)))
        self.evidence = nn.Sequential(nn.Linear((dim - 1) * 5 + H * 3, 144), nn.LayerNorm(144), nn.SiLU(), nn.Dropout(0.18), nn.Linear(144, 64))
        self.p = nn.Sequential(nn.Linear(aux, 96), nn.LayerNorm(96), nn.SiLU(), nn.Dropout(0.15))
        self.classify = nn.Sequential(nn.Linear(64 + 96, 96), nn.LayerNorm(96), nn.SiLU(), nn.Dropout(0.18), nn.Linear(96, 1))
        pos = (torch.arange(PATCHES) + 0.5) / PATCHES
        self.register_buffer('location', torch.stack([pos - 0.5, torch.cos(np.pi * pos), torch.cos(2 * np.pi * pos), torch.sin(2 * np.pi * pos)], 1))

    def forward(self, x, aux):
        B = len(x)
        E = self.embed(x)
        position = self.pos(self.location)
        support = []
        target = []
        positions = []
        for cut in (0, 1):
            k = torch.arange(PATCHES, device=x.device)
            i = k // 2 % 2 == cut
            support.append((E[:, :, i] + position[i]).reshape(-1, PATCHES // 2, E.shape[-1]))
            target.append(x[:, :, ~i, :-1])
            positions.append(position[~i])
        keys = torch.cat(support, 0)
        h = self.norm(keys + self.attn(keys, keys, keys, need_weights=False)[0])
        h = h + self.mlp(h)
        pooled = h.mean(1).reshape(2, B, 2, -1)
        pos = torch.stack(positions)[:, None, None, :, :].expand(-1, B, 2, -1, -1).reshape(-1, PATCHES // 2, E.shape[-1])

        def forecast(h):
            local = self.attn(pos, h, h, need_weights=False)[0]
            mean, logsd = self.dec(torch.cat([local, h.mean(1, keepdim=True).expand_as(local)], 2)).chunk(2, 2)
            return (mean.reshape(2, B, 2, PATCHES // 2, -1), logsd.reshape(2, B, 2, PATCHES // 2, -1).clamp(-2.7, 2.3))
        obs = torch.stack(target)
        a, la = forecast(h)
        hr = h.reshape(2, B, 2, PATCHES // 2, -1).flip(2).reshape_as(h)
        c, lc = forecast(hr)

        def loss(mean, ls):
            return 2.5 * torch.log1p(((obs - mean) * (-ls).exp()).square() / 4) + ls
        own = loss(a, la)
        diff = loss(c, lc) - own
        side = diff.mean((0, 3))
        pr = pooled.mean(0)
        vec = torch.cat([side[:, 0], side[:, 1], torch.tanh(diff).mean((0, 2, 3)), own.mean((0, 2, 3)), diff.abs().mean((0, 2, 3)), pr[:, 0], pr[:, 1], abs(pr[:, 0] - pr[:, 1])], 1)
        return (self.classify(torch.cat([self.evidence(vec), self.p(aux)], 1))[:, 0], own.mean())

def run():
    a, b = (make('train'), make('test'))
    ids, tids = (a[0], b[0])
    n = len(ids)
    te = n + np.arange(len(tids))
    patch = np.concatenate([a[1], b[1]]).astype(np.float32)
    del a, b
    y = pd.read_csv(ROOT / 'data/train_labels.csv', dtype={'id': str}).set_index('id').loc[ids, 'label'].to_numpy()
    yy = torch.as_tensor(y, dtype=torch.float32, device='cpu')
    aux = np.vstack([phys('train', ids), phys('test', tids)])
    pred = np.empty(n)
    test = []
    dest = OUT / 'patch_forecast_models'
    dest.mkdir(exist_ok=True)
    for f, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(ids, y)):
        core, inner = train_test_split(tr, test_size=0.12, stratify=y[tr], random_state=816 + f)
        torch.manual_seed(1227 + f)
        rng = np.random.default_rng(1227 + f)
        mu = patch[core].mean((0, 1, 2))
        sd = patch[core].std((0, 1, 2)).clip(0.035)
        x = torch.as_tensor(((patch - mu) / sd).clip(-7, 7), device='cpu', dtype=torch.float32)
        m = aux[core].mean(0)
        s = aux[core].std(0).clip(0.08)
        p = torch.as_tensor(((aux - m) / s).clip(-7, 7), device='cpu', dtype=torch.float32)
        model = Predictor(x.shape[-1], p.shape[-1]).to('cpu')
        opt = torch.optim.AdamW(model.parameters(), lr=0.00075, weight_decay=0.025)

        def predict(rows):
            model.eval()
            with torch.no_grad():
                return np.concatenate([model(x[j], p[j])[0].sigmoid().cpu().numpy() for j in np.array_split(rows, max(1, len(rows) // 180))])
        best, bep = (-1, 0)
        ck = dest / f'fold{f}.pt'
        for ep in range(34):
            model.train()
            for j in np.array_split(rng.permutation(core), max(1, len(core) // 200)):
                opt.zero_grad(set_to_none=True)
                z, actual = model(x[j], p[j])
                loss = F.binary_cross_entropy_with_logits(z, yy[j])
                loss = 0.25 * loss + actual if ep < 3 else loss + 0.15 * actual
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
            score = roc_auc_score(y[inner], predict(inner))
            if ep >= 3 and score > best + 8e-05:
                best, bep = (score, ep)
                torch.save(model.state_dict(), ck)
            if ep in (10, 18):
                for group in opt.param_groups:
                    group['lr'] *= 0.52
            if ep - bep >= 7:
                break
        model.load_state_dict(torch.load(ck, map_location='cpu', weights_only=True))
        pred[va] = predict(va)
        test.append(predict(te))
        np.savez_compressed(dest / f'fold{f}.npz', validation_ids=ids[va], mu=mu, sd=sd, m=m, s=s)
        print('real patch forecasts', f, bep, roc_auc_score(y[va], pred[va]), flush=True)
    np.savez_compressed(OUT / 'lawpatchforecast.npz', ids=ids, tids=tids, oof=pred, test=np.mean(test, 0))
    print('real patch OOF', roc_auc_score(y, pred), flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        torch.set_num_threads(1)
        run()
