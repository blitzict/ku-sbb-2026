# CPU port; original experiments/weight_space/white_mixture_program.py, SHA256 731170c73a13aef3052925b020de3570ea91a135c47921538a6815bfb6699778
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Fit conditional odds for competing physical interventions and nuisance changes."""
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
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits
from partition_evidence import OUT, ROOT

def evidence_table(part):
    fields = {}
    context = []
    groups = []
    names = []

    def table(name):
        if name not in fields:
            fields[name] = pd.read_parquet(OUT / f'{name}_{part}.parquet')
            assert fields[name].index.equals(next(iter(fields.values())).index)
        return fields[name]

    def ctx(name, keys):
        context.append(table(name)[keys].to_numpy())

    def hypothesis(name, views):
        blocks = [table(source)[cols].to_numpy() for source, cols in views]
        groups.append(np.column_stack(blocks))
        names.append(name)
    views = ['jump_law_' + s for s in ('state', 'pulse', 'nullstate', 'prestate', 'smoothvol')]
    ctx('innov_v2', ['pre_white', 'pre_walk', 'pre_phi', 'shared_pre_sd', 'shared_mean_pre', 'start'])
    ctx('state', ['pre2', 'pre_ar1', 'pre_ar2', 'pre_cluster', 'prior_scale', 'walk', 'clustering'])
    ctx('curve', ['gate', 'curvature', 'slope'])
    ctx('pulse', ['gate', 'width', 'scale', 'dispersion'])
    ctx('phase_law', ['df_l', 'amp_l', 'scale_l'])
    ctx('periodic', ['pre_fraction', 'period', 'gate'])
    ctx('mixture_law', ['pre_bim', 'sep_pre', 'balance_pre'])
    ctx('jump_law_state', [f'pre{i}' for i in range(12)])
    ids = next(iter(fields.values())).index
    lengths = pd.read_parquet(ROOT / f'clean/features_{part}.parquet', columns=['n_pre', 'n_post']).loc[ids]
    context.append(np.log(lengths.to_numpy()))
    for kind in ('level', 'scale'):
        hypothesis(kind, [(v, [kind + str(i) for i in range(6)]) for v in views])
    hypothesis('shape', [(v, ['shape0', 'shape1', 'shape0_robust', 'shape1_robust']) for v in views] + [('mixture_law', ['d_bf', 's_bf', 'p_bf', 'all_bf']), ('ar_law', ['tail_bf'])])
    for k in (1, 2):
        hypothesis('memory' + str(k), [(f'memory_law{k}', ['jump0', 'jump1', 'jump2', 'jump3', 'jump4', 'independent_memory', 'fixed_memory_new_scale', 'mixed']), ('ar_law', ['memory_bf'])])
    hypothesis('trend', [('curve', ['obs_step', 'slope_step', 'global_linear', 'global_quad', 'local_quad', 'combo', 'mixture']), ('trend_population', ['level', 'aggregate'] + ['stop' + str(i) for i in range(3)] + ['start' + str(i) for i in range(3)]), ('trend', ['level_bf', 'hinge_bf', 'free_level_bf', 'scale_bf', 'shape_bf'])])
    hypothesis('oscillator', [('phase_law', ['amplitude_bf', 'phase_bf', 'reset_bf', 'shape_bf', 'scale_bf', 'level_bf']), ('periodic', ['phase_gain', 'freq_gain']), ('clock', ['restart', 'off', 'reverse', 'change'])])
    hypothesis('walk', [('walk_jump', ['scale', 'shape', 'both', 'step', 'total'] + [f'step{i}' for i in range(8)]), ('ar_law', ['walk_level', 'walk_scale', 'walk_jump']), ('pulse', ['step', 'step_given_pulse'])])
    hypothesis('volatility', [('state', ['vol_split_bf', 'vol_bf', 'ar_change']), ('vol_smoothing', ['sv3'] + [f'sv{x}' for x in range(33, 45)])])
    x = np.column_stack(groups).clip(-32, 240).astype(np.float32)
    context = np.arcsinh(np.column_stack(context)).astype(np.float32)
    assert np.isfinite(x).all() and np.isfinite(context).all()
    return (ids.to_numpy(dtype=str), x, context, [g.shape[1] for g in groups])

class PhysicalOdds(nn.Module):

    def __init__(self, n_ctx, sizes):
        super().__init__()
        self.sizes = sizes
        dim = sum(sizes)
        groups = len(sizes)
        self.ctx = nn.Sequential(nn.Linear(n_ctx, 48), nn.LayerNorm(48), nn.SiLU(), nn.Dropout(0.1), nn.Linear(48, 36), nn.SiLU())
        self.weights = nn.Linear(36, 2 * dim)
        self.family = nn.Linear(36, 2 * (groups + 1))
        self.modulate = nn.Linear(36, dim)
        self.slope = nn.Parameter(torch.full((dim,), -0.5))
        self.tail = nn.Parameter(torch.full((dim,), -1.6))
        self.nuisance = nn.Parameter(torch.full((dim,), -2.8))
        self.readout = nn.Sequential(nn.Linear(groups * 2 + 36, 48), nn.SiLU(), nn.Dropout(0.15), nn.Linear(48, 1))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)
        for m in (self.weights, self.family, self.modulate):
            nn.init.normal_(m.weight, std=0.013)
            nn.init.zeros_(m.bias)

    def forward(self, x, c):
        c = self.ctx(c)
        a, b = self.weights(c).chunk(2, 1)
        q = F.softplus(self.slope + 0.8 * torch.tanh(self.modulate(c)))
        val = q * (x.clamp(-3, 5) + F.softplus(self.tail) * torch.asinh(x - x.clamp(-3, 5)))
        nul = val * self.nuisance.sigmoid()
        pos = [torch.logsumexp(v + F.log_softmax(w, 1), 1) for v, w in zip(val.split(self.sizes, 1), a.split(self.sizes, 1))]
        neg = [torch.logsumexp(v + F.log_softmax(w, 1), 1) for v, w in zip(nul.split(self.sizes, 1), b.split(self.sizes, 1))]
        pos, neg = (torch.stack(pos, 1), torch.stack(neg, 1))
        fa, fb = self.family(c).chunk(2, 1)
        mix = lambda p, w: torch.logsumexp(F.pad(p, (1, 0)) + F.log_softmax(w, 1), 1)
        odds = mix(pos, fa) - mix(neg, fb)
        correction = self.readout(torch.cat([pos, neg, c], 1))[:, 0]
        return odds + correction

@torch.no_grad()
def predict(net, x, c, idx):
    net.eval()
    return torch.cat([net(x[j], c[j]).sigmoid() for j in np.array_split(idx, max(1, len(idx) // 1024))]).cpu().numpy()

def run():
    ids, x, c, sizes = evidence_table('train')
    tids, tx, tc, ts = evidence_table('test')
    assert sizes == ts
    n = len(ids)
    te = n + np.arange(len(tids))
    y = pd.read_csv(ROOT / 'data/train_labels.csv', dtype={'id': str}).set_index('id').loc[ids, 'label'].to_numpy()
    both = np.vstack([c, tc])
    x = torch.tensor(np.vstack([x, tx]), device='cpu')
    labels = torch.tensor(y, device='cpu', dtype=torch.float32)
    oof = np.empty(n)
    tests = []
    folder = OUT / 'conditional_physics_models'
    folder.mkdir(exist_ok=True)
    for f, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(ids, y)):
        core, inner = train_test_split(tr, test_size=0.12, stratify=y[tr], random_state=941 + f)
        m = both[core].mean(0)
        s = both[core].std(0).clip(0.06)
        c = torch.tensor(((both - m) / s).clip(-8, 8), device='cpu')
        torch.manual_seed(418 + f)
        r = np.random.default_rng(418 + f)
        net = PhysicalOdds(c.shape[1], sizes).to('cpu')
        opt = torch.optim.AdamW(net.parameters(), lr=0.002, weight_decay=0.016)
        best, bep = (-1, 0)
        ck = folder / f'fold{f}.pt'
        for ep in range(85):
            net.train()
            order = r.permutation(core)
            for j in np.array_split(order, max(1, len(core) // 320)):
                opt.zero_grad(set_to_none=True)
                logodds = net(x[j], c[j])
                loss = F.binary_cross_entropy_with_logits(logodds, labels[j])
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 3.0)
                opt.step()
            v = roc_auc_score(y[inner], predict(net, x, c, inner))
            if v > best + 4e-05:
                best, bep = (v, ep)
                torch.save(net.state_dict(), ck)
            if ep in (20, 44):
                for g in opt.param_groups:
                    g['lr'] *= 0.48
            if ep - bep >= 11:
                break
        net.load_state_dict(torch.load(ck, map_location='cpu', weights_only=True))
        oof[va] = predict(net, x, c, va)
        tests.append(predict(net, x, c, te))
        np.savez(folder / f'fold{f}_input.npz', center=m, scale=s, train_ids=ids[tr], validation_ids=ids[va])
        print('conditional program', f, bep, roc_auc_score(y[va], oof[va]), flush=True)
    np.savez_compressed(OUT / 'lawconditional.npz', ids=ids, tids=tids, oof=oof, test=np.mean(tests, axis=0))
    print('conditional program OOF', roc_auc_score(y, oof), flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        torch.set_num_threads(1)
        run()
