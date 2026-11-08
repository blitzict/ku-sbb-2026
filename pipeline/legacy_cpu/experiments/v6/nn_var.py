# CPU port of experiments/v6/nn_var.py, source sha256 709fb82e0650a3e20f44aad0b9e964ee5fb4ca0d3fe33fdba210abfe9de65218. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV; import rank_gauss_bounded as RB
"""MLP (CPU port) with reversal / flip augmentation + test-time averaging (user-approved 2026-11-02). Reads the matrices exported by
train_var.py --export (originals, shift negatives, rev/flip/revflip copies). Honest nested 5-fold CV on the canonical folds: per fold the
training set = original training rows + their shift negatives + (--aug) the 3 transformed copies of those rows; rank-gauss fitted on the
training rows only; early stopping on an inner 10% split of the original training rows (copies of inner rows are excluded from training).
'_tta' = mean prediction over the 4 versions of each validation / test series. Test = mean over the 5 fold models x seeds.
Usage: .venv/Scripts/python nn_var.py --tag VARN_mlp --mat tune_out/var/mat_cggd.npz [--aug] [--seeds 3]"""
import sys, json, time, argparse, pathlib, warnings, ctypes, gc; warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
try: ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x4000)
except Exception: pass
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
D = KU_ENV.root(); O = D / "tune_out"
ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--mat", required=True); ap.add_argument("--aug", action="store_true")
ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--hidden", type=int, default=512)
ap.add_argument("--layers", type=int, default=2); ap.add_argument("--drop", type=float, default=0.3); ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=0.02); ap.add_argument("--bs", type=int, default=1024); ap.add_argument("--patience", type=int, default=6); ap.add_argument("--seed-offset", type=int, default=0); ap.add_argument("--kinds", default="rev,flip,revflip"); ap.add_argument("--plr", type=int, default=0, help="PLR periodic embedding dim per feature (0 = off; Gorishniy et al. 2022)"); ap.add_argument("--plr-k", type=int, default=16); ap.add_argument("--plr-sigma", type=float, default=0.1); ap.add_argument("--pair-w", type=float, default=0.0, help="weight of an in-batch pairwise logistic (AUC surrogate) loss added to BCE"); ap.add_argument("--extra", action="store_true", help="use the extra real-segment rows (E, Ey, Esrc) stored in the matrix file"); ap.add_argument("--ema", type=float, default=0.0, help="weight-space averaging: EMA decay of the weights (0 = off), evaluated and kept instead of the raw weights")
a = ap.parse_args(); t0 = time.time(); torch.set_num_threads(KU_ENV.threads()); dev = "cpu"
M = np.load(a.mat, allow_pickle=True); Xn, Xtn, An, y, w, ids, src = (M[k] for k in ("X", "Xt", "A", "y", "w", "ids", "src"))
KINDS = tuple(a.kinds.split(",")); NV = 1 + len(KINDS); VART = {k: M[f"VT_{k}"] for k in KINDS}
print(f"device {dev}  features {Xn.shape[1]}  aug={a.aug}  ({time.time()-t0:.0f}s)", flush=True)


class PLR(nn.Module):
    """Periodic(x) = [sin 2pi c x, cos 2pi c x], c ~ N(0, sigma) learnable per feature; then per-feature Linear(2k -> e) + ReLU."""
    def __init__(self, d):
        super().__init__(); self.c = nn.Parameter(torch.randn(d, a.plr_k) * a.plr_sigma)
        self.W = nn.Parameter(torch.randn(d, 2 * a.plr_k, a.plr) / (2 * a.plr_k) ** 0.5); self.b = nn.Parameter(torch.zeros(d, a.plr))
    def forward(self, x):
        v = 2 * torch.pi * x[..., None] * self.c; v = torch.cat([torch.sin(v), torch.cos(v)], -1)
        return torch.relu(torch.einsum("nfk,fke->nfe", v, self.W) + self.b).flatten(1)


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__(); L = [PLR(d)] if a.plr else []; k = d * a.plr if a.plr else d
        for _ in range(a.layers): L += [nn.Linear(k, a.hidden), nn.BatchNorm1d(a.hidden), nn.SiLU(), nn.Dropout(a.drop)]; k = a.hidden
        self.net = nn.Sequential(*L, nn.Linear(k, 1))

    def forward(self, x): return self.net(x).squeeze(-1)


def rankgauss(fit_rows, outs, ref_path=None):
    """Per column: mid-rank against the sorted training column -> normal quantile. NaN -> 0. Returns tensors.
    Bounded memory: column-block sort, row-chunk transform, training rows transformed in place; bit-identical to the whole-matrix
    code kept below, and writes the same KU_ENV.rank_reference array (auc_lab_20261107/claude/audit/test_rank_bounded.py)."""
    srt, nv = RB.sorted_reference(fit_rows, ref_path)
    res = [RB.transform(fit_rows, srt, nv, out=torch.from_numpy(fit_rows))]
    while outs: res.append(RB.transform(outs.pop(0), srt, nv))
    return res


def rankgauss_whole_matrix(fit_rows, outs):
    """Original whole-matrix version, kept for reference only."""
    ref = torch.as_tensor(fit_rows, device=dev); ref = torch.where(torch.isnan(ref), torch.full_like(ref, float("inf")), ref)
    srt = torch.sort(ref, 0).values.T.contiguous(); nv = torch.isfinite(srt).sum(1, keepdim=True).clamp(min=1).float(); del ref; res = []
    for arr in [fit_rows] + outs:
        G = torch.empty(arr.shape, device=dev)
        for i in range(0, len(arr), 20000):
            Z = torch.as_tensor(arr[i:i + 20000], device=dev); nan = torch.isnan(Z); Zs = torch.where(nan, torch.zeros_like(Z), Z).T.contiguous()
            u = ((torch.searchsorted(srt, Zs).float() + torch.searchsorted(srt, Zs, right=True).float()) / 2 + 0.5) / (nv + 1)
            g = (torch.erfinv(2 * u.clamp(1e-4, 1 - 1e-4) - 1) * 2 ** 0.5).T; G[i:i + 20000] = torch.where(nan, torch.zeros_like(g), g)
        res.append(G)
    return res


def predict(model, Z):
    model.eval(); out = []
    with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev != "cpu"):
        for i in range(0, len(Z), 8192): out.append(model(Z[i:i + 8192]).float().cpu().numpy())
    return np.concatenate(out)


def train(Ztr, ytr, Zin, yin, seed):
    torch.manual_seed(seed); model = MLP(Ztr.shape[1]).to(dev); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * ((len(Ztr) + a.bs - 1) // a.bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.15)
    yt = torch.as_tensor(ytr, dtype=torch.float32, device=dev); lossf = nn.BCEWithLogitsLoss(); best, state, bad = -1, None, 0
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(a.ema), use_buffers=True) if a.ema > 0 else None
    for ep in range(a.epochs):
        model.train(); perm = torch.randperm(len(Ztr), device=dev)
        for i in range(0, len(perm), a.bs):
            idx = perm[i:i + a.bs]; xb = Ztr[idx] + 0.05 * torch.randn(len(idx), Ztr.shape[1], device=dev)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev != "cpu"): out = model(xb).float()
            loss = lossf(out, yt[idx])
            if a.pair_w > 0:
                yb = yt[idx]; sp, sn = out[yb > 0.5], out[yb < 0.5]
                if len(sp) and len(sn):
                    m = min(len(sp), len(sn)); sp = sp[torch.randperm(len(sp), device=dev)[:m]]; sn = sn[torch.randperm(len(sn), device=dev)[:m]]
                    d = sp[:, None] - sn[None, :]; loss = loss + a.pair_w * torch.nn.functional.softplus(-d).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            if ema is not None: ema.update_parameters(model)
        em = ema.module if ema is not None else model
        auc = roc_auc_score(yin, predict(em, Zin))
        if auc > best: best, bad, state = auc, 0, {k: v.detach().clone() for k, v in em.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience: break
    if state is None: state = {k: v.detach().clone() for k, v in em.state_dict().items()}     # Inner AUC undefined on a tiny fold.
    model.load_state_dict(state); return model, best, ep + 1


folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(Xn, y)); oof = np.zeros(len(y)); tta = np.zeros(len(y))
tp = np.zeros(len(Xtn)); ttp = np.zeros(len(Xtn)); ver = np.zeros((NV, len(y))); vert = np.zeros((NV, len(Xtn)))
for k, (tr, va) in enumerate(folds):
    rng = np.random.RandomState(k + 100 * a.seed_offset); perm = rng.permutation(tr); inner, core = perm[:len(tr) // 10], perm[len(tr) // 10:]
    VAR = {k: M[f"V_{k}"] for k in KINDS}
    sel = np.isin(src, ids[core]); parts = [(Xn, core), (An, sel)]; ys = [y[core], np.zeros(sel.sum())]
    if a.aug:
        for kd in KINDS: parts.append((VAR[kd], core)); ys.append(y[core])
    if a.extra: se = np.isin(M["Esrc"], ids[core]) | ~np.isin(M["Esrc"], ids); parts.append((M["E"], se)); ys.append(M["Ey"][se])
    Xtr = RB.stack(parts); ytr = np.concatenate(ys); del parts; gc.collect()   # == np.vstack(parts).astype(np.float32), no part copies
    outs = [Xn[inner], Xn[va]] + [VAR[kd][va] for kd in KINDS] + [Xtn] + [VART[kd] for kd in KINDS]; del VAR; gc.collect()
    WD = D / 'weights' / a.tag
    save_ref = __import__('os').environ.get('SAVE_WEIGHTS', '1') != '0'
    if save_ref: WD.mkdir(parents=True, exist_ok=True)
    Z = rankgauss(Xtr, outs, WD / f'fold{k}_rankgauss_ref.npz' if save_ref else None); del Xtr, outs; gc.collect(); Ztr, Zin, Zv = Z[0], Z[1], Z[2:2 + NV]; Zt = Z[2 + NV:2 + 2 * NV]
    pv = np.zeros((NV, len(va))); pt = np.zeros((NV, len(Xtn))); eps = []
    for s in range(a.seeds):
        model, best, ep = train(Ztr, ytr, Zin, y[inner], 1000 * k + s + 100 * a.seed_offset); eps.append(ep)
        WD = D / 'weights' / a.tag
        if __import__('os').environ.get('SAVE_WEIGHTS', '1') != '0': WD.mkdir(parents=True, exist_ok=True); torch.save(model.state_dict(), WD / f'fold{k}_seed{s}.pt')
        for j in range(NV): pv[j] += predict(model, Zv[j]) / a.seeds; pt[j] += predict(model, Zt[j]) / a.seeds
        del model
    oof[va] = pv[0]; tta[va] = pv.mean(0); tp += pt[0] / 5; ttp += pt.mean(0) / 5; ver[:, va] = pv; vert += pt / 5
    del Z, Ztr, Zin, Zv, Zt
    print(f"  fold {k}: {roc_auc_score(y[va], oof[va]):.4f}  tta {roc_auc_score(y[va], tta[va]):.4f}  epochs {eps}  ({time.time()-t0:.0f}s)", flush=True)
sig = lambda z: 1 / (1 + np.exp(-z)); rep = lambda s: f"{roc_auc_score(y, s):.5f} / {roc_auc_score(y, s, sample_weight=w):.5f}"
print(f"{a.tag}: plain {rep(oof)}   tta {rep(tta)}", flush=True)
for t, o, te in ((a.tag, oof, tp), (a.tag + "_tta", tta, ttp)):
    np.save(O / f"oof_{t}.npy", sig(o)); np.save(O / f"test_{t}.npy", sig(te))
    json.dump({"note": f"nn_var aug={a.aug} mat={a.mat} seeds={a.seeds} kinds={a.kinds} plr={a.plr} ema={a.ema} pair_w={a.pair_w} hidden={a.hidden} layers={a.layers} drop={a.drop} device={dev}",
               "cv": float(roc_auc_score(y, o)), "cv_test_weighted": float(roc_auc_score(y, o, sample_weight=w))}, open(O / f"best_{t}.json", "w"), indent=1)
np.savez(O / f"ver_{a.tag}.npz", oof=sig(ver).astype(np.float32), test=sig(vert).astype(np.float32), kinds=np.array(["orig"] + list(KINDS)))
print(f"done {time.time()-t0:.0f}s", flush=True)
