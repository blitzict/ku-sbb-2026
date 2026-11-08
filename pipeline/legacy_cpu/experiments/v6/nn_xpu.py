# CPU port of experiments/v6/nn_xpu.py, source sha256 abeb3ec764c92ce12bdfec7edd0449118ae556f8208314d22dc1d113d74a80fb. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV; import rank_gauss_bounded as RB
"""MLP stack member on the v6 feature table (CPU port; historical file name kept). Honest nested 5-fold CV:
per fold, a QuantileTransformer (rank-gauss) is fitted on the training rows only, the boundary-shift negatives of the
training-fold sources are added, and early stopping uses an inner 10% split of the training rows (never the validation fold).
Usage: .venv/Scripts/python nn_xpu.py --tag v6_mlp [--rd] [--seeds 3] [--epochs 60] [--hidden 512] [--drop 0.3]
Writes tune_out/oof_<tag>.npy, test_<tag>.npy, best_<tag>.json. RAM: float32 tables, ~0.5 GB."""
import sys, json, time, argparse, pathlib, warnings, ctypes, gc; warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8"); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
try: ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x4000)
except Exception: pass
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from featset import load, O
D_WEIGHTS = O.parent / "weights"
ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--rd", action="store_true")
ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--hidden", type=int, default=512)
ap.add_argument("--layers", type=int, default=2); ap.add_argument("--drop", type=float, default=0.3); ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=0.02); ap.add_argument("--bs", type=int, default=512); ap.add_argument("--patience", type=int, default=8)
ap.add_argument("--no-test", action="store_true"); ap.add_argument("--blocks", default="grid,grid2"); ap.add_argument("--no-aug", action="store_true"); ap.add_argument("--cols", default=None)
a = ap.parse_args(); t0 = time.time(); torch.set_num_threads(KU_ENV.threads())
dev = "cpu"
X, Xt, A, src, y, w, folds = load(blocks=tuple(b for b in a.blocks.split(",") if b), rd=a.rd, with_aug=not a.no_aug)
if a.cols:
    import pandas as pd; kc = [c for c in pd.read_csv(a.cols).feature if c in X.columns]; X, Xt, A = X[kc], Xt[kc], A[kc]
ids = X.index.to_numpy(); cols = list(X.columns)
Xn, Xtn, An = X.to_numpy(), Xt.to_numpy(), A.to_numpy(); del X, Xt, A; gc.collect()
print(f"device {dev}  features {Xn.shape[1]}  shift rows {len(An)}  ({time.time()-t0:.0f}s)", flush=True)


class MLP(nn.Module):
    def __init__(self, d, hidden, layers, drop):
        super().__init__(); L = []; k = d
        for _ in range(layers):
            L += [nn.Linear(k, hidden), nn.BatchNorm1d(hidden), nn.SiLU(), nn.Dropout(drop)]; k = hidden
        L += [nn.Linear(k, 1)]; self.net = nn.Sequential(*L)

    def forward(self, x): return self.net(x).squeeze(-1)


def transform(fit_rows, *others, seed=0):
    """Rank-gauss: per column, rank each value against the sorted training-fold column, map to a normal quantile.
    NaNs go to 0 (the median) after the transform.  Returns device tensors."""
    ref = torch.as_tensor(fit_rows, device=dev); ref = torch.where(torch.isnan(ref), torch.full_like(ref, float("inf")), ref)
    srt = torch.sort(ref, dim=0).values.T.contiguous()                                    # (C, N)
    n_valid = torch.isfinite(srt).sum(1, keepdim=True).clamp(min=1).float()               # (C, 1)
    out = []
    for arr in (fit_rows,) + others:
        Z = torch.as_tensor(arr, device=dev); nan = torch.isnan(Z); Zs = torch.where(nan, torch.zeros_like(Z), Z)
        lo = torch.searchsorted(srt, Zs.T.contiguous(), right=False).float(); hi = torch.searchsorted(srt, Zs.T.contiguous(), right=True).float()
        u = ((lo + hi) / 2 + 0.5) / (n_valid + 1); u = u.clamp(1e-4, 1 - 1e-4)
        G = (torch.erfinv(2 * u - 1) * 2 ** 0.5).T; G = torch.where(nan, torch.zeros_like(G), G); out.append(G.float().contiguous())
    return out


def predict(model, Z, bs=4096):
    model.eval(); out = []
    with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev != "cpu"):
        for i in range(0, len(Z), bs): out.append(model(torch.as_tensor(Z[i:i+bs], device=dev)).float().cpu().numpy())
    return np.concatenate(out)


def train_one(Ztr, ytr, Zin, yin, Zouts, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = MLP(Ztr.shape[1], a.hidden, a.layers, a.drop).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * ((len(Ztr) + a.bs - 1) // a.bs), pct_start=0.15)
    Zt_ = Ztr; yt_ = torch.as_tensor(ytr, dtype=torch.float32, device=dev); lossf = nn.BCEWithLogitsLoss()
    best, best_state, bad = -1, None, 0
    for ep in range(a.epochs):
        model.train(); perm = torch.randperm(len(Zt_), device=dev)
        for i in range(0, len(perm), a.bs):
            idx = perm[i:i+a.bs]; xb = Zt_[idx]; xb = xb + 0.05 * torch.randn_like(xb)          # light input jitter
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev != "cpu"): loss = lossf(model(xb).float(), yt_[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        auc = roc_auc_score(yin, predict(model, Zin))
        if auc > best: best, bad, best_state = auc, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience: break
    if best_state is not None:
        model.load_state_dict(best_state)
    if __import__("os").environ.get("SAVE_WEIGHTS", "1") != "0": (D_WEIGHTS / a.tag).mkdir(parents=True, exist_ok=True); torch.save(model.state_dict(), D_WEIGHTS / a.tag / f"seed{seed}.pt")
    return best, ep + 1, [predict(model, Z) for Z in Zouts]


def fit_fold(tr_rows, ytr_rows, outs, seed):
    rng = np.random.RandomState(seed); perm = rng.permutation(len(tr_rows)); n_in = len(perm) // 10
    inner, core = perm[:n_in], perm[n_in:]
    save_ref = __import__("os").environ.get("SAVE_WEIGHTS", "1") != "0"
    if save_ref: (D_WEIGHTS / a.tag).mkdir(parents=True, exist_ok=True)
    fit = tr_rows[core]; srt, nv = RB.sorted_reference(fit, D_WEIGHTS / a.tag / f"seed{seed}_rankgauss_ref.npz" if save_ref else None, dtype=fit.dtype)
    Z = [RB.transform(fit, srt, nv, out=torch.from_numpy(fit) if fit.dtype == np.float32 else None)] + [RB.transform(x, srt, nv) for x in (tr_rows[inner], *outs)]
    del fit, srt, nv; gc.collect()   # bounded memory, bit-identical to transform() below (auc_lab_20261107/claude/audit/test_rank_bounded.py)
    best, eps, preds = train_one(Z[0], ytr_rows[core], Z[1], ytr_rows[inner], Z[2:], seed)
    return best, eps, preds


oof = np.zeros(len(y)); test = np.zeros(len(Xtn)); log = []
for k, (tr, va) in enumerate(folds):
    sel = np.isin(src, ids[tr]); Xtr = np.concatenate([Xn[tr], An[sel]]); ytr = np.concatenate([y[tr], np.zeros(sel.sum())])
    pv = np.zeros(len(va)); pt = np.zeros(len(Xtn))
    for s in range(a.seeds):
        best, eps, (p_va, p_te) = fit_fold(Xtr, ytr, [Xn[va], Xtn], seed=100 * k + s)
        pv += p_va / a.seeds; pt += p_te / a.seeds; log.append((k, s, best, eps))
    oof[va] = pv; test += pt / len(folds)
    print(f"  fold {k} auc {roc_auc_score(y[va], pv):.4f}  inner-best {np.mean([l[2] for l in log if l[0]==k]):.4f}  epochs {[l[3] for l in log if l[0]==k]}  ({time.time()-t0:.0f}s)", flush=True)
    del Xtr; gc.collect()
cv, cvw = roc_auc_score(y, oof), roc_auc_score(y, oof, sample_weight=w)
print(f"{a.tag}: CV AUC {cv:.5f}  test-weighted {cvw:.5f}  ({time.time()-t0:.0f}s)", flush=True)
sig = lambda z: 1 / (1 + np.exp(-z))                       # save probabilities so the logistic stacker can take logits
np.save(O / f"oof_{a.tag}.npy", sig(oof))
if not a.no_test: np.save(O / f"test_{a.tag}.npy", sig(test))      # test = average of the 5 fold models x seeds (no full refit; MLP epochs come from inner split)
json.dump({"note": f"v6 nn_xpu rd={a.rd} seeds={a.seeds} hidden={a.hidden} layers={a.layers} drop={a.drop} device={dev}", "cv": float(cv),
           "cv_test_weighted": float(cvw), "n_features": int(Xn.shape[1]), "fold_log": log}, open(O / f"best_{a.tag}.json", "w"), indent=1)
