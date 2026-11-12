# CPU port of experiments/redesign/nested_mlp.py, source sha256 027ebf89d48712eab974c9b85a3b145d46b8363b22d4385a4cf5d8c78129db96, with verified model saving. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV
"""A different learner on the same rich physics table (2026-11-04, Codex request): MLP on ns3 + mechanism tables + extras + nested white-law priors
(original view), canonical StratifiedKFold(5, rs=0) with law_outer0 fold priors. Per fold: rank-Gauss fitted on the fold's training rows, early stopping
on an inner 10% of the training rows (never the validation fold), BCE + 0.5 in-batch pairwise loss, 3 seeds averaged. CPU port.
Writes tune_out/oof_MLPMECH.npy, tune_out/test_MLPMECH.npy."""
import sys, time, ctypes, warnings, pathlib; warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
try: ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x4000)
except Exception: pass
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
torch.set_num_threads(KU_ENV.threads()); dev = "cpu"; D = KU_ENV.root(); O = D / "tune_out"; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); import save_models as SM; WS = O / "weight_space"; t0 = time.time()
M = np.load(O / "var/mat_ns3_c7.npz", allow_pickle=True); ids, tids = M["ids"], M["tid"]; y = M["y"]; w = M["w"]; cols = list(M["cols"])
T = ["harmonic", "ar_law", "bounds", "curve", "noise_tests", "partition", "periodic", "phase_law", "trend", "innov_v2", "mixture_law",
     "jump_law_pulse", "jump_law_state", "jump_law_nullstate", "jump_law_prestate", "jump_law_smoothvol", "volatility_law"]
base = [pd.concat([pd.DataFrame(M["X"] if p == "train" else M["Xt"], index=ids if p == "train" else tids, columns=cols)] +
                  [pd.read_parquet(WS / f"{t}_{p}.parquet").loc[ids if p == "train" else tids].add_prefix(f"{t}__") for t in T], axis=1) for p in ("train", "test")]


def rankgauss(ref, *arrs):
    ref = torch.tensor(np.nan_to_num(ref, nan=0.0), device=dev); srt = torch.sort(ref, 0).values.T.contiguous(); n = ref.shape[0]; out = []
    for A in arrs:
        Z = torch.tensor(np.nan_to_num(A, nan=0.0), device=dev).T.contiguous()
        u = ((torch.searchsorted(srt, Z).float() + torch.searchsorted(srt, Z, right=True).float()) / 2 + 0.5) / (n + 1)
        out.append((torch.erfinv(2 * u.clamp(1e-4, 1 - 1e-4) - 1) * 2 ** 0.5).T.contiguous())
    return out


def net(d):
    return nn.Sequential(nn.Linear(d, 512), nn.BatchNorm1d(512), nn.SiLU(), nn.Dropout(0.3), nn.Linear(512, 256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(0.3), nn.Linear(256, 1)).to(dev)


def train(Xtr, ytr, Xin, yin, seed):
    torch.manual_seed(seed); m = net(Xtr.shape[1]); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.02); yt = torch.tensor(ytr, dtype=torch.float32, device=dev)
    E, bs = int(__import__("os").environ.get("KU_SBB_NN_EPOCHS", "40")), 512; sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=E * ((len(Xtr) + bs - 1) // bs), pct_start=0.15); best, state, bad = -1, None, 0
    for ep in range(E):
        m.train(); perm = torch.randperm(len(Xtr), device=dev)
        for i in range(0, len(perm), bs):
            j = perm[i:i + bs]; out = m(Xtr[j] + 0.05 * torch.randn_like(Xtr[j])).squeeze(-1); yb = yt[j]; loss = nn.functional.binary_cross_entropy_with_logits(out, yb)
            sp, sn = out[yb > .5], out[yb < .5]
            if len(sp) and len(sn): loss = loss + 0.5 * nn.functional.softplus(-(sp[:, None] - sn[None, :])).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        m.eval()
        with torch.no_grad(): auc = roc_auc_score(yin, m(Xin).squeeze(-1).cpu().numpy())
        if auc > best: best, bad, state = auc, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 6: break
    m.load_state_dict(state); m.eval(); return m


oof = np.zeros(len(y)); test = np.zeros(len(tids))
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(ids, y)):
    L = [pd.read_parquet(WS / f"law_outer0/fold{k}_{p}.parquet").loc[ids if p == "train" else tids].add_prefix("wnlaw__") for p in ("train", "test")]
    Xa, Xt = (pd.concat([base[j], L[j]], axis=1).to_numpy(np.float32) for j in (0, 1))
    perm = np.random.RandomState(k).permutation(tr); inner, core = perm[:len(tr) // 10], perm[len(tr) // 10:]
    Zc, Zi, Zv, Zt = rankgauss(Xa[core], Xa[core], Xa[inner], Xa[va], Xt)
    SM.features(O, "MLPMECH", pd.concat([base[0].iloc[:1], L[0].iloc[:1]], axis=1).columns); np.savez_compressed(SM.folder(O) / f"MLPMECH_fold{k}_rankgauss_ref.npz", sorted=np.sort(np.nan_to_num(Xa[core], nan=0.0), 0).astype(np.float32))
    pv, pt_ = np.zeros(len(va)), np.zeros(len(tids))
    for s in range(3):
        m = train(Zc, y[core], Zi, y[inner], 100 * k + s); torch.save(m.state_dict(), SM.folder(O) / f"MLPMECH_fold{k}_seed{s}.pt")
        with torch.no_grad(): pv += m(Zv).squeeze(-1).cpu().numpy() / 3; pt_ += m(Zt).squeeze(-1).cpu().numpy() / 3
    oof[va] = pv; test += pt_ / 5; print(f"fold {k}: {roc_auc_score(y[va], pv):.4f}  ({time.time()-t0:.0f}s)", flush=True)
sig = lambda z: 1 / (1 + np.exp(-z)); np.save(O / "oof_MLPMECH.npy", sig(oof)); np.save(O / "test_MLPMECH.npy", sig(test))
print(f"MLPMECH: {roc_auc_score(y, oof):.5f} / {roc_auc_score(y, oof, sample_weight=w):.5f}", flush=True)
