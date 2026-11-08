# CPU port of experiments/redesign/nested_rank.py, source sha256 1ee1360f503dfb1a060281695da938eef879689e7f9c5f43e8cde8d7bbfa59b4, with verified model saving. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV
"""SOURCE-RANK variant (custom objective: one joint score per source = WT-weighted margin of its views; 78% pairwise source ranking + 22% per-view BCE).
Crop/transform-augmented nested mechanism tree (2026-11-04). Like nested_tree.py (ns3 + Codex mechanism tables + nested white-law priors + jump-law
tables), but each training fold also gets transformed copies (views) of its own training series, each copy carrying ITS OWN view's features:
  ns3 view features   tune_out/var/mat_ns3_c7.npz V_<v> / VT_<v>
  mechanism tables    tune_out/weight_space/view_<v>/<table>_{train,test}.parquet (original ids)
  harmonic            tune_out/var/harm_{train,test}_<v>.parquet (ids + suffix <v>)
  nested white law    tune_out/weight_space/view_<v>/law_outer0/fold{k}_{train,test}.parquet (priors from fold k training labels only)
  jump laws           tune_out/weight_space/jump_law_<src><v>_{train,test}.parquet
Validation / test scored on every view; weighted logit TTA (orig 2, crop views --crop-w, rev/flip 1). Test = mean over the 5 fold trees.
Usage: python nested_crop.py --tag MECHC1_s0 --views c1 [--extra jump_law_pulse,jump_law_state] [--seed 0] [--crop-w 0.125]"""
import sys, time, gc, ctypes, warnings, pathlib, argparse; warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
try: ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x4000)
except Exception: pass
import numpy as np, pandas as pd, xgboost as xgb
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--views", default="c1"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--extra", default="jump_law_pulse,jump_law_state"); ap.add_argument("--crop-w", type=float, default=0.125); ap.add_argument("--fold-seed", type=int, default=0, help="base fold split (law_outer<seed> priors must exist)"); ap.add_argument("--n-splits", type=int, default=5); ap.add_argument("--law-dir", default=None); ap.add_argument("--gate-phi", type=float, default=None, help="train only on series whose before-segment phi1_ar1 exceeds this (label-free family gate)"); ap.add_argument("--mask", default="", help="feature groups to drop from every view, e.g. ns3,wnlaw"); ap.add_argument("--contribs", action="store_true", help="save boosters + OOF/test per-feature-group SHAP contributions (orig view)"); ap.add_argument("--pair-frac", type=float, default=0.78); ap.add_argument("--pairs-per-source", type=float, default=4.0); a = ap.parse_args(); LAWD = a.law_dir or f"law_outer{a.fold_seed}"
D = KU_ENV.root(); O = D / "tune_out"; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); import save_models as SM; WS = O / "weight_space"; t0 = time.time()
VIEWS = [v for v in a.views.split(",") if v]; EXTRA = [t for t in a.extra.split(",") if t]
M = np.load(O / "var/mat_ns3_c7.npz", allow_pickle=True); ids, tids = M["ids"], M["tid"]; y, w = M["y"], M["w"]; cols = list(M["cols"])
T = ["harmonic", "ar_law", "bounds", "curve", "noise_tests", "partition", "periodic", "phase_law", "trend", "innov_v2", "mixture_law"]


def table(t, part, view):
    idx = ids if part == "train" else tids
    if t.startswith("redesign:"): return pd.read_parquet(O / f"redesign/{t[9:]}_{part}{'' if view is None else '_' + view}.parquet").loc[idx]
    if view is None: return pd.read_parquet(WS / f"{t}_{part}.parquet").loc[idx]
    if t == "harmonic":
        suf = {"rev": "r", "flip": "f", "revflip": "rf"}.get(view, view); h = pd.read_parquet(O / f"var/harm_{part}_{view}.parquet"); h.index = [i[: -len(suf)] for i in h.index]; return h.loc[idx]
    if t.startswith("jump_law_"):
        f = WS / f"view_{view}/{t}_{part}.parquet"
        return pd.read_parquet(f if f.exists() else WS / f"{t}{view}_{part}.parquet").loc[idx]
    return pd.read_parquet(WS / f"view_{view}/{t}_{part}.parquet").loc[idx]


def base(view):
    """features without the fold-specific white-law block: (train frame, test frame)"""
    out = []
    for part in ("train", "test"):
        X = M["X"] if view is None and part == "train" else M["Xt"] if view is None else M[f"V_{view}"] if part == "train" else M[f"VT_{view}"]
        parts = ([] if "ns3" in a.mask.split(",") else [pd.DataFrame(X, index=ids if part == "train" else tids, columns=cols)]) + [table(t, part, view).add_prefix(f"{t}__") for t in T + EXTRA if t not in a.mask.split(",")]
        out.append(pd.concat(parts, axis=1))
    return out


def law(view, k):
    p = WS / (LAWD if view is None else f"view_{view}/{LAWD}")
    if "wnlaw" in a.mask.split(","): return [pd.DataFrame(index=ids), pd.DataFrame(index=tids)]
    return [pd.read_parquet(p / f"fold{k}_{part}.parquet").loc[ids if part == "train" else tids].add_prefix("wnlaw__") for part in ("train", "test")]


ALL = [None] + VIEWS; B = {v: base(v) for v in ALL}
GPHI = pd.read_parquet(O / "generator_segments.parquet").query("part == 'pre'").set_index("id").loc[ids].phi1_ar1.to_numpy()
for v in VIEWS: assert list(B[v][0].columns) == list(B[None][0].columns), f"column mismatch in view {v}"
WT = np.array([2.0] + [a.crop_w if v.startswith("c") else 1.0 for v in VIEWS])
P = {"objective": "binary:logistic", "device": "cpu", "tree_method": "hist", "nthread": KU_ENV.threads(), "seed": a.seed, "grow_policy": "lossguide", "max_depth": 0, "verbosity": 0,
     "max_bin": 64, "learning_rate": 0.009549022390375738, "subsample": 0.8624554592415571, "colsample_bytree": 0.3282007743260973, "colsample_bynode": 0.7222561808470165,
     "reg_lambda": 0.2992087055688533, "reg_alpha": 0.01697153710754649, "gamma": 2.349446308922097, "min_child_weight": 0.4623536055943267, "max_leaves": 73}
ver = np.zeros((len(ALL), len(y))); vert = np.zeros((len(ALL), len(tids)))
FN = list(pd.concat([B[None][0].iloc[:1], law(None, 0)[0].iloc[:1]], axis=1).columns) if False else None
print(f"{a.tag}: views {VIEWS}, extra {EXTRA}, {B[None][0].shape[1] + (0 if "wnlaw" in a.mask else 36)} cols ({time.time()-t0:.0f}s)", flush=True)
for k, (tr, va) in enumerate(StratifiedKFold(a.n_splits, shuffle=True, random_state=a.fold_seed).split(ids, y)):
    rows = np.load(WS / f"{LAWD}/fold{k}_rows.npz", allow_pickle=True); assert set(rows["validation_ids"].astype(str)) == set(ids[va].astype(str))
    if a.gate_phi is not None: tr = tr[GPHI[tr] > a.gate_phi]
    F = {v: [pd.concat([B[v][j], law(v, k)[j]], axis=1).to_numpy(np.float32) for j in (0, 1)] for v in ALL}
    Xtr = np.vstack([F[v][0][tr] for v in ALL]); ytr = np.tile(y[tr], len(ALL))
    d = xgb.QuantileDMatrix(Xtr, ytr, max_bin=64); del Xtr
    nv, nt = len(ALL), len(tr); Av = (WT / WT.sum())[:, None]; yt = y[tr].astype(float); pos, neg = np.where(yt == 1)[0], np.where(yt == 0)[0]; rng = np.random.default_rng(1000 * k + a.seed)
    NP = int(a.pairs_per_source * nt); sg = lambda u: 1 / (1 + np.exp(-u))
    def obj(predt, dtrain):
        Mg = predt.reshape(nv, nt); s = (Av * Mg).sum(0)
        i, j = pos[rng.integers(len(pos), size=NP)], neg[rng.integers(len(neg), size=NP)]; dd = s[i] - s[j]; q = sg(-dd); hq = q * (1 - q)
        G = np.bincount(i, -q, nt) + np.bincount(j, q, nt); H = np.bincount(i, hq, nt) + np.bincount(j, hq, nt); c = 2 * NP / nt; G, H = G / c, H / c
        pv = sg(Mg); gv, hv = pv - yt[None, :], pv * (1 - pv)
        g = a.pair_frac * Av * G[None, :] + (1 - a.pair_frac) * gv; h = a.pair_frac * Av * H[None, :] + (1 - a.pair_frac) * hv
        return g.ravel(), np.maximum(h.ravel(), 1e-6)
    PR = {kk: vv for kk, vv in P.items() if kk != "objective"}; PR.update({"base_score": 0.5, "disable_default_eval_metric": 1})
    b = xgb.train(PR, d, int(__import__("os").environ.get("KU_SBB_TREE_ROUNDS", "1600")), obj=obj)
    SM.features(O, a.tag, pd.concat([B[None][0].iloc[:1], law(None, k)[0].iloc[:1]], axis=1).columns, views=ALL[1:], view_weights=WT.tolist(), fold_seed=a.fold_seed); SM.save_xgb(O, a.tag, k, b, F[None][0][va], margin=True)
    for i, v in enumerate(ALL):
        ver[i, va] = expit(b.predict(xgb.QuantileDMatrix(F[v][0][va], ref=d, max_bin=64), output_margin=True)); vert[i] += expit(b.predict(xgb.QuantileDMatrix(F[v][1], ref=d, max_bin=64), output_margin=True)) / a.n_splits
    if a.contribs:
        fn = list(pd.concat([B[None][0], law(None, k)[0]], axis=1).columns); grp = np.array(["ns3" if c in cols else c.split("__")[0] for c in fn]); GN = list(dict.fromkeys(grp))
        b.save_model(str(O / f"redesign/boosters/{a.tag}_fold{k}.ubj")); (O / f"redesign/boosters/{a.tag}_features.json").write_text(__import__("json").dumps({"features": fn, "groups": grp.tolist()}))
        agg = lambda C: np.column_stack([C[:, :-1][:, grp == g].sum(1) for g in GN] + [C[:, -1]])
        if k == 0: CO, CT = np.zeros((len(y), len(GN) + 1), np.float32), np.zeros((len(tids), len(GN) + 1), np.float32)
        CO[va] = agg(b.predict(xgb.DMatrix(F[None][0][va]), pred_contribs=True)); CT += agg(b.predict(xgb.DMatrix(F[None][1]), pred_contribs=True)) / a.n_splits
    print(f"  fold {k}: orig {roc_auc_score(y[va], ver[0, va]):.4f}  ({time.time()-t0:.0f}s)", flush=True); del d, b, F; gc.collect()
lg = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6)); comb = lambda V: expit((WT[:, None] * lg(V)).sum(0) / WT.sum())
oof, test = comb(ver), comb(vert); rep = lambda s: f"{roc_auc_score(y, s):.5f} / {roc_auc_score(y, s, sample_weight=w):.5f}"
print(f"{a.tag}: orig-only {rep(ver[0])}   wtta {rep(oof)}   " + "  ".join(f"{v} alone {roc_auc_score(y, ver[i + 1]):.4f}" for i, v in enumerate(VIEWS)), flush=True)
np.save(O / f"oof_{a.tag}.npy", oof); np.save(O / f"test_{a.tag}.npy", test); np.savez(O / f"ver_{a.tag}.npz", oof=ver.astype(np.float32), test=vert.astype(np.float32), kinds=np.array(["orig"] + VIEWS))
if a.contribs: np.savez(O / f"redesign/contrib_{a.tag}.npz", oof=CO, test=CT, groups=np.array(GN + ["bias"]), ids=ids.astype(str), tids=tids.astype(str)); print("contribs groups", GN, flush=True)
