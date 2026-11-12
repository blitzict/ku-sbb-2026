# CPU port of experiments/redesign/nested_cat.py, source sha256 3492c9715f6a4570eab3c53893e65c6c00666087e3fa2cfcf6f662f65e64d692, with verified model saving. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV
"""CATBOOST variant (symmetric trees, CPU port, early stopping on an inner 10% of the training fold's SOURCES, orig rows of those sources as eval set).
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
ap.add_argument("--extra", default="jump_law_pulse,jump_law_state"); ap.add_argument("--crop-w", type=float, default=0.125); ap.add_argument("--fold-seed", type=int, default=0, help="base fold split (law_outer<seed> priors must exist)"); ap.add_argument("--n-splits", type=int, default=5); ap.add_argument("--law-dir", default=None); a = ap.parse_args(); LAWD = a.law_dir or f"law_outer{a.fold_seed}"
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
        parts = [pd.DataFrame(X, index=ids if part == "train" else tids, columns=cols)] + [table(t, part, view).add_prefix(f"{t}__") for t in T + EXTRA]
        out.append(pd.concat(parts, axis=1))
    return out


def law(view, k):
    p = WS / (LAWD if view is None else f"view_{view}/{LAWD}")
    return [pd.read_parquet(p / f"fold{k}_{part}.parquet").loc[ids if part == "train" else tids].add_prefix("wnlaw__") for part in ("train", "test")]


ALL = [None] + VIEWS; B = {v: base(v) for v in ALL}
for v in VIEWS: assert list(B[v][0].columns) == list(B[None][0].columns), f"column mismatch in view {v}"
WT = np.array([2.0] + [a.crop_w if v.startswith("c") else 1.0 for v in VIEWS])
P = {"objective": "binary:logistic", "device": "cpu", "tree_method": "hist", "nthread": KU_ENV.threads(), "seed": a.seed, "grow_policy": "lossguide", "max_depth": 0, "verbosity": 0,
     "max_bin": 64, "learning_rate": 0.009549022390375738, "subsample": 0.8624554592415571, "colsample_bytree": 0.3282007743260973, "colsample_bynode": 0.7222561808470165,
     "reg_lambda": 0.2992087055688533, "reg_alpha": 0.01697153710754649, "gamma": 2.349446308922097, "min_child_weight": 0.4623536055943267, "max_leaves": 73}
ver = np.zeros((len(ALL), len(y))); vert = np.zeros((len(ALL), len(tids)))
print(f"{a.tag}: views {VIEWS}, extra {EXTRA}, {B[None][0].shape[1] + 36} cols ({time.time()-t0:.0f}s)", flush=True)
for k, (tr, va) in enumerate(StratifiedKFold(a.n_splits, shuffle=True, random_state=a.fold_seed).split(ids, y)):
    rows = np.load(WS / f"{LAWD}/fold{k}_rows.npz", allow_pickle=True); assert set(rows["validation_ids"].astype(str)) == set(ids[va].astype(str))
    F = {v: [pd.concat([B[v][j], law(v, k)[j]], axis=1).to_numpy(np.float32) for j in (0, 1)] for v in ALL}
    Xtr = np.vstack([F[v][0][tr] for v in ALL]); ytr = np.tile(y[tr], len(ALL))
    from catboost import CatBoostClassifier
    perm = np.random.RandomState(700 + k).permutation(tr); inner, core = perm[:len(tr) // 10], perm[len(tr) // 10:]
    Xc = np.vstack([F[v][0][core] for v in ALL]); yc = np.tile(y[core], len(ALL)); del Xtr
    m = CatBoostClassifier(depth=6, learning_rate=0.05, iterations=int(__import__("os").environ.get("KU_SBB_CAT_ITERS", "5000")), l2_leaf_reg=5, random_strength=1, border_count=64, loss_function="Logloss", eval_metric="AUC",
                           random_seed=a.seed, verbose=0, allow_writing_files=False, task_type="CPU", thread_count=KU_ENV.threads(), od_type="Iter", od_wait=300, use_best_model=True)
    m.fit(Xc, yc, eval_set=(F[None][0][inner], y[inner])); print(f"    best iteration {m.get_best_iteration()}", flush=True); del Xc
    SM.features(O, a.tag, pd.concat([B[None][0].iloc[:1], law(None, k)[0].iloc[:1]], axis=1).columns, views=ALL[1:], view_weights=WT.tolist()); SM.save_cat(O, a.tag, k, m, F[None][0][va])
    for i, v in enumerate(ALL):
        ver[i, va] = m.predict_proba(F[v][0][va])[:, 1]; vert[i] += m.predict_proba(F[v][1])[:, 1] / a.n_splits
    d = b = None
    print(f"  fold {k}: orig {roc_auc_score(y[va], ver[0, va]):.4f}  ({time.time()-t0:.0f}s)", flush=True); del d, b, F; gc.collect()
lg = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6)); comb = lambda V: expit((WT[:, None] * lg(V)).sum(0) / WT.sum())
oof, test = comb(ver), comb(vert); rep = lambda s: f"{roc_auc_score(y, s):.5f} / {roc_auc_score(y, s, sample_weight=w):.5f}"
print(f"{a.tag}: orig-only {rep(ver[0])}   wtta {rep(oof)}   " + "  ".join(f"{v} alone {roc_auc_score(y, ver[i + 1]):.4f}" for i, v in enumerate(VIEWS)), flush=True)
np.save(O / f"oof_{a.tag}.npy", oof); np.save(O / f"test_{a.tag}.npy", test); np.savez(O / f"ver_{a.tag}.npz", oof=ver.astype(np.float32), test=vert.astype(np.float32), kinds=np.array(["orig"] + VIEWS))
