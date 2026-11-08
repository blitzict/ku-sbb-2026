# CPU port of experiments/v6/train_cv.py, source sha256 2d891200115f70c5ce43a462d503eefb18f6b9420466c9ef99dbfbdb29ff36a6. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV
"""Tier-1 experiments on top of v5 (clean + grid + grid2 + boundary-shift negatives), honest nested 5-fold CV.

Usage: python train_cv.py --tag <tag> --model xgb|lgb|cat [--rd] [--colsample 1.0] [--obj gce:0.5] [--refit-mult 1.15] [--no-test]
  --rd            add the ratio/difference block (3 columns per clean _pre/_post pair, built in memory, no parquet)
  --colsample c   override colsample_bytree/bynode (xgb) or feature_fraction (lgb)
  --obj gce:q     generalised cross-entropy robust loss for xgb (q in (0,1]); default log-loss
  --refit-mult m  multiply rounds for the full-data refit only (idea 6)
Memory: everything float32, one process, QuantileDMatrix on the host. CPU only.
Writes tune_out/oof_<tag>.npy, test_<tag>.npy, best_<tag>.json and prints plain + test-weighted CV AUC next to v5."""
import sys, json, time, pathlib, argparse, warnings, ctypes, gc; warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
try: ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x4000)   # BELOW_NORMAL
except Exception: pass
D = KU_ENV.root(); O = D / "tune_out"
ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--model", default="xgb")
ap.add_argument("--rd", action="store_true"); ap.add_argument("--colsample", type=float); ap.add_argument("--obj", default="logloss")
ap.add_argument("--refit-mult", type=float, default=1.0); ap.add_argument("--no-test", action="store_true")
ap.add_argument("--blocks", default="grid,grid2"); ap.add_argument("--contrast", action="store_true", help="add evidence-contrast columns (max/2nd/spread/argmax over per-type evidences)"); ap.add_argument("--mono", action="store_true", help="monotone (+1) constraints on evidence-magnitude columns"); ap.add_argument("--cols", default=None, help="csv with a feature column: keep only these columns"); ap.add_argument("--extra", default=None, help="json of extra xgb params to override"); ap.add_argument("--drop-train", default=None, help="csv of ids to drop from the TRAINING folds only (validation keeps them)"); ap.add_argument("--no-aug", action="store_true", help="train on the 24,000 labelled series only (no boundary-shift rows)"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--threads", type=int, default=KU_ENV.threads())
ap.add_argument("--cat-iters", type=int, default=3000); ap.add_argument("--cat-boosting", default="Ordered")
ap.add_argument("--cat-desc", action="store_true", help="CatBoost: add categorical process/break descriptors (BIC order, mixture argmax, persistence band, df band)"); ap.add_argument("--cat-gpu", action="store_true", help="CatBoost with the historical GPU-member settings (no rsm), trained on the CPU"); ap.add_argument("--xgb-policy", default="depthwise", help="depthwise | lossguide (leaf-wise, LightGBM-shaped)")
a = ap.parse_args(); t0 = time.time()


def rp(p, cols=None):
    return pd.read_parquet(p, columns=cols).astype(np.float32)


X = rp(D / "clean/features_train.parquet"); cols = list(X.columns); Xt = rp(D / "clean/features_test.parquet", cols)
A = rp(O / "aug_features.parquet", cols)
M = pd.read_csv(O / "aug_meta.csv", dtype=str).set_index("id").loc[A.index]; keep = (M.kind == "shift").to_numpy()
A = A[keep]; src = M.source[keep].to_numpy(); del M
blocks = [b for b in a.blocks.split(",") if b]
for b in blocks:
    B = rp(O / f"{b}_train.parquet").loc[X.index]; bc = list(B.columns)
    Bt = rp(O / f"{b}_test.parquet", bc).loc[Xt.index]; Ba = rp(O / f"aug_{b}_features.parquet", bc).loc[A.index]
    B.columns = Bt.columns = Ba.columns = [f"{b}__{c}" for c in bc]
    X = pd.concat([X, B], axis=1); Xt = pd.concat([Xt, Bt], axis=1); A = pd.concat([A, Ba], axis=1); del B, Bt, Ba
CATC = []
if a.cat_desc:
    def cdesc(df):
        ps = df["lr5__lr_persist"].to_numpy(); dfp = df["lr5__lr_df_pre"].to_numpy()
        return pd.DataFrame({"cd_pbic": df["lr5__lr_p_bic"].fillna(-1).astype(int).astype(str), "cd_mix": df["lr5__lr_mix_argmax_t"].fillna(-1).astype(int).astype(str),
                             "cd_persist": pd.Series(np.digitize(np.nan_to_num(ps), [-0.2, 0.3, 0.8, 0.95]), index=df.index).astype(str),
                             "cd_df": pd.Series(np.digitize(np.nan_to_num(dfp, nan=1e4), [4, 6, 10, 25, 60]), index=df.index).astype(str)}, index=df.index)
    CD, CDt, CDa = cdesc(X), cdesc(Xt), cdesc(A); CATC = list(CD.columns)
if a.cols:
    keepc = [c for c in pd.read_csv(a.cols).feature if c in X.columns]; X, Xt, A = X[keepc], Xt[keepc], A[keepc]; print(f"kept {len(keepc)} columns from {a.cols}", flush=True)
if a.cat_desc: X = pd.concat([X, CD], axis=1); Xt = pd.concat([Xt, CDt], axis=1); A = pd.concat([A, CDa], axis=1); print(f"categorical descriptors {CATC}", flush=True)
if a.rd:
    stems = [c[:-4] for c in cols if c.endswith("_pre") and (c[:-4] + "_post") in cols]

    def rd_block(df):
        out = {}; eps = np.float32(1e-6)
        for s in stems:
            p, q = df[s + "_pre"].to_numpy(), df[s + "_post"].to_numpy()
            out[f"rd_{s}_lr"] = np.log((np.abs(q) + eps) / (np.abs(p) + eps))            # log magnitude ratio
            out[f"rd_{s}_ad"] = np.abs(q - p)                                               # absolute gap
            out[f"rd_{s}_sr"] = (q - p) / (np.abs(q) + np.abs(p) + eps)                     # symmetric relative diff in [-1, 1]
        return pd.DataFrame(out, index=df.index).astype(np.float32)
    X = pd.concat([X, rd_block(X)], axis=1); Xt = pd.concat([Xt, rd_block(Xt)], axis=1); A = pd.concat([A, rd_block(A)], axis=1)
    print(f"rd block: {len(stems)} pairs -> {3*len(stems)} columns", flush=True)
if a.no_aug: A = A.iloc[:0]; src = src[:0]
if a.contrast:
    EV = {"level": ("lr__lr_level_lbf_t", 0.0), "scale": ("lr__lr_scale_lbf_t", 0.0), "shape": ("lr__lr_shape_lr", np.log(12)), "gn": ("lr__lr_gn_shape_lr_sym", np.log(10)), "bim": ("lr__lr_bim_shape_lr_sym", np.log(25)), "ar": ("lr__lr_ar_lr", 0.0)}
    TS = ["ks", "wasserstein", "levene", "dd__dd_wasserstein", "dd__dd_ks", "dd__dd_levene"]
    ref = pd.concat([X, Xt]); mu = {c: float(ref[c].mean()) for c, _ in EV.values()}; sdv = {c: float(ref[c].std() + 1e-6) for c, _ in EV.values()}
    tmu = {c: float(ref[c].mean()) for c in TS}; tsd = {c: float(ref[c].std() + 1e-6) for c in TS}
    def contrast(df):
        Z = np.column_stack([((df[c].to_numpy() - off) / 2 - mu[c]) / sdv[c] if False else (df[c].to_numpy() - off - mu[c]) / sdv[c] for c, off in EV.values()]); srt = np.sort(Z, 1)[:, ::-1]
        out = {"ct_ev_max": srt[:, 0], "ct_ev_2nd": srt[:, 1], "ct_ev_3rd": srt[:, 2], "ct_ev_max_minus_2nd": srt[:, 0] - srt[:, 1], "ct_ev_max_minus_rest": srt[:, 0] - srt[:, 1:].mean(1), "ct_ev_sum": Z.sum(1)}
        am = Z.argmax(1)
        for k, nm in enumerate(EV): out[f"ct_ev_is_{nm}"] = (am == k).astype(np.float32)
        T = np.column_stack([(df[c].to_numpy() - tmu[c]) / tsd[c] for c in TS]); ts = np.sort(T, 1)[:, ::-1]
        out.update({"ct_ts_max": ts[:, 0], "ct_ts_2nd": ts[:, 1], "ct_ts_spread": ts[:, 0] - ts[:, -1], "ct_ts_mean": T.mean(1)})
        return pd.DataFrame(out, index=df.index).astype(np.float32)
    X = pd.concat([X, contrast(X)], axis=1); Xt = pd.concat([Xt, contrast(Xt)], axis=1); A = pd.concat([A, contrast(A)], axis=1); print(f"contrast block: {X.shape[1]} columns now", flush=True)
y = pd.read_csv(D / "data/train_labels.csv", dtype={"id": str}).set_index("id").label.loc[X.index].to_numpy()
ids = X.index.to_numpy()
for df in (X, Xt, A):
    num = [c for c in df.columns if c not in CATC]; df[num] = df[num].replace([np.inf, -np.inf], np.nan)
gc.collect(); print(f"features {X.shape[1]}, shift rows {len(A)}, RAM tables {(X.memory_usage().sum()+Xt.memory_usage().sum()+A.memory_usage().sum())/1e6:.0f} MB ({time.time()-t0:.0f}s)", flush=True)
w = KU_ENV.diag_weights(O)
folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y))

px = json.load(open(O / "best_xgb5.json")); PX = dict(px["params"], device="cpu", nthread=a.threads, seed=a.seed); RX = px["iters"]
PL = {"objective": "binary", "metric": "auc", "verbosity": -1, "num_threads": a.threads, "seed": a.seed, "learning_rate": 0.0128, "num_leaves": 76,
      "max_depth": -1, "min_data_in_leaf": 15, "feature_fraction": 0.3552, "bagging_fraction": 0.8938, "bagging_freq": 1,
      "lambda_l1": 0.3403, "lambda_l2": 1.1182, "min_gain_to_split": 0.4257}; RL = 499
if a.xgb_policy == "lossguide":        # LightGBM-shaped XGBoost (historical member, fixed settings)
    PX.update(grow_policy="lossguide", max_leaves=76, max_depth=0, learning_rate=0.0128, min_child_weight=3.75, subsample=0.8938,
              colsample_bytree=0.3552, colsample_bynode=1.0, reg_alpha=0.3403, reg_lambda=1.1182, gamma=0.4257); RX = RL
if a.extra: PX.update(json.loads(a.extra)); RX = int(PX.pop("rounds", RX))
PX.update(device="cpu", tree_method="hist")          # Enforced after every merge, so no saved or CLI parameter can select a GPU.
if str(PX.get("objective", "")).startswith("rank:"): PX.pop("base_score", None); PX["eval_metric"] = "auc"
if a.colsample is not None:
    PX["colsample_bytree"] = PX["colsample_bynode"] = a.colsample; PL["feature_fraction"] = a.colsample
if a.mono:
    import re
    pat = re.compile(r"(_lbf_|_maxlr_|lr_level_abs_t|lr_shape_lr|lr_gn_shape_lr|lr_bim_shape_lr|lr_ar_lr$|^ks$|^wasserstein$|^levene$|^dd__dd_wasserstein$|^dd__dd_ks$|^dd__dd_levene$|^ct_ev_max|^ct_ts_max|^ct_ts_spread|_2s_(cvm|ad|energy|wass|ks)$|_agap$|^abs_|^dd__dd_chow|_cusum_at_b$|cabs_at_b$|cz2_at_b$)")
    mono = tuple(1 if pat.search(c) else 0 for c in X.columns); PX["monotone_constraints"] = "(" + ",".join(map(str, mono)) + ")"; print(f"monotone +1 on {sum(mono)} of {len(mono)} columns", flush=True)
obj = None
if a.obj.startswith("gce"):
    q = float(a.obj.split(":")[1])

    def obj(pred, dtrain):                      # weighted-logloss Newton step with weight p_y^q (GCE gradient, positive surrogate hessian)
        yy = dtrain.get_label(); p = 1 / (1 + np.exp(-pred)); py = np.where(yy == 1, p, 1 - p); wq = np.clip(py, 1e-6, 1) ** q
        return wq * (p - yy), np.maximum(wq * p * (1 - p), 1e-6)
    PX.pop("objective", None); PX["base_score"] = 0.5


def host(df):
    """pandas -> contiguous float32 array, without keeping a second copy longer than one fit."""
    return np.ascontiguousarray(df.to_numpy(np.float32))


SAVE = None


def fit_xgb(Xtr, ytr, Xva, rounds):
    import xgboost as xgb
    dtr = xgb.QuantileDMatrix(host(Xtr), ytr, max_bin=PX.get("max_bin", 256)); dva = xgb.QuantileDMatrix(host(Xva), ref=dtr, max_bin=PX.get("max_bin", 256))
    bst = xgb.train(PX, dtr, rounds, obj=obj)
    if str(PX.get("objective", "")).startswith("rank:"):          # ranking objective: margins are scores; squash so the stack's logit input = the score
        p = 1 / (1 + np.exp(-np.clip(bst.predict(dva, output_margin=True), -30, 30)))
    else: p = bst.predict(dva)
    if SAVE: bst.save_model(str(SAVE) + '.ubj'); KU_ENV.check_xgb(str(SAVE) + '.ubj', bst, dva)
    del dtr, dva, bst; gc.collect(); return p


def fit_lgb(Xtr, ytr, Xva, rounds):
    import lightgbm as lgb
    b = lgb.train(PL, lgb.Dataset(Xtr, ytr, free_raw_data=True), rounds)
    p = b.predict(Xva)
    if SAVE:
        b.save_model(str(SAVE) + ".txt")
        if float(np.abs(lgb.Booster(model_file=str(SAVE) + ".txt").predict(Xva) - p).max()) > 1e-6: raise RuntimeError("saved LightGBM model differs")
    return p


def fit_cat(Xtr, ytr, Xva, rounds):
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(boosting_type=a.cat_boosting, depth=6, learning_rate=0.03, iterations=rounds, l2_leaf_reg=5, random_strength=2,
                           bagging_temperature=0.3, border_count=64, eval_metric="AUC", loss_function="Logloss", thread_count=a.threads,
                           random_seed=a.seed, verbose=0, allow_writing_files=False,
                           **({"task_type": "CPU"} if a.cat_gpu else {"task_type": "CPU", "rsm": 0.3}))
    m.fit(Xtr, ytr, cat_features=CATC or None)
    p = m.predict_proba(Xva)[:, 1]
    if SAVE:
        m.save_model(str(SAVE) + '.cbm'); c = CatBoostClassifier(); c.load_model(str(SAVE) + '.cbm')
        if float(np.abs(c.predict_proba(Xva)[:, 1] - p).max()) > 1e-6: raise RuntimeError('saved CatBoost model differs')
    return p


fit, R = {"xgb": (fit_xgb, RX), "lgb": (fit_lgb, RL), "cat": (fit_cat, a.cat_iters)}[a.model]
oof = np.zeros(len(y))
drop = set(pd.read_csv(a.drop_train, dtype=str).iloc[:, 0]) if a.drop_train else set(); keepmask = ~np.isin(ids, list(drop))
for k, (tr, va) in enumerate(folds):
    tr = tr[keepmask[tr]]
    sel = np.isin(src, ids[tr]); Xtr = pd.concat([X.iloc[tr], A[sel]]); ytr = np.concatenate([y[tr], np.zeros(sel.sum())])
    oof[va] = fit(Xtr, ytr, X.iloc[va], R); del Xtr
    print(f"  {a.model} fold {k} {roc_auc_score(y[va], oof[va]):.4f} ({time.time()-t0:.0f}s)", flush=True)
cv, cvw = roc_auc_score(y, oof), roc_auc_score(y, oof, sample_weight=w)
ref = {}
for m in ("xgb", "lgb"):
    f = O / f"oof_v5_{m}.npy"
    if f.exists(): o = np.load(f); ref[m] = (roc_auc_score(y, o), roc_auc_score(y, o, sample_weight=w))
print(f"{a.tag} ({a.model}): CV AUC {cv:.5f}  test-weighted {cvw:.5f}   | v5 same model: " + ", ".join(f"{m} {p:.5f}/{pw:.5f}" for m, (p, pw) in ref.items()), flush=True)
np.save(O / f"oof_{a.tag}.npy", oof)
info = {"note": f"v6 train_cv contrast={a.contrast} mono={a.mono} cols={a.cols} extra={a.extra} drop_train={a.drop_train} no_aug={a.no_aug} model={a.model} rd={a.rd} colsample={a.colsample} obj={a.obj} blocks={blocks} refit_mult={a.refit_mult}",
        "cv": float(cv), "cv_test_weighted": float(cvw), "n_features": int(X.shape[1])}
if not a.no_test:
    import os; SAVE = (D / 'weights' / a.tag) if os.environ.get('SAVE_WEIGHTS', '1') != '0' else None
    if SAVE: SAVE.parent.mkdir(exist_ok=True); json.dump({'columns': list(X.columns), 'note': info['note']}, open(str(SAVE) + '.meta.json', 'w'), indent=1)
    rounds = int(round(R * a.refit_mult)); test = fit(pd.concat([X[keepmask], A]), np.concatenate([y[keepmask], np.zeros(len(A))]), Xt, rounds)
    np.save(O / f"test_{a.tag}.npy", test); info["refit_rounds"] = rounds
json.dump(info, open(O / f"best_{a.tag}.json", "w"), indent=1); print(f"done {time.time()-t0:.0f}s", flush=True)
