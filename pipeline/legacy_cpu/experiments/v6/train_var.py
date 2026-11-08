# CPU port of experiments/v6/train_var.py, source sha256 ae82955fefee511bb3c5e515fa6ab271f5cfc57b31143b59eb68aa18e6c0047c. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV
"""Time-reversal / sign-flip augmentation + test-time averaging (user-approved 2026-11-02), leaf-wise XGBoost, CPU port.
Column set: tune_out/selected_ns3.csv (170 cols over clean, grid, grid2, dd, lr5, lrz, gh); the calibration block is taken from lrz6_* (recomputed
with the current features_lr mixture so originals and transformed series match).
Honest CV: canonical 5 folds; a fold's training set = its original rows + boundary-shift negatives of its sources + (with --aug) the rev / flip /
revflip copies of its own training series (same labels). Validation = original rows; '_tta' = mean prediction over the 4 versions of each validation
series (features only, no labels). Test: model refit on everything, prediction averaged over the 4 versions (plain also saved).
Usage: python train_var.py --tag VAR_base [--aug] [--kinds rev,flip,revflip] [--rounds 499] [--seed 0] [--params json]"""
import sys, json, time, pathlib, argparse, warnings, ctypes, gc; warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
try: ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x4000)
except Exception: pass
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
D = KU_ENV.root(); O = D / "tune_out"; V = O / "var"
ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--aug", action="store_true"); ap.add_argument("--kinds", default="rev,flip,revflip")
ap.add_argument("--rounds", type=int, default=499); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--params", default=None); ap.add_argument("--no-shift", action="store_true"); ap.add_argument("--colfile", default="tune_out/selected_ns3.csv")
ap.add_argument("--tta-kinds", default="rev,flip,revflip"); ap.add_argument("--aug-weight", type=float, default=1.0, help="sample weight of each transformed copy (originals and shift rows 1)"); ap.add_argument("--model", default="xgb", help="xgb | cat (CatBoost, historical GPU-member settings, on the CPU)"); ap.add_argument("--cat-iters", type=int, default=3000); ap.add_argument("--sym", action="store_true", help="append mean and spread over the orig/rev/flip/revflip versions of each feature (series-level, shared by all copies of a series; needs --no-shift)"); ap.add_argument("--extra", default="", help="extra real-segment rows: var stem (e.g. hv) with tune_out/var/<stem>_meta.csv (id, source, label, kind)"); ap.add_argument("--extra-w", type=float, default=1.0); ap.add_argument("--extra-kinds", default="h0,h1,p"); ap.add_argument("--ic-blocks", action="store_true", help="XGB interaction constraints: features may only interact within their feature block"); ap.add_argument("--frac", type=float, default=1.0, help="learning-curve probe: keep this fraction of each training fold (stratified by seed)"); ap.add_argument("--kind-w", default="", help="per-kind training weights, e.g. c1=0.5,c2=0.5 (overrides --aug-weight)"); ap.add_argument("--export", default=None, help="save the assembled matrices to this .npz (for nn_var.py on the Arc) and exit")
a = ap.parse_args(); KW = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.kind_w.split(",") if kv}; t0 = time.time(); KINDS = [k for k in a.kinds.split(",") if k]; SUF = {"rev": "r", "flip": "f", "revflip": "rf", "c1": "c1", "c2": "c2", "c3": "c3", "c4": "c4", "c5": "c5", "c6": "c6", "d2a": "d2a", "d2b": "d2b", "a2": "a2"}
COLS = list(pd.read_csv(D / a.colfile).feature)
BLOCKS = {"grid": "grid", "grid2": "grid2", "dd": "dd", "lr5": "lr5", "lrz": "lrz6", "gh": "gh", "sh": "sh", "cl": "cl", "lr": "lr5", "sn": "sn", "zp": "zp", "kn": "kn", "wh": "wh", "ps": "ps", "lm": "lm", "harm": "harm"}          # column prefix -> file prefix


_C = {"cols": None}


def need(prefix): return [c[len(prefix) + 2:] for c in COLS if c.startswith(prefix + "__")]


CLEAN = [c for c in COLS if "__" not in c]


def assemble(get):
    """get(file_prefix_or_'clean') -> DataFrame indexed by id; returns the 170 columns in COLS order."""
    _C["cols"] = CLEAN
    parts = [get("clean")[CLEAN]] if CLEAN else []
    for cp, fp in BLOCKS.items():
        cols = need(cp)
        _C["cols"] = cols
        if cols: B = get(fp)[cols]; B.columns = [f"{cp}__{c}" for c in cols]; parts.append(B)
    _C["cols"] = None
    X = pd.concat(parts, axis=1)[COLS].astype(np.float32); return X.replace([np.inf, -np.inf], np.nan)


rp = lambda p: pd.read_parquet(p, columns=_C["cols"])     # Reads only the requested columns, for the 8 GB target.
X = assemble(lambda fp: rp(D / "clean/features_train.parquet") if fp == "clean" else rp(O / f"{fp}_train.parquet")); ids = X.index.to_numpy()
Xt = assemble(lambda fp: rp(D / "clean/features_test.parquet") if fp == "clean" else rp(O / f"{fp}_test.parquet")).loc[rp(D / "clean/features_test.parquet").index]
A = assemble(lambda fp: rp(O / "aug_features.parquet") if fp == "clean" else rp(O / f"aug_{fp}_features.parquet"))
M = pd.read_csv(O / "aug_meta.csv", dtype=str).set_index("id"); A = A.loc[[i for i in A.index if M.loc[i, "kind"] == "shift"]]; src = M.loc[A.index, "source"].to_numpy()
if a.no_shift: A = A.iloc[:0]; src = src[:0]
y = pd.read_csv(D / "data/train_labels.csv", dtype={"id": str}).set_index("id").label.loc[ids].to_numpy(); X = X.loc[ids]
TTAK = [k for k in a.tta_kinds.split(",") if k]; ALLK = list(dict.fromkeys(KINDS + TTAK))
VAR = {k: assemble(lambda fp, k=k: rp(V / f"{fp}_train_{k}.parquet")).loc[[i + SUF[k] for i in ids]].to_numpy() for k in ALLK}
VART = {k: assemble(lambda fp, k=k: rp(V / f"{fp}_test_{k}.parquet")).loc[[i + SUF[k] for i in Xt.index]].to_numpy() for k in TTAK}
Xn, Xtn, An = X.to_numpy(), Xt.to_numpy(), A.to_numpy()
if a.extra:
    XE, YE, SE = [], [], []
    for st in a.extra.split(","):                                   # several extra sets, e.g. hv,hvb
        EM = pd.read_csv(V / f"{st}_meta.csv", dtype={"id": str, "source": str}); EM = EM[EM.kind.isin(a.extra_kinds.split(","))]
        XE.append(assemble(lambda fp: rp(V / f"{fp}_train_{st}.parquet")).loc[EM.id].to_numpy()); YE.append(EM.label.to_numpy()); SE.append(EM.source.to_numpy())
    Xe, ye, srce = np.vstack(XE), np.concatenate(YE), np.concatenate(SE)
    print(f"extra rows {len(ye)} ({a.extra_kinds}), label rate {ye.mean():.3f}", flush=True)
if a.sym:
    assert a.no_shift, "--sym needs --no-shift (shift negatives have no rev/flip versions)"
    def symf(*vs): S = np.stack(vs); return np.hstack([np.nanmean(S, 0), np.nanmax(S, 0) - np.nanmin(S, 0)]).astype(np.float32)
    Str = symf(Xn, VAR["rev"], VAR["flip"], VAR["revflip"]); Ste = symf(Xtn, VART["rev"], VART["flip"], VART["revflip"])
    Xn, Xtn, An = np.hstack([Xn, Str]), np.hstack([Xtn, Ste]), np.zeros((0, Xn.shape[1] + Str.shape[1]), np.float32)
    VAR = {k: np.hstack([v, Str]) for k, v in VAR.items()}; VART = {k: np.hstack([v, Ste]) for k, v in VART.items()}
w = KU_ENV.diag_weights(O)
P = {"objective": "binary:logistic", "device": "cpu", "tree_method": "hist", "nthread": KU_ENV.threads(), "seed": a.seed, "grow_policy": "lossguide", "max_leaves": 76, "max_depth": 0,
     "learning_rate": 0.0128, "min_child_weight": 3.75, "subsample": 0.8938, "colsample_bytree": 0.7, "colsample_bynode": 1.0, "reg_alpha": 0.3403, "reg_lambda": 1.1182,
     "gamma": 0.4257, "max_bin": 256, "verbosity": 0}
R = a.rounds
if a.export:
    np.savez(a.export, X=Xn, Xt=Xtn, A=An, y=y, w=(np.ones(len(y)) if w is None else w), ids=ids, src=src, tid=Xt.index.to_numpy(), cols=np.array(COLS), **{f"V_{k}": v for k, v in VAR.items()}, **{f"VT_{k}": v for k, v in VART.items()}, **({"E": Xe, "Ey": ye, "Esrc": srce} if a.extra else {}))
    print(f"exported {a.export}"); sys.exit(0)
if a.params: extra = json.loads(a.params); R = int(extra.pop("rounds", R)); P.update(extra)
P.update(device="cpu", tree_method="hist")           # Enforced after every merge.
if a.ic_blocks:
    blk = [c.split("__")[0] if "__" in c else "clean" for c in COLS]; P["interaction_constraints"] = str([[i for i, b in enumerate(blk) if b == u] for u in dict.fromkeys(blk)])
print(f"{a.tag}: {Xn.shape[1]} cols, shift rows {len(An)}, aug={a.aug} kinds={KINDS}, rounds {R} ({time.time()-t0:.0f}s)", flush=True)


SAVE = None


def fit_predict(Xtr, ytr, outs, wtr=None):
    if a.model == "cat":
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(depth=6, learning_rate=0.03, iterations=a.cat_iters, l2_leaf_reg=5, random_strength=2, bagging_temperature=0.3, border_count=64,
                               loss_function="Logloss", random_seed=a.seed, verbose=0, allow_writing_files=False, thread_count=KU_ENV.threads(), task_type="CPU")
        m.fit(Xtr, ytr, sample_weight=wtr); res = [m.predict_proba(o)[:, 1] for o in outs]
        if SAVE:
            m.save_model(str(SAVE) + '.cbm'); c = CatBoostClassifier(); c.load_model(str(SAVE) + '.cbm')
            if float(np.abs(c.predict_proba(outs[0])[:, 1] - res[0]).max()) > 1e-6: raise RuntimeError('saved CatBoost model differs')
        del m; gc.collect(); return res
    d = xgb.QuantileDMatrix(Xtr, ytr, weight=wtr, max_bin=P["max_bin"]); b = xgb.train(P, d, R)
    res = [b.predict(xgb.QuantileDMatrix(o, ref=d, max_bin=P["max_bin"])) for o in outs]
    if SAVE: b.save_model(str(SAVE) + '.ubj'); KU_ENV.check_xgb(str(SAVE) + '.ubj', b, xgb.QuantileDMatrix(outs[0], ref=d, max_bin=P["max_bin"]))
    del d, b; gc.collect()
    if P["objective"].startswith("rank"): res = [1 / (1 + np.exp(-r)) for r in res]
    return res


folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(Xn, y)); oof = np.zeros(len(y)); tta = np.zeros(len(y)); ver = np.zeros((1 + len(TTAK), len(y)), np.float32)
for k, (tr, va) in enumerate(folds):
    if a.frac < 1: tr = np.sort(np.random.default_rng(100 + k).permutation(tr)[:int(len(tr) * a.frac)])
    sel = np.isin(src, ids[tr]); parts = [Xn[tr], An[sel]]; ys = [y[tr], np.zeros(sel.sum())]; ws = [np.ones(len(tr) + sel.sum())]
    if a.aug:
        for kind in KINDS: parts.append(VAR[kind][tr]); ys.append(y[tr]); ws.append(np.full(len(tr), KW.get(kind, a.aug_weight)))
    if a.extra: se = np.isin(srce, ids[tr]) | ~np.isin(srce, ids); parts.append(Xe[se]); ys.append(ye[se]); ws.append(np.full(se.sum(), a.extra_w))
    fit_matrix, fit_labels, fit_weights = np.vstack(parts), np.concatenate(ys), np.concatenate(ws)
    del parts, ys, ws  # Release copied fold rows before the quantized tree matrix is allocated.
    preds = fit_predict(fit_matrix, fit_labels, [Xn[va]] + [VAR[kd][va] for kd in TTAK], fit_weights)
    del fit_matrix, fit_labels, fit_weights
    oof[va] = preds[0]; tta[va] = np.mean(preds, 0); ver[:, va] = np.stack(preds)
    print(f"  fold {k}: {roc_auc_score(y[va], oof[va]):.4f}  tta {roc_auc_score(y[va], tta[va]):.4f}  ({time.time()-t0:.0f}s)", flush=True)
rep = lambda s: f"{roc_auc_score(y, s):.5f} / {roc_auc_score(y, s, sample_weight=w):.5f}"
print(f"{a.tag}: plain {rep(oof)}   tta {rep(tta)}", flush=True)
parts = [Xn, An]; ys = [y, np.zeros(len(An))]; ws = [np.ones(len(Xn) + len(An))]
if a.aug:
    for kind in KINDS: parts.append(VAR[kind]); ys.append(y); ws.append(np.full(len(Xn), KW.get(kind, a.aug_weight)))
if a.extra: parts.append(Xe); ys.append(ye); ws.append(np.full(len(ye), a.extra_w))
import os
SAVE = (D / 'weights' / a.tag) if os.environ.get('SAVE_WEIGHTS', '1') != '0' else None
if SAVE: SAVE.parent.mkdir(exist_ok=True); json.dump({'columns': list(COLS), 'kinds': KINDS, 'tta_kinds': TTAK, 'params': P, 'rounds': R}, open(str(SAVE) + '.meta.json', 'w'), indent=1)
tp = fit_predict(np.vstack(parts), np.concatenate(ys), [Xtn] + [VART[kd] for kd in TTAK], np.concatenate(ws))
np.savez(O / f"ver_{a.tag}.npz", oof=ver, test=np.stack(tp).astype(np.float32), kinds=np.array(["orig"] + TTAK))
np.save(O / f"oof_{a.tag}.npy", oof); np.save(O / f"oof_{a.tag}_tta.npy", tta); np.save(O / f"test_{a.tag}.npy", tp[0]); np.save(O / f"test_{a.tag}_tta.npy", np.mean(tp, 0))
for t in (a.tag, a.tag + "_tta"):
    s = oof if t == a.tag else tta
    json.dump({"note": f"train_var aug={a.aug} kinds={KINDS} tta={TTAK} rounds={R} params={a.params}", "cv": float(roc_auc_score(y, s)), "cv_test_weighted": float(roc_auc_score(y, s, sample_weight=w))}, open(O / f"best_{t}.json", "w"), indent=1)
print(f"done {time.time()-t0:.0f}s", flush=True)
