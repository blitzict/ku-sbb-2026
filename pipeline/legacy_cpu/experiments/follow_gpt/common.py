# CPU port of experiments/follow_gpt/common.py, source sha256 1e0b3b412d7ee2ec042fe16432682fa597ca0778b0e877e9fdf7341e8606aef0.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KW
"""Shared pieces for following the GPT weight-edit recipe on other members (2026-11-04): build the view-stack design, nested
non-negative logistic CV on several meta splits."""
import json, numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
O = "tune_out/"; WE = O + "weight_edit_20261104/"
ids = pd.read_parquet("clean/features_train.parquet", columns=[]).index; tids = pd.read_parquet("clean/features_test.parquet", columns=[]).index
y = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id").label.loc[ids].to_numpy().astype(float)
w = KU_ENV.diag_weights(O)
L = lambda p: logit(np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6))
V4 = ["orig", "rev", "flip", "revflip"]


def ver(tags, kinds, base=O, part="oof"):
    out = []
    for k in kinds:
        acc = []
        for t in tags:
            z = np.load(base + f"ver_{t}.npz"); kk = list(z["kinds"]); acc.append(L(z[part][kk.index(k)]))
        out.append(np.mean(acc, 0))
    return out


def seq_views(tags, part="oof", sub="seqpart"):
    """per-view logits of nn_seq runs, mean over runs: oof from tune_out/seqpart/<tag>_f*.npz (pv), test = mean over folds (pt)."""
    res = None
    for t in tags:
        if part == "oof":
            M = np.zeros((4, len(ids)))
            for f in range(5): z = np.load(O + f"{sub}/{t}_f{f}.npz", allow_pickle=True); M[:, z["va"]] = np.clip(z["pv"], -30, 30)
        else: M = np.mean([np.clip(np.load(O + f"{sub}/{t}_f{f}.npz", allow_pickle=True)["pt"], -30, 30) for f in range(5)], 0)
        res = M if res is None else res + M
    return list(res / len(tags))


SEQ5 = ["W_SEQ_q", "W_SEQ_q_s1", "W_SEQ_qh", "W_SEQ_q_s2", "W_SEQ_qh_s1"]


def gpt_design(part="oof", seq="bag", mlp=None, seqdir="seqpart"):
    """GPT's 20-input recipe (views of the 3 XGB groups + edited-head MLP views + L3 + L4 + SEQ); seq='views' expands SEQ into 4 views; mlp overrides MLP view source."""
    g = lambda t: L(np.load(O + f"{part}_{t}.npy"))
    cols = [g("L3_rank_lr03_s0"), g("L4_fam_gh_b2")]
    cols += ver(["VARC4hv_ns3_s0", "VARC4hv_ns3_s1"], V4, part=part) + ver(["VARC4_lm_s0", "VARC4_lm_s1"], V4, part=part) + ver(["VARC4_full_s0", "VARC4_full_s1"], V4, part=part)
    mt, mb = mlp or (["head_v2_MLPHV_big", "head_v2_MLPHV_big_o1", "head_v2_MLPHV_big_o2"], WE)
    cols += ver(mt, V4 + ["c2"], base=mb, part=part)
    cols += seq_views(SEQ5, part, seqdir) if seq == "views" else [g("W_SEQ_qb5")]
    return np.column_stack(cols)


def fit(X, yy, C=0.1):
    k = X.shape[1]
    def f(th):
        z = X @ th[:k] + th[k]; p = 1 / (1 + np.exp(-z))
        return C * np.sum(np.logaddexp(0, z) - yy * z) + 0.5 * th[:k] @ th[:k], np.concatenate([C * (X.T @ (p - yy)) + th[:k], [C * np.sum(p - yy)]])
    return minimize(f, np.zeros(k + 1), jac=True, method="L-BFGS-B", bounds=[(0, None)] * k + [(None, None)], options={"maxiter": 5000}).x


def nested(X, rs):
    o = np.zeros(len(y))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=rs).split(X, y): th = fit(X[tr], y[tr]); o[va] = X[va] @ th[:-1] + th[-1]
    return roc_auc_score(y, o), roc_auc_score(y, o, sample_weight=w)


def report(nm, X, splits=(7, 29, 123, 2024)):
    r = [nested(X, rs) for rs in splits]
    print(f"{nm:40s} " + "  ".join(f"{a:.5f}/{b:.5f}" for a, b in r) + f"   mean {np.mean([a for a, _ in r]):.5f}/{np.mean([b for _, b in r]):.5f}", flush=True)
    return r
