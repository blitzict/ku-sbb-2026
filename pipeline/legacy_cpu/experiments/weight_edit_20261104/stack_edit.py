# CPU port of experiments/weight_edit_20261104/stack_edit.py, source sha256 a994b3e73d89aa94f2270581ac98ae16dcd8c5cc786e69333a8225d77ed3c358. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV
"""Reproducible, cross-fitted edits of saved ensemble coefficients.

All scores refer to meta-level CV on existing OOF base predictions, not to a
fresh, fully nested retraining of the complete pipeline or hidden test labels.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import argparse
import ctypes
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

ROOT = KU_ENV.root()
OUT = ROOT / "tune_out" / "weight_edit_20261104"


def setup():
    try:
        KU_ENV.low_priority()
    except (AttributeError, OSError):
        pass
    torch.set_num_threads(KU_ENV.threads())
    OUT.mkdir(parents=True, exist_ok=True)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_data(base):
    meta = json.loads(base.read_text())
    names = meta["members"]
    assert len(names) == len(set(names))
    ids = pd.read_parquet(ROOT / "clean/features_train.parquet", columns=[]).index
    tids = pd.read_parquet(ROOT / "clean/features_test.parquet", columns=[]).index
    labels = pd.read_csv(ROOT / "data/train_labels.csv", dtype={"id": str}).set_index("id")
    assert ids.is_unique and tids.is_unique and labels.index.is_unique
    assert set(ids) == set(labels.index) and not set(ids).intersection(tids)
    y = labels.loc[ids, "label"].to_numpy().astype(np.float64)
    wt = KU_ENV.diag_weights(ROOT / "tune_out")
    wt = np.ones(len(y)) if wt is None else wt
    mats = []
    hashes = {}
    for prefix, size in [("oof", len(ids)), ("test", len(tids))]:
        cols = []
        for name in names:
            path = ROOT / "tune_out" / f"{prefix}_{name}.npy"
            p = np.load(path, allow_pickle=False)
            assert p.shape == (size,) and np.isfinite(p).all(), str(path)
            assert (p >= 0).all() and (p <= 1).all(), str(path)
            cols.append(logit(np.clip(p, 1e-6, 1 - 1e-6)))
            hashes[str(path.relative_to(ROOT))] = sha(path)
        mats.append(np.column_stack(cols).astype(np.float64))
    audit = {
        "base": str(base.relative_to(ROOT)), "base_sha256": sha(base),
        "n_train": len(y), "n_test": len(tids), "positives": int(y.sum()),
        "members": len(names), "prediction_hashes": hashes,
        "dummy_auc": float(roc_auc_score(y, np.full(len(y), 0.5))),
        "reported_baseline": meta.get("cv"),
        "validation_scope": "Meta-level CV on pre-existing OOF predictions; not full-pipeline nested CV.",
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2))
    return names, ids, tids, y, wt, mats[0], mats[1], meta


def logistic_fit(X, y, C=0.1):
    k = X.shape[1]
    def f(th):
        z = X @ th[:k] + th[k]
        r = expit(z) - y
        cost = C * np.sum(np.logaddexp(0, z) - y * z) + 0.5 * th[:k] @ th[:k]
        grad = np.r_[C * (X.T @ r) + th[:k], C * r.sum()]
        return cost, grad
    opt = minimize(f, np.zeros(k + 1), jac=True, method="L-BFGS-B",
                   bounds=[(0, None)] * k + [(None, None)], options={"maxiter": 5000})
    if not opt.success:
        raise RuntimeError(f"baseline optimizer failed: {opt.message}")
    return opt.x


def ranking_fit(X, y, initial, mode, temp, anchor, seed, pairs):
    """Fixed Monte Carlo positive/negative pairs; no evaluation rows used."""
    rng = np.random.default_rng(seed)
    pos = rng.choice(np.flatnonzero(y == 1), pairs)
    neg = rng.choice(np.flatnonzero(y == 0), pairs)
    xd = torch.as_tensor(X, dtype=torch.float32, device="cpu")
    ip = torch.as_tensor(pos, device="cpu")
    im = torch.as_tensor(neg, device="cpu")
    ref = torch.as_tensor(initial[:-1], dtype=torch.float32, device="cpu")
    def f(coef):
        b = torch.tensor(coef, dtype=torch.float32, device="cpu", requires_grad=True)
        z = xd @ b
        d = (z[ip] - z[im]) / temp
        loss = torch.nn.functional.softplus(-d).mean() if mode == "pairlog" else torch.sigmoid(-d).mean()
        loss = loss + anchor * ((b - ref) ** 2).sum()
        loss.backward()
        return float(loss.detach().cpu()), b.grad.detach().cpu().numpy().astype(np.float64)
    opt = minimize(f, initial[:-1], jac=True, method="L-BFGS-B",
                   bounds=[(0, None)] * (len(initial) - 1),
                   options={"maxiter": 250, "ftol": 2e-9, "gtol": 2e-6, "maxls": 30})
    assert np.isfinite(opt.x).all()
    # Retain a sensible probability offset; it has no effect on within-fit AUC.
    z = X @ opt.x
    bias = float(initial[-1])
    for _ in range(20):
        p = expit(z + bias)
        step = (p - y).sum() / max(float((p * (1 - p)).sum()), 1e-8)
        bias -= float(np.clip(step, -2, 2))
        if abs(step) < 1e-9:
            break
    return np.r_[opt.x, bias], {"success": bool(opt.success), "message": str(opt.message), "iterations": int(opt.nit)}


def score(y, z, wt):
    return {"auc": float(roc_auc_score(y, z)), "weighted_auc": float(roc_auc_score(y, z, sample_weight=wt))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="submissions/sub_seqw5_stack.json")
    ap.add_argument("--pairs", type=int, default=524288)
    ap.add_argument("--splits", type=int, nargs="+", default=[7, 29])
    a = ap.parse_args()
    setup()
    names, ids, tids, y, wt, X, T, meta = load_data(ROOT / a.base)
    start = time.time()
    configs = {
        "pairlog": ("pairlog", 1.0, 0.001),
        "auc_t075": ("smoothauc", 0.75, 0.002),
        "auc_t025": ("smoothauc", 0.25, 0.002),
        "auc_t010": ("smoothauc", 0.10, 0.01),
    }
    results, oofs, coefs = {}, {}, {}
    for split_seed in a.splits:
        pred = {name: np.zeros(len(y)) for name in ["baseline", *configs]}
        fitmeta = []
        for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=split_seed).split(X, y)):
            init = logistic_fit(X[tr], y[tr])
            pred["baseline"][va] = X[va] @ init[:-1] + init[-1]
            for name, (mode, temp, anchor) in configs.items():
                theta, info = ranking_fit(X[tr], y[tr], init, mode, temp, anchor,
                                          20261104 + split_seed * 100 + fold, a.pairs)
                pred[name][va] = X[va] @ theta[:-1] + theta[-1]
                fitmeta.append({"fold": fold, "name": name, **info})
                coefs[f"s{split_seed}_f{fold}_{name}"] = theta
            print(f"stack split {split_seed} fold {fold}: " + " ".join(f"{name}={roc_auc_score(y[va], z[va]):.5f}" for name, z in pred.items()) + f" ({time.time()-start:.0f}s)", flush=True)
        results[str(split_seed)] = {name: score(y, p, wt) for name, p in pred.items()}
        results[str(split_seed)]["optimizer_status"] = fitmeta
        for name, p in pred.items():
            oofs[f"s{split_seed}_{name}"] = p
        print(json.dumps({"split": split_seed, "scores": {k: v for k, v in results[str(split_seed)].items() if k != "optimizer_status"}}), flush=True)
        (OUT / "stack_results.json").write_text(json.dumps(results, indent=2))
        np.savez(OUT / "stack_oof.npz", **oofs, ids=np.asarray(ids, dtype=str), y=y, sample_weight=wt)
    if 7 in a.splits:
        assert meta.get("reported_baseline") is None or abs(results["7"]["baseline"]["auc"] - meta["reported_baseline"]) < 2e-5, "baseline mismatch"
    init = logistic_fit(X, y)
    full_weights = {"baseline": init.tolist()}
    full_pred = {"baseline": expit(T @ init[:-1] + init[-1])}
    for name, (mode, temp, anchor) in configs.items():
        th, info = ranking_fit(X, y, init, mode, temp, anchor, 20261104, a.pairs)
        full_weights[name] = th.tolist()
        full_pred[name] = expit(T @ th[:-1] + th[-1])
    artifact = {"members": names, "weights_and_intercepts": full_weights,
                "settings": configs, "scores": results, "source_base": a.base,
                "status": "experimental candidates; compare held-out meta scores before promotion",
                "validation_scope": "CV of meta learner on existing OOF base predictions; not fully nested base retraining",
                "elapsed_seconds": time.time() - start}
    wd = ROOT / "weights_experiments" / "weight_edit_20261104"
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "stack_coefficients.json").write_text(json.dumps(artifact, indent=2))
    np.savez(OUT / "stack_test.npz", **full_pred, ids=np.asarray(tids, dtype=str))
    np.savez(OUT / "stack_fold_coefficients.npz", **coefs)
    print(f"Saved coefficients and measured candidates in {wd}; {time.time()-start:.0f}s", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
