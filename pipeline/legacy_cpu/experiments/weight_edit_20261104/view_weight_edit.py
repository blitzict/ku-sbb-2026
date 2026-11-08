# CPU port of experiments/weight_edit_20261104/view_weight_edit.py, source sha256 8194cf8208f43ac0f9212cd77ddd3ae1ffabe21fdd7dc9d6c81309e9c4fcd379. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV
"""Relax hand-picked augmentation weights, with group shrinkage and meta CV.

The initial model is exactly the existing stack. Each selected member is expanded
into its saved test-time views. A penalty keeps learned view weights near the
original proportions; all final coefficients remain nonnegative.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import json
import ctypes
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import logit, expit
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from stack_edit import ROOT, OUT, load_data, logistic_fit, score, sha


GROUPS = {
    "VARC4hv_ns3_wc": (["VARC4hv_ns3_s0", "VARC4hv_ns3_s1"], .125),
    "VARC4_lm_wc": (["VARC4_lm_s0", "VARC4_lm_s1"], .125),
    "VARC4_full_wc2": (["VARC4_full_s0", "VARC4_full_s1"], .125),
    "MLPHV_bag_w": (["MLPHV_big", "MLPHV_big_o1", "MLPHV_big_o2"], .5),
}


def expand(names, X, T, archive_paths=None):
    archive_paths = archive_paths or {}
    cols, tcols, labels, groups, hashes = [], [], [], [], {}
    for member_i, name in enumerate(names):
        start = len(cols)
        if name not in GROUPS:
            cols.append(X[:, member_i])
            tcols.append(T[:, member_i])
            labels.append(name)
            weights = np.ones(1)
        else:
            tags, crop = GROUPS[name]
            values = []
            tests = []
            kinds = None
            for tag in tags:
                path = archive_paths.get(tag, ROOT / "tune_out" / f"ver_{tag}.npz")
                z = np.load(path, allow_pickle=False)
                if kinds is not None:
                    assert np.array_equal(kinds, z["kinds"])
                kinds = z["kinds"]
                values.append(logit(np.clip(z["oof"].astype(float), 1e-6, 1 - 1e-6)))
                tests.append(logit(np.clip(z["test"].astype(float), 1e-6, 1 - 1e-6)))
                hashes[str(path.relative_to(ROOT))] = sha(path)
            v = np.mean(values, axis=0)
            t = np.mean(tests, axis=0)
            wt = {"orig": 2., "flip": 2., "rev": 1., "revflip": 1.}
            weights = np.array([wt.get(str(k), crop) for k in kinds])
            weights /= weights.sum()
            # Existing probabilities were clipped again after the weighted
            # average. Add that deterministic rounding/clipping correction to
            # every view so the original coefficients reproduce exactly.
            v += (X[:, member_i] - weights @ v)[None]
            t += (T[:, member_i] - weights @ t)[None]
            assert np.max(np.abs(weights @ v - X[:, member_i])) < 1e-10
            assert np.max(np.abs(weights @ t - T[:, member_i])) < 1e-10
            cols.extend(list(v))
            tcols.extend(list(t))
            labels.extend([f"{name}::{k}" for k in kinds])
        groups.append((start, len(cols), weights))
    return np.column_stack(cols), np.column_stack(tcols), labels, groups, hashes


def fit_views(X, y, groups, init, penalty):
    m = X.shape[1]
    Q = np.zeros((m, m))
    for start, end, w in groups:
        one = np.ones(end - start)
        P = np.eye(end - start) - w[:, None] * one[None, :]
        Q[start:end, start:end] = np.outer(one, one) + penalty * (P.T @ P)
    initial = np.r_[np.concatenate([init[k] * w for k, (_, _, w) in enumerate(groups)]), init[-1]]
    def f(th):
        b = th[:-1]
        z = X @ b + th[-1]
        r = expit(z) - y
        q = Q @ b
        cost = .1 * np.sum(np.logaddexp(0, z) - y * z) + .5 * b @ q
        grad = np.r_[.1 * X.T @ r + q, .1 * r.sum()]
        return cost, grad
    opt = minimize(f, initial, jac=True, method="L-BFGS-B", bounds=[(0, None)] * m + [(None, None)],
                   options={"maxiter": 2000, "ftol": 1e-11, "gtol": 1e-6})
    assert opt.success, str(opt.message)
    return opt.x


def main():
    try:
        KU_ENV.low_priority()
    except (AttributeError, OSError):
        pass
    start = time.time()
    names, ids, tids, y, wt, X, T, base = load_data(ROOT / "submissions/sub_seqw5_stack.json")
    V, VT, labels, groups, hashes = expand(names, X, T)
    assert np.isfinite(V).all() and np.isfinite(VT).all()
    penalties = [10., 100., 1000.]
    results, saved = {}, {}
    for seed in [7, 29]:
        pred = {"baseline": np.zeros(len(y)), **{f"view_{p:g}": np.zeros(len(y)) for p in penalties}}
        for f, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)):
            init = logistic_fit(X[tr], y[tr])
            pred["baseline"][va] = X[va] @ init[:-1] + init[-1]
            for penalty in penalties:
                th = fit_views(V[tr], y[tr], groups, init, penalty)
                pred[f"view_{penalty:g}"][va] = V[va] @ th[:-1] + th[-1]
            print(f"View weights: split {seed} fold {f} ({time.time()-start:.0f}s)", flush=True)
        results[str(seed)] = {key: score(y, p, wt) for key, p in pred.items()}
        saved.update({f"s{seed}_{key}": p for key, p in pred.items()})
        print(json.dumps({"seed": seed, "results": results[str(seed)]}), flush=True)
        (OUT / "view_results.json").write_text(json.dumps(results, indent=2))
    np.savez(OUT / "view_oof.npz", **saved, y=y, ids=np.asarray(ids, dtype=str), sample_weight=wt)
    init = logistic_fit(X, y)
    coefficients, tests = {}, {}
    for penalty in penalties:
        th = fit_views(V, y, groups, init, penalty)
        key = f"view_{penalty:g}"
        coefficients[key] = th.tolist()
        tests[key] = expit(VT @ th[:-1] + th[-1])
    artifact = {"labels": labels, "weights_and_intercepts": coefficients,
                "source_members": names, "expanded_groups": GROUPS,
                "source_view_hashes": hashes, "results": results,
                "validation_scope": "Exploratory meta CV on saved OOF base predictions; not fully nested base retraining",
                "status": "experimental; select using documented held-out comparisons"}
    wd = ROOT / "weights_experiments/weight_edit_20261104"
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "view_coefficients.json").write_text(json.dumps(artifact, indent=2))
    np.savez(OUT / "view_test.npz", **tests, ids=np.asarray(tids, dtype=str))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
