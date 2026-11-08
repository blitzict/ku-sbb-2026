# CPU port of experiments/weight_edit_20261104/neural_head_edit.py, source sha256 3083b9ce708095e5b3c61aadb34bb94ec82519094fe524c505553c0528afb38e. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV; import rank_gauss_bounded as RB
"""Edit saved MLP output weights, with selection inside each original fold.

Rebuild exact rank-Gaussian preprocessing from the original source matrix. The
stored rank references were subsampled and cannot reproduce original inference.
Frozen encoder and BatchNorm statistics; only net.12 weight/bias may change.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import argparse
import ctypes
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

ROOT = KU_ENV.root()
OUT = ROOT / "tune_out/weight_edit_20261104"
WD = ROOT / "weights_experiments/weight_edit_20261104/neural_head"
KINDS = ["orig", "rev", "flip", "revflip", "c1", "c2"]
VIEW_WEIGHTS = np.array([2, 1, 2, 1, .5, .5]) / 7


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        layers = []
        for _ in range(3):
            layers.extend([nn.Linear(dim, 1024), nn.BatchNorm1d(1024), nn.SiLU(), nn.Dropout(.4)])
            dim = 1024
        self.net = nn.Sequential(*layers, nn.Linear(dim, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def rank_reference(rows):
    t = torch.as_tensor(rows, device="cpu")
    t = torch.where(torch.isnan(t), torch.full_like(t, float("inf")), t)
    srt = torch.sort(t, 0).values.T.contiguous()
    nv = torch.isfinite(srt).sum(1, keepdim=True).clamp(min=1).float()
    return srt, nv


def transform(rows, srt, nv):
    z = torch.as_tensor(rows, device="cpu")
    nan = torch.isnan(z)
    zs = torch.where(nan, torch.zeros_like(z), z).T.contiguous()
    u = ((torch.searchsorted(srt, zs).float() + torch.searchsorted(srt, zs, right=True).float()) / 2 + .5) / (nv + 1)
    g = (torch.erfinv(2 * u.clamp(1e-4, 1 - 1e-4) - 1) * 2 ** .5).T
    return torch.where(nan, torch.zeros_like(g), g)


def edit_head(H, y, original, core, inner, seed):
    """Pairwise logistic head adaptation; choose lambda/blend using inner only."""
    original_w, original_b = original
    htr = H[torch.as_tensor(core, device="cpu")]
    scale = htr.std(0).clamp(min=.03)
    x = htr / scale
    b0 = htr @ original_w + original_b
    hin = H[torch.as_tensor(inner, device="cpu")]
    base_inner = (hin @ original_w + original_b).cpu().numpy()
    base_auc = roc_auc_score(y[inner], base_inner)
    best_auc, best_alpha, best_lam = base_auc, 0., None
    best_delta = np.zeros(H.shape[1], np.float64)
    rng = np.random.default_rng(seed)
    pos = torch.as_tensor(rng.choice(np.flatnonzero(y[core] == 1), 262144), device="cpu")
    neg = torch.as_tensor(rng.choice(np.flatnonzero(y[core] == 0), 262144), device="cpu")
    trials = []
    for lam in (.03, .3, 3.):
        def objective(delta):
            d = torch.tensor(delta, device="cpu", dtype=torch.float32, requires_grad=True)
            z = b0 + x @ d
            loss = torch.nn.functional.softplus(-(z[pos] - z[neg])).mean() + lam * (d * d).sum()
            loss.backward()
            return float(loss.detach().cpu()), d.grad.detach().cpu().numpy().astype(np.float64)
        fit = minimize(objective, np.zeros(H.shape[1]), jac=True, method="L-BFGS-B",
                       options={"maxiter": 160, "gtol": 2e-6, "ftol": 2e-9, "maxls": 30})
        delta = torch.as_tensor(fit.x, dtype=torch.float32, device="cpu") / scale
        change = (hin @ delta).cpu().numpy()
        for alpha in (.25, .5, 1.):
            auc = float(roc_auc_score(y[inner], base_inner + alpha * change))
            trials.append({"lambda": lam, "alpha": alpha, "inner_auc": auc,
                           "optimizer_success": bool(fit.success), "optimizer_message": str(fit.message)})
            if fit.success and auc > best_auc + .0003:
                best_auc, best_alpha, best_lam = auc, alpha, lam
                best_delta = fit.x.copy()
    new_w = original_w + best_alpha * torch.as_tensor(best_delta, dtype=torch.float32, device="cpu") / scale
    # Pairwise loss leaves the intercept unidentified. Preserve the original
    # mean probability on core rows, including its augmentation-induced prior,
    # so edited and unedited folds retain comparable calibration. Matching y's
    # prior only for edited folds would introduce inconsistent cross-fold offsets.
    train_z = (htr @ new_w).cpu().numpy()
    original_mean = float(torch.sigmoid(b0).mean().cpu())
    bias = float(original_b.cpu())
    if best_alpha:
        for _ in range(20):
            p = expit(train_z + bias)
            step = (p - original_mean).sum() / max(float((p * (1 - p)).sum()), 1e-8)
            bias -= float(np.clip(step, -2, 2))
            if abs(step) < 1e-8:
                break
    return new_w, bias, {"inner_baseline": float(base_auc), "inner_selected": float(best_auc),
                         "alpha": best_alpha, "lambda": best_lam, "trials": trials}


def report(y, p, w):
    return {"auc": float(roc_auc_score(y, p)), "weighted_auc": float(roc_auc_score(y, p, sample_weight=w))}


def main():
    global OUT, WD
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--source", default="MLPHV_big")
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--run-name", default="head_v2_MLPHV_big")
    ap.add_argument("--edit-seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    args = ap.parse_args()
    assert args.run_name.replace('_', '').isalnum()
    assert len(set(args.edit_seeds)) == len(args.edit_seeds) and all(s in range(4) for s in args.edit_seeds)
    OUT = OUT / args.run_name
    WD = WD / args.run_name
    try:
        KU_ENV.low_priority()
    except (AttributeError, OSError):
        pass
    torch.set_num_threads(KU_ENV.threads())
    OUT.mkdir(parents=True, exist_ok=True)
    WD.mkdir(parents=True, exist_ok=True)
    started = time.time()
    path = ROOT / "tune_out/var/mat_cggd_hv.npz"
    matrix_hash = digest(path)
    z = np.load(path, allow_pickle=True)
    X = {"orig": z["X"], **{k: z[f"V_{k}"] for k in KINDS[1:]}}
    T = {"orig": z["Xt"], **{k: z[f"VT_{k}"] for k in KINDS[1:]}}
    A, E, ey, esrc = z["A"], z["E"], z["Ey"], z["Esrc"]
    y, ids, tids, src, wt = (z[k] for k in ["y", "ids", "tid", "src", "w"])
    index = pd.read_parquet(ROOT / "clean/features_train.parquet", columns=[]).index
    target = pd.read_csv(ROOT / "data/train_labels.csv", dtype={"id": str}).set_index("id").loc[index, "label"].to_numpy()
    assert np.array_equal(ids, index.to_numpy()) and np.array_equal(y, target)
    source_index = pd.Index(ids)
    source_rows = source_index.get_indexer(src)
    extra_rows = source_index.get_indexer(esrc)
    assert (source_rows >= 0).all() and (extra_rows >= 0).all(), "Unidentified augmented source"
    archived = np.load(ROOT / "tune_out" / f"ver_{args.source}.npz", allow_pickle=False)
    assert np.array_equal(archived["kinds"], KINDS)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(X["orig"], y))
    print(f"Loaded matrix: {len(ids)} train, {len(tids)} test, {X['orig'].shape[1]} features; GPU {"cpu"} ({time.time()-started:.0f}s)", flush=True)
    for fold in args.folds:
        part = OUT / f"head_fold{fold}.npz"
        if args.resume and part.exists():
            print(f"Reusing completed fold {fold}", flush=True)
            continue
        tr, va = folds[fold]
        perm = np.random.RandomState(fold + 100 * args.seed_offset).permutation(tr)
        inner, core = perm[:len(tr) // 10], perm[len(tr) // 10:]
        in_core = np.zeros(len(ids), dtype=bool)
        in_core[core] = True
        ashift, ex = in_core[source_rows], in_core[extra_rows]
        assert not set(ids[core]).intersection(ids[va]) and not set(ids[inner]).intersection(ids[va])
        # Same rows as np.vstack([...]).astype(np.float32), sorted column block by column block straight from the sources, so the
        # training block is never materialised. Bit-identical srt/nv use (auc_lab_20261107/claude/audit/test_rank_bounded.py).
        rows = [(X["orig"], core), (A, ashift), *[(X[k], core) for k in KINDS[1:]], (E, ex)]
        print(f"Fold {fold}: reconstructing exact transform on {RB.rows_of(rows)} training/augmentation rows ({time.time()-started:.0f}s)", flush=True)
        srt, nv = RB.sorted_reference(rows)
        del rows
        gc.collect()
        models, states, hashes = [], [], []
        for seed in range(4):
            checkpoint = ROOT / "weights" / args.source / f"fold{fold}_seed{seed}.pt"
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            assert all(torch.isfinite(v).all() for v in state.values())
            m = MLP(X["orig"].shape[1]).to("cpu")
            m.load_state_dict(state)
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
            states.append(state)
            models.append(m)
            hashes.append(digest(checkpoint))
        n = len(y) + len(tids)
        hidden = {seed: torch.zeros((n, 1024), device="cpu") for seed in args.edit_seeds}
        plain = np.zeros((4, 6, n), dtype=np.float32)
        for view_idx, kind in enumerate(KINDS):
            for off, data in [(0, X[kind]), (len(y), T[kind])]:
                for start in range(0, len(data), 8192):
                    end = min(len(data), start + 8192)
                    batch = transform(data[start:end], srt, nv)
                    with torch.no_grad(), torch.autocast("cpu", enabled=False):
                        for seed, model in enumerate(models):
                            h = model.net[:-1](batch)
                            pred = model.net[-1](h).squeeze(-1).float()
                            plain[seed, view_idx, off + start:off + end] = pred.cpu().numpy()
                            if seed in hidden:
                                hidden[seed][off + start:off + end] += float(VIEW_WEIGHTS[view_idx]) * h.float()
            print(f"Fold {fold}: encoded {kind} ({time.time()-started:.0f}s)", flush=True)
        expected = archived["oof"][:, va]
        reproduced = expit(plain[:, :, va].mean(0))
        err = np.abs(reproduced - expected)
        print(f"Fold {fold}: archived prediction reproduction max={err.max():.3g}, mean={err.mean():.3g}", flush=True)
        # BF16 backend matmul kernels can differ slightly with batch sizes.
        # A material mismatch must stop all edits before scores are claimed.
        assert err.mean() < 2e-4 and err.max() < .01, "Checkpoint/preprocessing reproduction failed"
        edited, original, edits = [], [], []
        for seed in args.edit_seeds:
            w0 = models[seed].net[-1].weight.detach().flatten().float()
            b0 = models[seed].net[-1].bias.detach().flatten()[0].float()
            bpred = (hidden[seed] @ w0 + b0).cpu().numpy()
            wnew, bnew, info = edit_head(hidden[seed][:len(y)], y, (w0, b0), core, inner, 20261104 + 100 * fold + seed)
            pred = (hidden[seed] @ wnew + bnew).cpu().numpy()
            new_state = {k: v.clone() for k, v in states[seed].items()}
            new_state["net.12.weight"] = wnew.detach().cpu()[None]
            new_state["net.12.bias"] = torch.tensor([bnew])
            assert all(torch.equal(new_state[k], states[seed][k]) for k in states[seed] if k not in ["net.12.weight", "net.12.bias"])
            torch.save(new_state, WD / f"fold{fold}_seed{seed}.pt")
            info.update({"fold": fold, "seed": seed, "source_sha256": hashes[seed],
                         "outer_baseline": report(y[va], bpred[va], wt[va]),
                         "outer_edited": report(y[va], pred[va], wt[va]),
                         "weight_delta_l2": float(torch.linalg.vector_norm(wnew - w0).cpu())})
            edits.append(info)
            original.append(bpred)
            edited.append(pred)
            print(f"Fold {fold} seed {seed}: original={info['outer_baseline']['auc']:.6f} edited={info['outer_edited']['auc']:.6f} inner-selected lambda={info['lambda']} alpha={info['alpha']} ({time.time()-started:.0f}s)", flush=True)
        np.savez(part, va=va, original=np.stack(original)[:, va], edited=np.stack(edited)[:, va],
                 original_test=np.stack(original)[:, len(y):], edited_test=np.stack(edited)[:, len(y):],
                 replay_max_error=float(err.max()), replay_mean_error=float(err.mean()),
                 metadata=json.dumps(edits))
        (WD / f"fold{fold}.json").write_text(json.dumps({"matrix_sha256": matrix_hash,
            "matrix": str(path.relative_to(ROOT)), "source_tag": args.source,
            "seed_offset": args.seed_offset, "calibration": "Preserve core mean probability of original head",
            "preprocessing": "Exact original training-fold rank transform; reconstruct using neural_head_edit.py.",
            "view_weights": dict(zip(KINDS, VIEW_WEIGHTS.tolist())), "edits": edits}, indent=2))
        del models, states, hidden, srt, nv
        gc.collect()
        pass
    if all((OUT / f"head_fold{f}.npz").exists() for f in range(5)):
        original = np.zeros((len(args.edit_seeds), len(y)))
        edited = np.zeros_like(original)
        ot, et, details = [], [], []
        for fold in range(5):
            p = np.load(OUT / f"head_fold{fold}.npz", allow_pickle=False)
            original[:, p["va"]] = p["original"]
            edited[:, p["va"]] = p["edited"]
            ot.append(p["original_test"])
            et.append(p["edited_test"])
            details.extend(json.loads(str(p["metadata"])))
        scores = {f"seed{s}": {"original": report(y, original[i], wt), "edited": report(y, edited[i], wt)} for i, s in enumerate(args.edit_seeds)}
        scores["seed_average"] = {"original": report(y, original.mean(0), wt), "edited": report(y, edited.mean(0), wt)}
        scores["method"] = "Frozen MLP encoder; pairwise logistic final-layer update; lambda/blend selected only on original inner split"
        scores["folds"] = details
        scores["matrix_sha256"] = matrix_hash
        scores["source"] = args.source
        scores["seed_offset"] = args.seed_offset
        scores["calibration"] = "Preserve original mean probability on core training rows"
        np.savez(OUT / "head_oof.npz", original=original, edited=edited,
                 original_test=np.mean(ot, axis=0), edited_test=np.mean(et, axis=0),
                 ids=ids.astype(str), test_ids=tids.astype(str), y=y, sample_weight=wt)
        (OUT / "head_results.json").write_text(json.dumps(scores, indent=2))
        print(json.dumps({k: v for k, v in scores.items() if k not in ["folds", "matrix_sha256"]}), flush=True)
    print(f"Completed requested folds in {time.time()-started:.0f}s", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
