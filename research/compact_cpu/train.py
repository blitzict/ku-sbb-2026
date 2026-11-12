"""Train the KU-SBB 0.879114 ensemble output layer on a single CPU thread.

Run: python train.py
Requires: numpy, scipy, scikit-learn, pandas, threadpoolctl; training_inputs.npz
belongs beside this script. Paths resolve relative to the script, so the pair
can be copied to another directory. Use --out-dir to change the output location.

Each example is a whole time series. The input pack contains its binary label,
earlier detectors' log-odds, and their transformed features. Fold-specific
features were prepared with training-side statistics. This script learns new
coefficients from zero five times for validation and once for the test set.
Earlier tree and neural-network checkpoints were trained in the original
pipeline. See method.html for the raw features and prepare_inputs.py for the
exact path from those checkpoints' archived outputs to this training pack.
"""
import argparse
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
REFERENCE_AUC = 0.8791142083333333
REFERENCE_SHA256 = "81be6dbdbcd0202876c0991950664e1f60ab34102fbbe63dec6f5ec254ca5064"


def fit_weights(x, base, labels):
    """Fit a small correction to base log-odds using labels and close ranking pairs."""
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    negatives = negatives[np.argsort(base[negatives])]
    left = np.searchsorted(base[negatives], base[positives] - 1.35)
    right = np.searchsorted(base[negatives], base[positives] + 1.35)
    has_neighbor = right > left
    count = (right - left)[has_neighbor]
    positives, left = positives[has_neighbor], left[has_neighbor]
    if not len(positives):
        raise ValueError("The training fold has no nearby positive–negative pairs.")

    # Sample 32 negatives across each positive's neighborhood. Sampling is
    # deterministic. The count factor preserves the relative mass of dense regions.
    rng = np.random.default_rng(1729)
    grid = (np.arange(32) + rng.random((len(positives), 1))) / 32
    i = np.repeat(positives, 32)
    j = negatives[(left[:, None] + (count[:, None] * grid).astype(int)).ravel()]
    pair_mass = np.repeat(count, 32) * np.exp(-0.5 * ((base[i] - base[j]) / 0.62) ** 2)
    pair_mass *= len(labels) * 0.53 / pair_mass.sum()

    def objective(w):
        scores = base + x @ w
        gap = (scores[j] - scores[i]) / 0.55
        # A misordered pair has positive gap. The two bincounts accumulate
        # its opposite contributions to the negative and positive examples.
        derivative = pair_mass * expit(gap)
        row_gradient = (0.47 * (expit(scores) - labels)
                        + np.bincount(j, weights=derivative, minlength=len(scores))
                        - np.bincount(i, weights=derivative, minlength=len(scores)))
        loss = (0.47 * np.sum(np.logaddexp(0, scores) - labels * scores)
                + 0.55 * pair_mass @ np.logaddexp(0, gap) + 84 * (w @ w))
        return loss, x.T @ row_gradient + 168 * w

    result = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B",
                      options={"maxiter": 300, "ftol": 1e-11})
    if not result.success:
        raise RuntimeError(f"Weight optimization failed: {result.message}")
    return 0.5 * result.x  # The confirmed recipe applies half of the fitted update.


def train_split(pack, name, labels):
    """Train on the leading rows and predict the trailing rows in the same basis."""
    train = pack[f"{name}_train"]
    query = pack[f"{name}_query"]
    x = pack[f"{name}_features"]
    base = pack[f"{name}_base"]
    size = len(train)
    if (x.ndim != 2 or len(x) != size + len(query) or base.shape != (len(x),)
            or not np.isfinite(x).all() or not np.isfinite(base).all()):
        raise ValueError(f"Invalid design matrix or base scores in {name}.")
    if name != "full":
        if not np.array_equal(np.sort(np.r_[train, query]), np.arange(len(labels))):
            raise ValueError(f"Overlapping or missing sources in {name}.")
    elif not np.array_equal(train, np.arange(len(labels))):
        raise ValueError("The final fit is missing training sources.")
    weights = fit_weights(x[:size], base[:size], labels[train])
    return query, base[size:] + x[size:] @ weights, weights


def run(source, out_dir):
    with np.load(source, allow_pickle=False) as pack:
        ids, y = pack["ids"], pack["labels"]
        if (len(ids) != 24000 or len(np.unique(ids)) != len(ids) or y.shape != ids.shape
                or set(np.unique(y)) != {0, 1}):
            raise ValueError("Expected the 24,000 uniquely identified, labelled training series.")
        oof = np.full(len(y), np.nan)
        for fold in range(5):
            rows, scores, _ = train_split(pack, f"fold{fold}", y)
            if np.isfinite(oof[rows]).any():
                raise ValueError("A source appears in more than one validation fold.")
            oof[rows] = scores
            print(f"Fold {fold + 1}/5  AUC {roc_auc_score(y[rows], scores):.8f}", flush=True)
        auc = roc_auc_score(y, oof)  # Pool all held-out scores before computing AUC.
        print(f"Primary out-of-fold AUC  {auc:.8f}", flush=True)
        if abs(auc - REFERENCE_AUC) > 1e-7:
            raise RuntimeError(f"Expected {REFERENCE_AUC:.8f}. Check the input pack and environment.")
        query, scores, weights = train_split(pack, "full", y)
        test_ids, order = pack["test_ids"], pack["submission_order"]
        if (len(test_ids) != 4000 or len(np.unique(test_ids)) != 4000
                or not np.array_equal(query, len(y) + np.arange(4000))
                or not np.array_equal(np.sort(order), np.arange(4000)) or not np.isfinite(scores).all()):
            raise ValueError("Invalid test scores or sample-submission order.")
        frame = pd.DataFrame({"id": test_ids[order], "break_probability": expit(scores)[order]})
    out_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out_dir) as tmp:
        csv = Path(tmp) / "submission.csv"
        csv_path = out_dir / csv.name
        frame.to_csv(csv, index=False, lineterminator="\r\n")
        digest = hashlib.sha256(csv.read_bytes()).hexdigest()
        artifact = Path(tmp) / "fit.npz"
        np.savez_compressed(artifact, ids=ids, oof_logits=oof, weights=weights, auc=auc)
        artifact.replace(out_dir / artifact.name)
        csv.replace(csv_path)
    print(f"Saved {len(frame):,} test scores to {csv_path}")
    print(f"Submission SHA-256  {digest}")
    print("Byte-identical to the confirmed CSV." if digest == REFERENCE_SHA256 else
          "CSV bytes differ from the reference. Numerical library versions can affect the final decimals.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs", type=Path, default=HERE / "training_inputs.npz", help="Portable training pack")
    parser.add_argument("--out-dir", type=Path, default=HERE / "run", help="Directory for submission.csv and fit.npz")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.inputs, args.out_dir)
