"""How a raw series becomes features: simplified versions of what try4 computes.

Run from C:/Users/Blitz/Downloads/Code/claude/ai:
    .venv/Scripts/python.exe explainer/feature_demo.py
"""
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

series = pd.read_parquet("data/train_series.parquet")
labels = pd.read_csv("data/train_labels.csv").set_index("id")["label"]

# grid for the Bayes-factor feature: the unknown mean and noise scale are averaged over, not fitted
MU = np.linspace(-3, 3, 61)            # mean, in units of the series' own robust scale
LOGS = np.linspace(-2.5, 1.5, 81)      # log noise scale


def loglik(x, mu, logs):
    """Gaussian log-likelihood of values x for every (mu, log-scale) pair on the grid."""
    n, s1, s2 = len(x), x.sum(), (x * x).sum()
    ss = s2 - 2 * mu[:, None] * s1 + n * mu[:, None] ** 2          # sum of (x - mu)^2
    return -0.5 * n * np.log(2 * np.pi) - n * logs[None, :] - ss / (2 * np.exp(2 * logs[None, :]))


def free_scale_evidence(x, j):
    """log Bayes factor: 'each side has its own noise scale' vs 'one shared scale'.
    Simplified version of try4's native_law_state__native_free_scale (which also tries 39 noise shapes)."""
    c = np.median(x); s = np.median(np.abs(x - c)) * 1.4826 + 1e-9
    z = (x - c) / s                                                  # standardise robustly
    a, b = loglik(z[:j], MU, LOGS), loglik(z[j:], MU, LOGS)          # [mu, scale] grids for each side
    lp = -np.log(len(MU)) - np.log(len(LOGS))                        # flat prior over the grid
    shared = logsumexp(a + b + lp)                                   # one mean, one scale for both sides
    free = logsumexp(logsumexp(a - np.log(len(LOGS)), 1) + logsumexp(b - np.log(len(LOGS)), 1) - np.log(len(MU)))  # one mean, two scales
    return free - shared


def features(one):
    """One series in, one row of features out."""
    x = one["value"].to_numpy(); j = int((one["period"] == 0).sum())
    pre, post = x[:j], x[j:]
    f = {}
    # 1. simple statistics (try1-style)
    f["mean_shift"] = abs(post.mean() - pre.mean()) / pre.std()
    f["log_std_ratio"] = abs(np.log(post.std() / pre.std()))
    # 2. a classical two-sample test: do before and after look like the same distribution?
    f["ks_stat"] = ks_2samp(pre, post).statistic
    # 3. memory: did the pull from one value to the next change?
    ac = lambda v: np.corrcoef(v[:-1], v[1:])[0, 1]
    f["lag1_autocorr_change"] = abs(ac(post) - ac(pre))
    # 4. Bayesian evidence for a change in noise volume AT the marked point
    f["free_scale_bf"] = free_scale_evidence(x, j)
    # 5. is the evidence strongest exactly at the marked point? (like boundary_law any_far_strong)
    far = [free_scale_evidence(x, k) for k in (len(x) // 5, len(x) * 4 // 5)]
    f["at_minus_far"] = f["free_scale_bf"] - max(far)
    return f


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    demo = ["5e5be0ca665e101471c693293110c5fc", "a8f08afc6687aac2e6132c6a6085e8e5"]
    rows = {i: features(series[series.id == i].sort_values("time")) for i in demo}
    table = pd.DataFrame(rows).T
    table.insert(0, "label", labels.loc[demo].to_numpy())
    print("Two real series -> one row of features each:\n")
    print(table.round(3).to_string())

    print("\nNow every training series (takes a minute or two)...")
    allf = pd.DataFrame({i: features(g) for i, g in series.sort_values(["id", "time"]).groupby("id", sort=False)}).T
    y = labels.loc[allf.index].to_numpy()
    print("\nHow well each feature separates breaks from no-breaks on its own (AUC, 0.5 = coin flip):")
    for c in allf.columns:
        v = allf[c].fillna(allf[c].median()).to_numpy()
        print(f"  {c:22s} {max(roc_auc_score(y, v), 1 - roc_auc_score(y, v)):.3f}")
