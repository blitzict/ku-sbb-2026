"""build the 0.879 report: figures (matplotlib) -> report_0879.html -> report_0879.pdf (headless chrome).

python qualification/build_report_0879.py
"""
import re
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
INK, ACC, MUT, LIGHT = "#18332e", "#19785d", "#7c918a", "#e7f2ed"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUT, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})

#feature families, grouped (columns per table from the cpu rebuild, id column excluded)
FAMILIES = [
    ("Base statistics", "moments, quantiles, ranks, KS/AD/MMD tests, autocorrelation, spectra", 977),
    ("Window grids", "the same comparisons on two fixed window grids around the boundary", 492 + 504),
    ("Increments & change tests", "differenced series, likelihood ratios (raw + z), Bayes factors, self-normalised", 198 + 92 + 100 + 26 + 17),
    ("Trend, volatility & spectrum fits", "linear trend, GARCH, power spectrum", 21 + 30 + 22),
    ("Periodic & phase", "periodic components, harmonic fit, phase likelihood", 37 + 36 + 18),
    ("Dynamics & memory", "innovations, AR law, trend/curve, long memory, Markov copula, mode residual", 60 + 21 + 21 + 20 + 81 + 72 + 13),
    ("Jumps, pulses & latent states", "5 jump-law nulls, pulse/state filters, walk-vs-jump, boundary laws, latent level", 5 * 149 + 12 + 21 + 44 + 114 + 40 + 66 + 204),
    ("Noise, tails & distribution laws", "noise tests, bounds, mixtures, partition, tails, quantile alignment, native laws, volatility laws", 84 + 26 + 15 + 48 + 56 + 42 + 556 + 50 + 48 + 32 + 26),
]


def fig_series():
    #synthetic example (no competition data in the report): level + volatility break, and the views built from it
    rng = np.random.default_rng(3)
    n, b = 600, 340
    x = np.r_[rng.normal(0, 1, b), rng.normal(.6, 1.6, n - b)]
    x = np.convolve(x, [.6, .3, .1], "same")
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 1.35), sharey=True)
    lo, hi = int(b * .25), int(b + (n - b) * .7)
    views = [("original", x, b), ("reversed", x[::-1], n - b), ("sign flip", -x, b), ("crop (c1-c4)", x[lo:hi], b - lo)]
    for a, (t, v, bb) in zip(ax, views):
        a.plot(v[:bb], lw=.5, c=INK)
        a.plot(np.arange(bb, len(v)), v[bb:], lw=.5, c=ACC)
        a.axvline(bb, c="#c0392b", ls="--", lw=.8)
        a.set_title(t, fontsize=8, color=INK)
        a.set_xticks([])
    ax[0].set_ylabel("value")
    fig.tight_layout(pad=.3)
    fig.savefig(FIG / "views.png", dpi=220)
    plt.close(fig)


def box(ax, x, y, w, h, title, sub, fc=LIGHT):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015", fc=fc, ec=ACC, lw=.8))
    ax.text(x + w / 2, y + h * .66, title, ha="center", va="center", fontsize=7.6, weight="bold", color=INK)
    ax.text(x + w / 2, y + h * .28, sub, ha="center", va="center", fontsize=6.2, color=INK)


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.4, 1.75))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    cols = [
        ("Raw series", "24,000 train\n4,000 test\nsplit at boundary"),
        ("Views", "orig + rev, flip,\nrevflip, 4 crops\n+ shift negatives"),
        ("Feature families", "8 groups\n~5,100 columns\n(+ per-view copies)"),
        ("Detectors", "51 base members\nXGB / LGB / Cat /\nMLP / seq-CNN\n+ mechanism trees"),
        ("Weight heads", "view stack, fusion,\nwhite-noise experts,\nlocal-rank correction"),
        ("Submission", "id,\nbreak_probability"),
    ]
    w, gap = .145, .026
    for j, (t_, s_) in enumerate(cols):
        x = j * (w + gap)
        ax.add_patch(FancyBboxPatch((x, .2), w, .72, boxstyle="round,pad=0.005,rounding_size=0.02",
                                    fc=ACC if j == 5 else LIGHT, ec=ACC, lw=.8))
        ax.text(x + w / 2, .8, t_, ha="center", va="center", fontsize=7, weight="bold", color="white" if j == 5 else INK)
        ax.text(x + w / 2, .45, s_, ha="center", va="center", fontsize=5.9, color="white" if j == 5 else INK, linespacing=1.25)
        if j < 5:
            ax.annotate("", (x + w + gap - .003, .56), (x + w + .003, .56), arrowprops=dict(arrowstyle="-|>", color=ACC, lw=1))
    ax.text(.5, .04, "473 recorded steps (pipeline/full_cpu_stages.json); each layer only sees out-of-fold predictions of the layer below",
            ha="center", fontsize=6.2, color=MUT)
    fig.savefig(FIG / "pipeline.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_families():
    names = [f[0] for f in FAMILIES][::-1]
    cols = [f[2] for f in FAMILIES][::-1]
    fig, ax = plt.subplots(figsize=(3.5, 2.35))
    #reversed order: index 0..3 = mechanism families (bottom), 4..7 = statistical
    bars = ax.barh(names, cols, color=[ACC] * 4 + ["#9cc9b6"] * 4)
    for r, c in zip(bars, cols):
        ax.text(c + 25, r.get_y() + r.get_height() / 2, f"{c:,}", va="center", fontsize=6.8, color=INK)
    ax.set_xlabel("columns")
    ax.set_xlim(0, max(cols) * 1.2)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout(pad=.3)
    fig.savefig(FIG / "families.png", dpi=220)
    plt.close(fig)


def fig_ladder():
    steps = [("grid\nXGB", .840), ("+ns3\nblocks", .845), ("+7\nviews", .850), ("+harm-\nonic", .859),
             ("+mech.\ncrops", .8745), ("full\nrecipe", .8791)]
    fig, ax = plt.subplots(figsize=(3.5, 2.35))
    x = np.arange(len(steps))
    v = [s[1] for s in steps]
    ax.plot(x, v, c=ACC, lw=1.2, marker="o", ms=4)
    for i, val in enumerate(v):
        ax.text(i, val + .0012, f"{val:.3f}", ha="center", fontsize=6.6, color=INK)
    ax.axhline(.86987, c="#c98a1b", ls=":", lw=1)
    ax.text(5.2, .8675, "try2 (0.870)", fontsize=6.3, color="#c98a1b", ha="right")
    ax.axhline(.83835, c=MUT, ls=":", lw=1)
    ax.text(5.2, .8360, "try1 (0.838)", fontsize=6.3, color=MUT, ha="right")
    ax.set_xticks(x, [s[0] for s in steps], fontsize=6.2)
    ax.set_ylim(.835, .885)
    ax.set_ylabel("out-of-fold ROC-AUC")
    fig.tight_layout(pad=.3)
    fig.savefig(FIG / "ladder.png", dpi=220)
    plt.close(fig)


def fig_scores():
    names = ["try1", "try2", "try3\nensemble", "try3 +\nmixed MLP", "try4\nselected"]
    local = [.83835, .86987, .87911, .87910, .87235]
    public = [.83313, .87539, .88077, .88078, .87583]
    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    x = np.arange(5)
    ax.bar(x - .18, local, .34, color="#9cc9b6", label="local OOF")
    ax.bar(x + .18, public, .34, color=ACC, label="Kaggle public")
    for i in range(5):
        ax.text(i + .18, public[i] + .001, ["0.83313", "0.87539", "0.88077", "0.88078", "0.87583"][i], ha="center", fontsize=5.8)
    ax.set_xticks(x, names, fontsize=6)
    ax.set_ylim(.82, .90)
    ax.legend(fontsize=6.3, frameon=False, loc="upper left", ncol=2)
    ax.set_ylabel("ROC-AUC")
    fig.tight_layout(pad=.3)
    fig.savefig(FIG / "scores.png", dpi=220)
    plt.close(fig)


def main():
    fig_series(); fig_pipeline(); fig_families(); fig_ladder(); fig_scores()
    html = (HERE / "report_0879_template.html").read_text(encoding="utf8")
    rows = "\n".join(f"<tr><td><b>{n}</b></td><td>{d}</td><td class=r>{c:,}</td></tr>" for n, d, c in FAMILIES)
    html = html.replace("{{FAMILY_ROWS}}", rows).replace("{{FAMILY_TOTAL}}", f"{sum(f[2] for f in FAMILIES):,}")
    out = HERE / "report_0879.html"
    out.write_text(html, encoding="utf8")
    chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    pdf = HERE / "report_0879.pdf"
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", out.as_uri()],
                   check=True, capture_output=True)
    pages = len(re.findall(rb"/Type\s*/Page[^s]", pdf.read_bytes()))
    print("pages", pages)
    if not 2 <= pages <= 3:
        raise SystemExit(f"report must be 2-3 pages, got {pages}")


if __name__ == "__main__":
    main()
