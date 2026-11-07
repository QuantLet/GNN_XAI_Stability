"""Plot paired IEEE-CIS contrasts and check the sensitivity of feature weights."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.instance_level_refitlevel import _topk_overlap


def paired_estimate(values):
    mean = np.mean(values)
    half = stats.t.ppf(0.975, len(values) - 1) * stats.sem(values)
    return mean, mean - half, mean + half


def draw(directory, figures):
    paired = pd.read_csv(directory / "paired.csv").set_index("seed")
    if paired.index.tolist() != list(range(201, 221)):
        raise ValueError("Expected 20 complete seed pairs")
    reference = pd.read_csv(directory / "wasserstein_by_seed.csv").set_index("seed")
    rows = []
    for seed, original in paired.iterrows():
        row = dict(seed=seed)
        with np.load(directory / f"{seed}_random.npz") as r, np.load(directory / f"{seed}_temporal.npz") as t:
            for split, data in (("random", r), ("temporal", t)):
                tab = data["tab"]
                if tab.shape != (5000, 486):
                    raise ValueError("Unexpected SHAP dimensions")
                for arm in ("hyb", "cov"):
                    row[f"{split}_{arm}"] = 10 * _topk_overlap(tab, data[arm][:, :486])
                np.testing.assert_allclose(row[f"{split}_hyb"] - row[f"{split}_cov"],
                                           10 * original[f"top10_{split}"], atol=1e-10)
            mass, gap = {}, {}
            for arm in ("tab", "hyb"):
                a, b = r[arm][:, :486], t[arm][:, :486]
                mass[arm] = (np.abs(a).mean(0) + np.abs(b).mean(0)) / 2
                gap[arm] = np.abs(np.sort(a, axis=0) - np.sort(b, axis=0)).mean(0)
                normalized = np.divide(gap[arm], mass[arm], out=np.zeros(486), where=mass[arm] > 0)
                np.testing.assert_allclose(normalized.mean(), reference.loc[seed, f"normalized_{arm}"])
            # Symmetric scale puts both models in the same feature-specific units.
            common = (mass["hyb"] + mass["tab"]) / 2
            row["common_scale"] = np.divide(gap["hyb"] - gap["tab"], common,
                                             out=np.zeros(486), where=common > 0).mean()
            live = (mass["hyb"] > 0) & (mass["tab"] > 0)
            row["both_active"] = (gap["hyb"][live] / mass["hyb"][live]
                                  - gap["tab"][live] / mass["tab"][live]).mean()
        rows.append(row)
    detail = pd.DataFrame(rows).set_index("seed")
    detail.to_csv(directory / "plot_comparisons_by_seed.csv")
    plot_estimates(paired, detail, figures)
    for key in detail:
        print(key, paired_estimate(detail[key]), "positive", (detail[key]>0).sum())


def plot_estimates(paired, detail, figures):
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    navy, rust = "#254d67", "#994c35"
    fig, ax = plt.subplots(figsize=(6.3, 2.55), layout="constrained")
    offsets = np.linspace(-0.16, 0.16, len(paired))
    ax.axvline(0, color="0.48", lw=0.85, ls="--")
    for y, comparator in [(1, "cov"), (0, "tab")]:
        values = paired[f"js_hyb_minus_{comparator}"]
        ax.scatter(values, y + offsets, color=navy, s=16, alpha=0.75)
        mean, lo, hi = paired_estimate(values)
        ax.errorbar(mean, y-0.32, xerr=[[mean-lo], [hi-mean]], fmt="D", color=rust,
                    ms=5, capsize=3, lw=1.7)
    ax.set(yticks=[0,1], yticklabels=["Tabular baseline","Gaussian control"],
           ylim=(-0.58,1.42), xlabel="Difference in mean JS divergence (graph minus comparator, nats)")
    ax.tick_params(axis="y", length=0)
    fig.savefig(figures / "js_paired_refits.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(6.25, 2.9), layout="constrained")
    for ax, split in zip(axes, ("random", "temporal")):
        hybrid, gaussian = detail[f"{split}_hyb"], detail[f"{split}_cov"]
        ax.plot([0, 1], np.vstack([hybrid, gaussian]), color="#bfc8cc", lw=0.9)
        ax.scatter(np.zeros(len(hybrid)), hybrid, color=navy, s=14, zorder=2)
        ax.scatter(np.ones(len(gaussian)), gaussian, color=rust, s=14, zorder=2)
        ax.set(xticks=[0,1], xticklabels=["Graph","Gaussian"], xlim=(-.26,1.26),
               title="Random holdout" if split=="random" else "Forward-time holdout")
        ax.grid(axis="y", lw=.45, color="0.9")
        ax.set_axisbelow(True)
        ax.set_yticks(np.arange(0, 11)*1.0)
    # The unit is a count out of ten; use a common zoomed axis in both panels.
    low, high = detail.iloc[:, :4].min().min(), detail.iloc[:, :4].max().max()
    axes[0].set_ylim(np.floor(low*2)/2-.08, np.ceil(high*2)/2+.08)
    axes[0].set_yticks(np.arange(np.floor(low*2), np.ceil(high*2)+1) / 2)
    axes[0].set_ylabel("Tabular top-ten features retained")
    fig.savefig(figures / "top10_paired_agreement.pdf")
    plt.close(fig)


if __name__ == "__main__":
    draw(ROOT / "results/strict_ieee/primary", ROOT / "submission/source/figures")
