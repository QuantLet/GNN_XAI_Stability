"""Quantlet F3_refit_level_results

The results that carry the paper's inference, in one figure. The panel this
replaces put BH-ranked flagged fractions at the centre, but the paper states those
per-feature flags have no direct inferential interpretation -- so the figure was
foregrounding the one object the text tells the reader not to weight.

Three panels, each an object the paper does act on:

  (a) the twenty full-pipeline replicates under the PINNED learner -- the primary
      estimand; its t-interval includes zero;
  (b) the twelve paired refits under the STOCHASTIC learner -- a different learner
      specification, whose interval excludes zero in the destabilizing direction;
  (c) the real hybrid against the attribution-scale-matched surrogate, read against
      the null built for that contrast's own 550-vs-1,189 geometry.

Read together: no specification supports a stabilization gain, and the apparent
positive under scale-sensitive estimators sits at the centre of its matched null.

Palette: Okabe-Ito subset (worst pairwise CVD separation 11.2 OKLab dE).

Datafile(s): seed_sensitivity.csv, refit_level_inference.csv,
             null_band_scalematched.csv  (all shipped in this folder)
Output:      F3_refit_level_results.pdf, .png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

BLUE, ORANGE, GREY, INK, MUTED = "#0072B2", "#D55E00", "#666666", "#1A1A1A", "#5A5A5A"
HERE = Path(__file__).resolve().parent


def _interval(x):
    n = len(x)
    m, sd = float(np.mean(x)), float(np.std(x, ddof=1))
    se = sd / np.sqrt(n)
    t = float(stats.t.ppf(0.975, n - 1))
    return m, m - t * se, m + t * se


def _dots(ax, x, colour, label, y=0.0):
    m, lo, hi = _interval(x)
    rng = np.random.default_rng(3)
    ax.scatter(x, y + rng.uniform(-0.055, 0.055, len(x)), s=22, color=colour,
               alpha=0.55, zorder=3, linewidths=0)
    ax.plot([lo, hi], [y - 0.16, y - 0.16], color=INK, lw=2.4, zorder=4,
            solid_capstyle="butt")
    ax.plot([m], [y - 0.16], "o", ms=6, color=INK, zorder=5)
    excl = "excludes 0" if (lo > 0 or hi < 0) else "includes 0"
    ax.text(0.5, 0.06, f"{m:+.4f}  [{lo:+.4f}, {hi:+.4f}]   {excl}",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8, color=INK)
    return m, lo, hi


def main():
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                                "ps.fonttype": 42})
    seed = pd.read_csv(HERE / "seed_sensitivity.csv")
    refit = pd.read_csv(HERE / "refit_level_inference.csv")
    band = pd.read_csv(HERE / "null_band_scalematched.csv").iloc[0]

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5))

    # (a) pinned full-pipeline replicates
    ax = axes[0]
    _dots(ax, seed.mean_delta.values, BLUE, "pinned")
    ax.set_title(f"(a)  Pinned, full pipeline  ($n = {len(seed)}$)", fontsize=9.5,
                 weight="bold", color=INK, loc="left")

    # (b) stochastic learner refits
    ax = axes[1]
    _dots(ax, refit.mean_delta.values, ORANGE, "stochastic")
    ax.set_title(f"(b)  Stochastic learner  ($n = {len(refit)}$)", fontsize=9.5,
                 weight="bold", color=INK, loc="left")

    # (c) real vs scale-matched, against its own-geometry null
    ax = axes[2]
    lo, hi, mu = float(band.md_lo), float(band.md_hi), float(band.md_mean)
    ax.axvspan(lo, hi, color=GREY, alpha=0.18, zorder=1)
    ax.plot([lo, lo], [-0.22, 0.22], color=GREY, lw=1.2, zorder=2)
    ax.plot([hi, hi], [-0.22, 0.22], color=GREY, lw=1.2, zorder=2)
    ax.plot([mu], [0.0], "|", ms=14, color=GREY, zorder=3)
    ax.plot([-0.0007], [0.0], "o", ms=8, color=INK, zorder=5)
    ax.text(0.5, 0.06,
            f"observed $-0.0007$ inside $[{lo:+.4f}, {hi:+.4f}]$",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8, color=INK)
    ax.text(0.5, 0.90, f"null mean {mu:+.4f}", transform=ax.transAxes,
            ha="center", fontsize=7.5, color=MUTED)
    ax.set_title(f"(c)  Real vs scale-matched, $1$-Wasserstein  ({int(band.n_draws)} draws)",
                 fontsize=9.5, weight="bold", color=INK, loc="left")

    for ax in axes:
        ax.axvline(0.0, color=MUTED, lw=0.9, ls="--", zorder=1)
        ax.set_ylim(-0.75, 0.45)
        ax.set_yticks([])
        ax.spines[["left", "right", "top"]].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.tick_params(axis="x", labelsize=8, colors=MUTED)
        ax.set_xlabel(r"paired $\bar{\delta}_k$   ($>0$: hybrid less stable)"
                      if ax is not axes[2] else
                      r"paired $\bar{\delta}^{W}_k$   ($>0$: hybrid less stable)",
                      fontsize=8.5, color=INK)
    axes[0].set_xlim(-0.021, 0.013)
    axes[1].set_xlim(-0.021, 0.040)
    axes[2].set_xlim(-0.017, 0.017)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"F3_refit_level_results.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved:", HERE / "F3_refit_level_results.pdf")


if __name__ == "__main__":
    main()
