"""
Quantlet F1_framework_overview

Schematic of the audit design. The previous version of this figure drew a
single-fit pipeline (features -> hybrid -> SHAP on two splits -> test), which is
precisely the object the paper argues is uninterpretable on its own: one fit's
flagged fraction carries ~40% training-RNG noise. It also used a red/green pair
that is indistinguishable under deuteranopia (OKLab dE = 3.8 against a floor of 8).

This version draws what the paper actually does, in two panels:

  (a) the inferential unit -- a single paired fit is a screen, not evidence; the
      unit is the COMPLETE REFIT, repeated over seeds, giving a distribution of
      the paired contrast and an ordinary interval;
  (b) the control ladder -- what each successive negative control holds fixed,
      and therefore what a surviving residual could still be attributed to.

Palette: Okabe-Ito subset, checked for CVD separation (worst pairwise OKLab dE
under deuteranopia/protanopia = 11.2, against 3.8 for the palette it replaces).
Text wears ink tokens, never a series colour.

Datafile(s): none
Output:      F1_framework_overview.pdf, .png

Run from inside this folder:
    python F1_framework_overview.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BLUE   = "#0072B2"   # baseline arm
ORANGE = "#D55E00"   # augmented (hybrid) arm
GREEN  = "#009E73"   # the real graph embedding
GREY   = "#666666"   # controls / null material
INK    = "#1A1A1A"
MUTED  = "#5A5A5A"
FAINT  = "#DDDDDD"


def set_rcparams():
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
    })


def _box(ax, x, y, w, h, label, fc="white", ec=GREY, tc=INK, fs=8.5, lw=1.1,
         weight="normal", alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.010,rounding_size=0.02",
                                fc=fc, ec=ec, lw=lw, zorder=2, alpha=alpha))
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, color=tc, zorder=3, weight=weight, linespacing=1.35)


def _arrow(ax, p, q, color=MUTED, lw=1.1, style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=9,
                                 lw=lw, color=color, zorder=2,
                                 shrinkA=1.5, shrinkB=1.5))


def panel_a(ax):
    ax.set_title("(a)  The inferential unit is the complete refit", fontsize=10,
                 weight="bold", color=INK, loc="left", pad=10)

    # one paired fit -- a screen, not evidence
    _box(ax, 0.02, 0.66, 0.34, 0.26, "", fc="#FBFBFB", ec=FAINT, lw=1.0)
    ax.text(0.19, 0.885, "a single paired fit", ha="center", fontsize=8.5,
            color=MUTED, style="italic")
    _box(ax, 0.06, 0.755, 0.12, 0.085, "baseline\n$A$", fc=BLUE, ec=BLUE, tc="white", fs=8)
    _box(ax, 0.21, 0.755, 0.12, 0.085, "hybrid\n$B$", fc=ORANGE, ec=ORANGE, tc="white", fs=8)
    _arrow(ax, (0.195, 0.752), (0.195, 0.712))
    ax.text(0.195, 0.690, r"one $\bar{\delta}_k$", ha="center", fontsize=8.5, color=INK)
    ax.text(0.385, 0.79, "a screen, not evidence:\ncarries $\\sim\\!40\\%$\nretraining noise",
            ha="left", va="center", fontsize=8, color=MUTED, linespacing=1.4)

    # the refit loop
    _arrow(ax, (0.19, 0.655), (0.19, 0.615), color=INK)
    ax.text(0.19, 0.585, "repeat over seeds $s = 1 \\dots N$", ha="center",
            fontsize=8.5, color=INK, weight="bold")

    for i, yy in enumerate((0.455, 0.365, 0.275)):
        a = 1.0 if i < 2 else 0.30
        _box(ax, 0.06, yy, 0.12, 0.065, "", fc=BLUE, ec=BLUE, lw=0.8, alpha=a)
        _box(ax, 0.21, yy, 0.12, 0.065, "", fc=ORANGE, ec=ORANGE, lw=0.8, alpha=a)
        if i < 2:
            ax.text(0.355, yy + 0.032, f"$\\bar{{\\delta}}_k^{{({i+1})}}$",
                    fontsize=8, color=INK, va="center")
    ax.text(0.355, 0.307, r"$\vdots$", fontsize=10, color=MUTED, va="center")
    ax.text(0.195, 0.215,
            "every stage re-drawn per seed:\nholdout, features, graph, GNN, both fits",
            ha="center", fontsize=7.5, color=MUTED, linespacing=1.35)

    # resulting distribution + interval
    _arrow(ax, (0.42, 0.44), (0.50, 0.44), color=INK)
    rng = np.random.default_rng(7)
    x = rng.normal(0.70, 0.042, 500)
    ax.hist(x, bins=18, range=(0.56, 0.85), bottom=0.375,
            weights=np.full(500, 0.00042), color=GREY, alpha=0.40, zorder=2)
    ax.plot([0.55, 0.87], [0.375, 0.375], color=MUTED, lw=1.0, zorder=3)
    ax.plot([0.645, 0.755], [0.325, 0.325], color=INK, lw=2.4, zorder=4,
            solid_capstyle="butt")
    ax.plot([0.70], [0.325], "o", ms=5.5, color=INK, zorder=5)
    ax.text(0.70, 0.245, "interval across independent refits\n(no band, no per-feature $p$)",
            ha="center", fontsize=8, color=INK, linespacing=1.4)
    ax.text(0.70, 0.545, "distribution of the paired contrast", ha="center",
            fontsize=8.5, color=INK)
    ax.plot([0.86, 0.86], [0.315, 0.515], color=MUTED, lw=0.9, ls="--", zorder=1)
    ax.text(0.875, 0.415, "$0$", ha="left", fontsize=8.5, color=MUTED)


def panel_b(ax):
    ax.set_title("(b)  The control ladder: what each rung holds fixed", fontsize=10,
                 weight="bold", color=INK, loc="left", pad=10)

    rungs = [
        ("tabular baseline",           "no added columns",             False),
        ("$+$ random columns",         "dimension",                    False),
        ("$+$ permuted embedding",     "$+$ per-column marginals",     False),
        ("$+$ covariance-matched",     "$+$ second-order covariance",  False),
        ("$+$ scale- / mass-matched",  "$+$ attribution dilution",     False),
        ("real GraphSAGE embedding",   "$+$ graph structure",          True),
    ]
    y0, dy, h, w = 0.14, 0.128, 0.092, 0.42
    for i, (name, holds, real) in enumerate(rungs):
        y = y0 + i * dy
        _box(ax, 0.05, y, w, h, name,
             fc=GREEN if real else "white", ec=GREEN if real else GREY,
             tc="white" if real else INK, fs=8.5,
             weight="bold" if real else "normal")
        ax.text(0.50, y + h / 2, holds, va="center", fontsize=8,
                color=INK if real else MUTED)

    ax.annotate("", xy=(0.028, y0 + 5 * dy + h + 0.01), xytext=(0.028, y0),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))
    ax.text(0.005, y0 + 2.6 * dy, "successively stricter", rotation=90,
            va="center", ha="center", fontsize=8, color=MUTED)
    ax.text(0.05, 0.985,
            "A residual against a rung can still be explained by anything below it.\n"
            "Only the top rung adds graph structure.",
            fontsize=8, color=MUTED, linespacing=1.4, va="top")


def main():
    set_rcparams()
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))
    for ax in axes:
        ax.set_xlim(-0.01, 1.0)
        ax.set_ylim(0.12, 1.02)
        ax.axis("off")
    panel_a(axes[0])
    panel_b(axes[1])
    fig.suptitle("Auditing SHAP stability across retraining regimes",
                 fontsize=11.5, weight="bold", color=INK, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    here = Path(__file__).resolve().parent
    for ext in ("pdf", "png"):
        fig.savefig(here / f"F1_framework_overview.{ext}", dpi=200)
    plt.close(fig)
    print("saved:", here / "F1_framework_overview.pdf")


if __name__ == "__main__":
    main()
