"""GNNXAI_instance_level -- per-row explanation agreement against both nulls

Self-contained: reads only the committed CSVs shipped in this Quantlet folder.
instance_level_agreement.csv and instance_level_null_band_matched.csv are
produced by scripts/instance_level_check.py.

Output: GNNXAI_instance_level.png (+ .pdf)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
C_OBS, C_BAND, C_ALT = "#c0392b", "#7f8c8d", "#2c7fb8"


def _save(fig, name):
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"{name}.{ext}", dpi=160, bbox_inches="tight")
    print("wrote", HERE / f"{name}.png")


NAME = "GNNXAI_instance_level"
OBS = pd.read_csv(HERE / "instance_level_agreement.csv")
NUL = pd.read_csv(HERE / "instance_level_null_band_matched.csv")
METRICS = [("topk_overlap_mean", "top-10 overlap"), ("spearman_mean", "Spearman corr.")]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    for ax, (col, nice) in zip(axes, METRICS):
        for i, split in enumerate(("random", "temporal")):
            n = NUL[(NUL.split == split) & (NUL.kind == "offdiagonal")][col]
            lo, hi = np.percentile(n, 2.5), np.percentile(n, 97.5)
            ax.plot([i, i], [lo, hi], lw=10, color=C_BAND, alpha=0.30,
                    solid_capstyle="butt")
            o = OBS[(OBS.A == "xgb_tab") & (OBS.B == "xgb_sage_l1")
                    & (OBS.split == split)].iloc[0][col]
            ax.scatter([i], [o], color=C_OBS, s=52, zorder=3)
            verdict = "inside" if lo <= o <= hi else "BELOW band"
            ax.text(i + 0.10, o, f"{o:.3f}\n({verdict})", fontsize=8, va="center")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["random split", "temporal split"], fontsize=9)
        ax.set_xlim(-0.4, 1.6)
        ax.set_ylabel(nice, fontsize=9)
        ax.set_title(nice, fontsize=9.5)
    handles = [plt.Line2D([], [], marker="o", ls="", color=C_OBS, label="observed hybrid"),
               plt.Rectangle((0, 0), 1, 1, facecolor=C_BAND, alpha=0.3,
                             label="shape-matched null band")]
    axes[0].legend(handles=handles, fontsize=7.5, loc="lower left")
    fig.suptitle("Per-row explanation agreement: on the temporal split the hybrid "
                 "falls BELOW its covariance-matched null", fontsize=10)
    _save(fig, NAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
