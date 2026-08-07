"""GNNXAI_scale_invariance -- scale response of the quantile-bin estimator vs equal-width

Self-contained: reads only the committed CSVs shipped in this Quantlet folder.
quantile_scale_invariance.csv is produced by scripts/quantile_scale_invariance.py.

Output: GNNXAI_scale_invariance.png (+ .pdf)
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


NAME = "GNNXAI_scale_invariance"
Q = pd.read_csv(HERE / "quantile_scale_invariance.csv")
ORDER = ["none", "global_1_over_r", "per_feature_1_over_rk"]
NICE = {"none": "observed", "global_1_over_r": "global rescale",
        "per_feature_1_over_rk": "per-feature rescale"}


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 3.8))
    x = np.arange(len(ORDER))
    for ax, col, ylab, ttl in (
            (ax1, "flagged_frac", "flagged fraction (%)", "(a) flagged fraction"),
            (ax2, "mean_delta", r"mean $\bar{\delta}_k$", "(b) panel mean")):
        for bt, c in (("quantile", C_OBS), ("equalwidth", C_ALT)):
            s = Q[Q.bin_type == bt].set_index("rescaling").loc[ORDER]
            v = s[col] * (100 if col == "flagged_frac" else 1)
            ax.plot(x, v, "o-", color=c, lw=2,
                    label="quantile (primary)" if bt == "quantile" else "equal-width")
        ax.set_xticks(x)
        ax.set_xticklabels([NICE[o] for o in ORDER], fontsize=8.5)
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_title(ttl, fontsize=9.5)
        ax.legend(fontsize=8)
    ax2.axhline(0.0, color="k", lw=0.8, ls=":")
    fig.suptitle("Undoing the attribution rescaling: equal-width loses its effect, "
                 "quantile barely moves", fontsize=10)
    _save(fig, NAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
