"""GNNXAI_degeneracy_support -- degenerate features and the live common support

Self-contained: reads only the committed CSVs shipped in this Quantlet folder.
degeneracy_audit.csv is the output of scripts/degeneracy_audit.py, which
reads the per-feature SHAP caches; this script only visualizes it.

Output: GNNXAI_degeneracy_support.png (+ .pdf)
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


NAME = "GNNXAI_degeneracy_support"
D = pd.read_csv(HERE / "degeneracy_audit.csv")
PAIRS = [("xgb_tab", "xgb_sage_l1", "XGBoost headline"),
         ("catb_tab", "catb_sage_l1", "CatBoost headline"),
         ("xgb_tab", "xgb_covmatched_emb", "XGB cov-matched control")]


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.9))

    # (a) dead-column asymmetry: the hybrid retires more tabular columns
    rows = [D[(D.A == a) & (D.B == b)].iloc[0] for a, b, _ in PAIRS]
    y = np.arange(len(rows))
    ax1.barh(y - 0.19, [r.dead_A for r in rows], height=0.36, color=C_BAND,
             label="baseline arm ($A$)")
    ax1.barh(y + 0.19, [r.dead_B for r in rows], height=0.36, color=C_OBS,
             label="augmented arm ($B$)")
    for i, r in enumerate(rows):
        ax1.text(r.dead_B + 0.7, i + 0.19, f"{int(r.dead_B)}", va="center", fontsize=8)
        ax1.text(r.dead_A + 0.7, i - 0.19, f"{int(r.dead_A)}", va="center", fontsize=8)
    ax1.set_yticks(y)
    ax1.set_yticklabels([lbl for _, _, lbl in PAIRS], fontsize=8.5)
    ax1.invert_yaxis()
    ax1.set_xlabel("tabular columns never split on (of 486)", fontsize=9)
    ax1.set_title("(a) degenerate mass is asymmetric across arms", fontsize=9.5)
    ax1.legend(fontsize=7.5, loc="lower right")

    # (b) full vs live support: the flagged fraction rises, the panel mean flips
    r = D[(D.A == "xgb_tab") & (D.B == "xgb_sage_l1")].iloc[0]
    ax2b = ax2.twinx()
    ax2.bar([0, 1], [r.frac_flagged_all * 100, r.frac_flagged_live * 100],
            width=0.5, color=C_BAND, alpha=0.55)
    ax2b.plot([0, 1], [r.mean_delta_all, r.mean_delta_live], "o-", color=C_OBS, lw=2)
    ax2b.axhline(0.0, color="k", lw=0.7, ls=":")
    for x, v in zip([0, 1], [r.frac_flagged_all * 100, r.frac_flagged_live * 100]):
        ax2.text(x, v + 0.4, f"{v:.1f}%", ha="center", fontsize=8.5)
    for x, v in zip([0, 1], [r.mean_delta_all, r.mean_delta_live]):
        ax2b.text(x, v, f"  {v:+.4f}", color=C_OBS, fontsize=8.5, va="bottom")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels([f"full support\n($p=486$)",
                         f"live common support\n($p={int(r.n_live)}$)"], fontsize=8.5)
    ax2.set_ylabel("flagged fraction (%)", fontsize=9)
    ax2b.set_ylabel(r"panel mean $\bar{\delta}_k$", color=C_OBS, fontsize=9)
    ax2.set_ylim(0, max(r.frac_flagged_all, r.frac_flagged_live) * 100 + 5)
    ax2.set_title("(b) the negative panel mean is carried by dropped columns", fontsize=9.5)

    _save(fig, NAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
