"""GNNXAI_estimator_adjudication -- bin-free estimators and the attribution-scale confound

Self-contained: reads only the committed CSVs shipped in this Quantlet folder.
binning_adjudication.csv and scale_confound.csv are produced by
scripts/binning_adjudication.py and scripts/scale_confound.py.

Output: GNNXAI_estimator_adjudication.png (+ .pdf)
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


NAME = "GNNXAI_estimator_adjudication"
A = pd.read_csv(HERE / "binning_adjudication.csv")
S = pd.read_csv(HERE / "scale_confound.csv")


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.9))

    # (a) each bin-free estimator against its own shape-matched null band
    ests = [e for e in A.estimator.unique()]
    for i, e in enumerate(ests):
        obs = A[(A.estimator == e) & (A.kind == "observed")].iloc[0]
        nul = A[(A.estimator == e) & (A.kind == "shape_matched_null")].iloc[0]
        lo, hi = nul.flagged_lo * 100, nul.flagged_hi * 100
        ax1.plot([lo, hi], [i, i], lw=9, color=C_BAND, alpha=0.30, solid_capstyle="butt")
        ax1.scatter([obs.flagged_frac * 100], [i], color=C_OBS, s=46, zorder=3)
        ax1.text(obs.flagged_frac * 100 + 0.8, i, f"{obs.flagged_frac*100:.1f}%",
                 va="center", fontsize=8.5, color=C_OBS)
    ax1.set_yticks(range(len(ests)))
    ax1.set_yticklabels([e.capitalize() for e in ests], fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel("flagged fraction (%)", fontsize=9)
    ax1.set_title("(a) bin-free estimators clear their own null", fontsize=9.5)
    ax1.set_xlim(38, 68)

    # (b) the scale account: how much of the effect a pure rescaling explains
    lbl = [p.replace(" ", "\n") for p in S.pair]
    x = np.arange(len(S))
    ax2.bar(x - 0.19, S.raw_frac_negative * 100, width=0.36, color=C_OBS,
            label=r"raw: share $\delta_k<0$")
    ax2.bar(x + 0.19, S.norm_frac_negative * 100, width=0.36, color=C_ALT,
            label=r"scale-normalised")
    ax2.axhline(50, color="k", ls=":", lw=0.9)
    for i, r in S.iterrows():
        ax2.text(i, 2, f"ratio {r.scale_ratio_median:.3f}", ha="center", fontsize=7.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(lbl, fontsize=8)
    ax2.set_ylabel(r"% of features with $\delta_k<0$", fontsize=9)
    ax2.set_ylim(0, 85)
    ax2.set_title("(b) removing the rescaling returns a coin flip", fontsize=9.5)
    ax2.legend(fontsize=7.5, loc="upper right")

    _save(fig, NAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
