"""GNNXAI_null_bands_matched -- shape-matched (486-vs-550) re-seeded null bands

Self-contained: reads only the committed CSVs shipped in this Quantlet folder.
The bands are produced by scripts/null_refit_covmatched.py and
scripts/null_bands_matched.py from the re-seeded refit caches; this script
only visualizes the assembled summaries.

Output: GNNXAI_null_bands_matched.png (+ .pdf)
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


NAME = "GNNXAI_null_bands_matched"
M = pd.read_csv(HERE / "null_bands_matched.csv")
R = pd.read_csv(HERE / "null_bands_all_regen.csv")

# observed headline values (Table 2 of the manuscript)
OBS = {"xgb": 36.0, "catb": 29.2}


def _band(df, **q):
    for k, v in q.items():
        df = df[df[k] == v]
    r = df.iloc[0]
    return r.lo * 100, r.hi * 100, r["mean"] * 100


def main():
    rows = []
    for lrn, name in (("xgb", "IEEE-CIS XGBoost"), ("catb", "IEEE-CIS CatBoost")):
        sd = R[(R.learner == ("XGBoost" if lrn == "xgb" else "CatBoost"))
               & (R.binning == "quantile")].iloc[0]
        rows.append((f"{name}\nsame-dimension 486-vs-486",
                     sd.lo * 100, sd.hi * 100, OBS[lrn]))
        lo, hi, _ = _band(M, learner=lrn, family="matched_headline_off", binning="quantile")
        rows.append((f"{name}\nshape-matched 486-vs-550", lo, hi, OBS[lrn]))

    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    y = np.arange(len(rows))
    for i, (_, lo, hi, obs) in enumerate(rows):
        ax.plot([lo, hi], [i, i], lw=9, color=C_BAND, alpha=0.30,
                solid_capstyle="butt", zorder=1)
        ax.plot([lo, lo], [i - .18, i + .18], color=C_BAND, lw=1.4)
        ax.plot([hi, hi], [i - .18, i + .18], color=C_BAND, lw=1.4)
        ax.scatter([obs], [i], color=C_OBS, zorder=3, s=46)
        # "at floor" rather than "below" when the observed value sits within half a
        # point of the lower edge: the manuscript reads that case as the band floor
        # (5th percentile), not as clearing the band downward.
        if obs > hi:
            verdict = "above"
        elif obs >= lo:
            verdict = "inside"
        elif lo - obs <= 0.5:
            verdict = "at floor"
        else:
            verdict = "below"
        ax.text(hi + 0.6, i, f"[{lo:.1f}, {hi:.1f}]  obs {obs:.1f}%  ({verdict})",
                va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("flagged fraction (%)", fontsize=9.5)
    ax.set_xlim(24, 60)
    ax.set_ylim(len(rows) - 0.3, -1.05)   # headroom for the legend strip
    ax.set_title("Headline flagged fraction against the same-dimension and the\n"
                 "shape-matched re-seeded null (quantile bins)", fontsize=10)
    handles = [plt.Line2D([], [], marker="o", ls="", color=C_OBS, label="observed headline"),
               plt.Rectangle((0, 0), 1, 1, facecolor=C_BAND, alpha=0.3,
                             label="95% re-seeded null band")]
    ax.legend(handles=handles, fontsize=8, loc="upper right", ncol=2, framealpha=0.9)
    _save(fig, NAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
