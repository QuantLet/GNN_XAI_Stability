"""GNNXAI_null_bands -- every headline and T2b flagged fraction against its OWN
(dataset, learner) leave-one-out pairwise re-seeded null band.

Self-contained: reads only the committed single-set CSVs shipped in this Quantlet
folder (single_set_stability.csv, T8_elliptic_stability_singleset.csv,
null_bands_all.csv) -- never the ~300 MB of re-seeded refit SHAP caches, which are
git-ignored and archived externally (see Datafile note / Data-availability
statement). The band summaries in null_bands_all.csv are the output of the
leave-one-out pairwise construction in null_bands_construction.py (which DOES need
those external caches); this script only visualizes them.

Output: GNNXAI_null_bands.png (+ .pdf)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SS = pd.read_csv(HERE / "single_set_stability.csv").set_index("label")
EL = pd.read_csv(HERE / "T8_elliptic_stability_singleset.csv").set_index("pair")


def _pct(df, key, col="flagged_frac"):
    return float(df.loc[key, col]) * 100.0


def load_bands():
    """(dataset, learner) -> (lo%, hi%) from the regenerated pairwise bands."""
    nb = pd.read_csv(HERE / "null_bands_all.csv")
    nb = nb[nb["binning"] == "quantile"]
    bands = {}
    for _, r in nb.iterrows():
        bands[(r["dataset"], r["learner"])] = (r["lo"] * 100, r["hi"] * 100)
    return bands


BLOCKS = [
    ("IEEE-CIS", "XGBoost", [
        ("headline",     lambda: _pct(SS, "headline_xgb"), "headline"),
        ("T2b random",   lambda: _pct(SS, "t2b:xgb_random_emb"), "control-T2b"),
        ("T2b permuted", lambda: _pct(SS, "t2b:xgb_permuted_emb"), "control-T2b"),
        ("T2b covmatch", lambda: _pct(SS, "t2b:xgb_covmatched_emb"), "control-T2b"),
    ]),
    ("IEEE-CIS", "CatBoost", [
        ("headline",     lambda: _pct(SS, "headline_catb"), "headline"),
        ("T2b random",   lambda: _pct(SS, "t2b:catb_random_emb"), "control-T2b"),
        ("T2b permuted", lambda: _pct(SS, "t2b:catb_permuted_emb"), "control-T2b"),
        ("T2b covmatch", lambda: _pct(SS, "t2b:catb_covmatched_emb"), "control-T2b"),
    ]),
    ("Elliptic", "XGBoost", [
        ("headline",     lambda: _pct(EL, "headline", "frac_flagged"), "headline"),
        ("T2b random",   lambda: _pct(EL, "t2b_random", "frac_flagged"), "control-T2b"),
        ("T2b covmatch", lambda: _pct(EL, "t2b_covmatched", "frac_flagged"), "control-T2b"),
    ]),
    ("Elliptic", "CatBoost", [
        ("headline",     lambda: _pct(EL, "catb_headline", "frac_flagged"), "headline"),
        ("T2b covmatch", lambda: _pct(EL, "catb_t2b_covmatched", "frac_flagged"), "control-T2b"),
    ]),
]

C_HEAD, C_CTRL, C_BAND = "#1f4e79", "#2e7d32", "#9e9e9e"


def main():
    bands = load_bands()
    fig, ax = plt.subplots(figsize=(11, 5.2))
    xpos, xticklabels, spans = [], [], []
    x = 0.0
    for ds, lr, pts in BLOCKS:
        lo, hi = bands[(ds, lr)]
        x0 = x
        for label, getter, role in pts:
            val = getter()
            color = C_HEAD if role == "headline" else C_CTRL
            above = val > hi + 1e-9
            ax.scatter([x], [val], s=150 if above else 90, marker="*" if above else "o",
                       facecolor=color, edgecolor="black", linewidths=0.8, zorder=5)
            ax.annotate(f"{val:.1f}", (x, val), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7.5)
            xpos.append(x); xticklabels.append(label); x += 1.0
        x1 = x - 1.0
        ax.add_patch(plt.Rectangle((x0 - 0.4, lo), (x1 - x0 + 0.8), hi - lo,
                     facecolor=C_BAND, alpha=0.20, zorder=1))
        ax.text((x0 + x1) / 2, hi + 1.2, f"null {lo:.1f}-{hi:.1f}",
                ha="center", fontsize=7, color="#555")
        spans.append((x0 - 0.4, x1 + 0.4, f"{ds}\n{lr}"))
        x += 0.8
    for a, b, lab in spans:
        ax.text((a + b) / 2, 6.5, lab, ha="center", va="top", fontsize=9, fontweight="bold")
        ax.axvline(b + 0.4, color="#ddd", lw=0.8)
    ax.set_xticks(xpos); ax.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("% of features flagged by the diagnostic")
    ax.set_ylim(8, 52)
    ax.set_title("Flagged fraction vs. own (dataset, learner) leave-one-out pairwise null band: "
                 "no headline or T2b clears its band on IEEE-CIS", fontsize=10)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_HEAD, markeredgecolor="k", label="headline (tab vs hybrid)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_CTRL, markeredgecolor="k", label="T2b control vs hybrid"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gray", markeredgecolor="k", markersize=12, label="above band (signal)"),
        plt.Rectangle((0, 0), 1, 1, facecolor=C_BAND, alpha=0.2, label="leave-one-out pairwise null band"),
    ]
    ax.legend(handles=handles, fontsize=7.5, loc="upper right", framealpha=0.9)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"GNNXAI_null_bands.{ext}", dpi=160, bbox_inches="tight")
    print("wrote", HERE / "GNNXAI_null_bands.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
