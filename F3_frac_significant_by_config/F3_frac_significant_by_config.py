"""
Quantlet F3_frac_significant_by_config

BH-flagged fraction across the 6 canonical configurations (4 hybrids + 2 controls) under block-bootstrap.

Datafile(s): stability_xgb_tab_vs_xgb_sage_l1_block_time.parquet, stability_xgb_tab_vs_xgb_sage_l2_block_time.parquet, stability_xgb_tab_vs_xgb_noise_inj_block_time.parquet, stability_xgb_tab_vs_xgb_random_emb_block_time.parquet, stability_xgb_tab_vs_xgb_permuted_emb_block_time.parquet, stability_catb_tab_vs_catb_sage_l1_block_time.parquet
Output:      F3_frac_significant_by_config.pdf

Run from inside this folder:
    python F3_frac_significant_by_config.py
or open F3_frac_significant_by_config.ipynb in Jupyter.
"""
from pathlib import Path

PALETTE = {
    "main_blue": "#003DA5",
    "ida_red":   "#C8102E",
    "forest":    "#228B22",
    "crimson":   "#DC143C",
    "grey":      "#777777",
    "light":     "#CCCCCC",
}

def set_rcparams():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25,
        "grid.linestyle": "--", "grid.linewidth": 0.5,
        "legend.frameon": False, "legend.fontsize": 10,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "figure.dpi": 100, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

def save_fig(fig, name, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{name}.pdf"; png = out_dir / f"{name}.png"
    fig.savefig(pdf, transparent=True); fig.savefig(png, transparent=True)
    return pdf, png

def legend_below(ax, y_offset=-0.18, ncol=None, fontsize=9):
    h, l = ax.get_legend_handles_labels()
    if not h: return
    if ncol is None: ncol = min(len(h), 4)
    ax.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, y_offset),
              ncol=ncol, frameon=False, fontsize=fontsize)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

set_rcparams()

# ---------------------------------------------------------------------------
# PRIMARY (quantile-bin) per-configuration flagged fractions.
#
# The original per-feature quantile SHAP/stability parquets that produced these
# fractions were lost in the 2026-06-22 disk crash (see STATUS.md); the shipped
# stability_*.parquet caches are the equal-width robustness spec / a partial
# reconstruction and do NOT reproduce the paper's primary-quantile Table 2/3
# fractions. This figure is therefore plotted from the published, table-backed
# quantile values in F3_published_quantile.csv (each cell cited to Table 2 or
# Table 3), so that Figure 3 matches Table 2 (tab:primary) and Table 3
# (tab:controls) exactly rather than the equal-width numbers previously shown.
# ---------------------------------------------------------------------------
df = pd.read_csv("F3_published_quantile.csv")
labels = df["label"].tolist()
vals = df["flagged_pct"].tolist()
# embedding hybrids (incl. the noise-injected perturbation of the real hybrid) in
# red; column-augmentation controls (random / permuted) in green.
color_for = {"hybrid": PALETTE["ida_red"], "hybrid_perturbation": PALETTE["ida_red"],
             "control": PALETTE["forest"]}
colors = [color_for[r] for r in df["role"]]

# Learner-specific IEEE-CIS pairwise null bands, read from the canonical
# results/tables/null_bands_all.csv. There is deliberately NO fallback: a silent
# default is how this figure came to print a superseded band ([35.8, 44.1]) after the
# table was re-anchored to the regenerated [36.0, 44.0]. The figure legend is outside
# the manuscript number audit's scope, so it must fail loudly here instead.
def _ieee_band(learner):
    f = Path(__file__).resolve().parents[3] / "results" / "tables" / "null_bands_all.csv"
    if not f.exists():
        raise SystemExit(f"missing {f}; run scripts/null_bands_assemble.py first")
    nb = pd.read_csv(f)
    row = nb[(nb["dataset"] == "IEEE-CIS") & (nb["learner"] == learner)
             & (nb["binning"] == "quantile")]
    if not len(row):
        raise SystemExit(f"no IEEE-CIS/{learner}/quantile row in {f.name}")
    return (round(float(row["lo"].iloc[0]) * 100, 1),
            round(float(row["hi"].iloc[0]) * 100, 1))

XGB_BAND = _ieee_band("XGBoost")
CATB_BAND = _ieee_band("CatBoost")

fig, ax = plt.subplots(figsize=(10, 5.2))
xs = np.arange(len(labels))
# shade the XGBoost null band across the five XGBoost bars (all but the last)
ax.axhspan(XGB_BAND[0], XGB_BAND[1], xmax=5/6, color=PALETTE["light"], alpha=0.5, zorder=0,
           label=f"XGBoost null band [{XGB_BAND[0]}, {XGB_BAND[1]}]")
ax.axhspan(CATB_BAND[0], CATB_BAND[1], xmin=5/6, color="#F4C2C2", alpha=0.5, zorder=0,
           label=f"CatBoost null band [{CATB_BAND[0]}, {CATB_BAND[1]}]")
bars = ax.bar(xs, vals, color=colors, edgecolor="black", linewidth=0.8, zorder=3)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 1.0, f"{v:.1f}%",
            ha="center", fontsize=9, zorder=4)
ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=45, ha="right",
                                      rotation_mode="anchor")
ax.set_ylabel("% flagged (quantile bins)")
ax.set_ylim(0, 65)
ax.set_title("Fraction of features flagged by the BH-ranked diagnostic, read against the empirical null band")
ax.legend(loc="upper right", fontsize=8, frameon=False)
fig.subplots_adjust(bottom=0.28)
pdf, png = save_fig(fig, "F3_frac_significant_by_config", ".")
plt.close(fig)
print("saved:", pdf, "values:", list(zip(labels, vals)))
