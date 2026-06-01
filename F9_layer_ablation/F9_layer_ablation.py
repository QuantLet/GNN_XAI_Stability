"""
Quantlet F9_layer_ablation

Layer ablation: per-feature stability gain delta_k under L1 vs L2 SAGE embeddings.

Datafile(s): stability_xgb_tab_vs_xgb_sage_l1_block_time.parquet, stability_xgb_tab_vs_xgb_sage_l2_block_time.parquet
Output:      F9_layer_ablation.pdf

Run from inside this folder:
    python F9_layer_ablation.py
or open F9_layer_ablation.ipynb in Jupyter.
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
d1 = pd.read_parquet("stability_xgb_tab_vs_xgb_sage_l1_block_time.parquet")
d2 = pd.read_parquet("stability_xgb_tab_vs_xgb_sage_l2_block_time.parquet")

fig, ax = plt.subplots(figsize=(8.5, 5.2))
ax.hist(d1["delta_obs"], bins=40, alpha=0.6, color=PALETTE["ida_red"],
        label=f"L1: mean delta_k = {d1['delta_obs'].mean():.4f}")
ax.hist(d2["delta_obs"], bins=40, alpha=0.5, color=PALETTE["main_blue"],
        label=f"L2: mean delta_k = {d2['delta_obs'].mean():.4f}")
ax.axvline(0, color=PALETTE["grey"], linestyle="--", lw=1)
ax.set_xlabel("delta_k = JS_hybrid - JS_tab")
ax.set_ylabel("# features")
ax.set_title("Layer ablation: per-feature stability gain (L1 vs L2)")
legend_below(ax, y_offset=-0.20)
fig.subplots_adjust(bottom=0.25)
pdf, png = save_fig(fig, "F9_layer_ablation", ".")
plt.close(fig)
print("saved:", pdf)
