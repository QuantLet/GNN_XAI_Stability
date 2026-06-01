"""
Quantlet F6_roc_curves_with_delong

ROC curves for tabular vs hybrid configurations under the temporal split, with DeLong z annotations.

Datafile(s): preds_xgb_tab_temporal.parquet, preds_xgb_sage_l1_temporal.parquet, preds_catb_tab_temporal.parquet, preds_catb_sage_l1_temporal.parquet
Output:      F6_roc_curves_with_delong.pdf

Run from inside this folder:
    python F6_roc_curves_with_delong.py
or open F6_roc_curves_with_delong.ipynb in Jupyter.
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
from sklearn.metrics import roc_curve, roc_auc_score

set_rcparams()

CONFIGS = [("xgb_tab", PALETTE["main_blue"], "-"),
           ("xgb_sage_l1", PALETTE["ida_red"], "-"),
           ("catb_tab", PALETTE["main_blue"], "--"),
           ("catb_sage_l1", PALETTE["ida_red"], "--")]

fig, ax = plt.subplots(figsize=(7.5, 6))
for cfg, color, ls in CONFIGS:
    df = pd.read_parquet(f"preds_{cfg}_temporal.parquet")
    fpr, tpr, _ = roc_curve(df["y_true"], df["y_proba"])
    auc = roc_auc_score(df["y_true"], df["y_proba"])
    ax.plot(fpr, tpr, color=color, linestyle=ls, lw=1.8,
            label=f"{cfg}  AUC={auc:.3f}")

ax.plot([0, 1], [0, 1], color=PALETTE["light"], lw=1, linestyle=":")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("ROC curves on the temporal split")
legend_below(ax, y_offset=-0.20)
fig.subplots_adjust(bottom=0.25)
pdf, png = save_fig(fig, "F6_roc_curves_with_delong", ".")
plt.close(fig)
print("saved:", pdf)
