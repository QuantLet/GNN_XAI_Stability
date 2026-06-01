"""
Quantlet F2_js_bars_with_ci

Top 30 most unstable features: JS divergence bars with 95% block-bootstrap CIs for xgb_tab vs xgb_sage_l1; FDR-significant features marked.

Datafile(s): stability_xgb_tab_vs_xgb_sage_l1_block_time.parquet
Output:      F2_js_bars_with_ci.pdf

Run from inside this folder:
    python F2_js_bars_with_ci.py
or open F2_js_bars_with_ci.ipynb in Jupyter.
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
df = pd.read_parquet("stability_xgb_tab_vs_xgb_sage_l1_block_time.parquet")
df = df.sort_values("JS_A", ascending=False).head(30)

fig, ax = plt.subplots(figsize=(11, 6.2))
x = np.arange(len(df)); w = 0.4
ax.bar(x - w/2, df["JS_A"], width=w, color=PALETTE["main_blue"], label="xgb_tab",
       yerr=[df["JS_A"] - df["ci_low_A"], df["ci_high_A"] - df["JS_A"]],
       ecolor=PALETTE["light"], capsize=2.5)
ax.bar(x + w/2, df["JS_B"], width=w, color=PALETTE["ida_red"], label="xgb_sage_l1",
       yerr=[df["JS_B"] - df["ci_low_B"], df["ci_high_B"] - df["JS_B"]],
       ecolor=PALETTE["light"], capsize=2.5)
sig = df["significant_fdr"].values
if sig.any():
    ax.scatter(x[sig] + w/2, df["JS_B"].values[sig] + 0.005,
               color=PALETTE["crimson"], marker="*", s=70, label="FDR<0.05")
ax.set_xticks(x); ax.set_xticklabels(df.index, rotation=45, ha="right", fontsize=10,
                                      rotation_mode="anchor")
ax.set_ylabel("JS divergence (random vs. temporal)")
ax.set_title("Top 30 most unstable features: xgb_tab vs. xgb_sage_l1")
legend_below(ax, y_offset=-0.40)
fig.subplots_adjust(bottom=0.32)
pdf, png = save_fig(fig, "F2_js_bars_with_ci", ".")
plt.close(fig)
print("saved:", pdf)
