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

CANONICAL = [
    ("xgb_tab", "xgb_sage_l1"),
    ("xgb_tab", "xgb_sage_l2"),
    ("xgb_tab", "xgb_noise_inj"),
    ("xgb_tab", "xgb_random_emb"),
    ("xgb_tab", "xgb_permuted_emb"),
    ("catb_tab", "catb_sage_l1"),
]
LABEL = {"xgb_tab": "xgb", "catb_tab": "catb",
         "xgb_sage_l1": "xgb+L1", "xgb_sage_l2": "xgb+L2",
         "catb_sage_l1": "catb+L1", "xgb_random_emb": "xgb+rand",
         "xgb_permuted_emb": "xgb+perm", "xgb_noise_inj": "xgb+noise"}

rows = []
for a, b in CANONICAL:
    df = pd.read_parquet(f"stability_{a}_vs_{b}_block_time.parquet")
    n_sig = int(df["significant_fdr"].sum()); n = len(df)
    rows.append((f"{LABEL[a]} vs {LABEL[b]}", 100.0 * n_sig / n))

labels = [r[0] for r in rows]; vals = [r[1] for r in rows]
colors = [PALETTE["ida_red"]] * 3 + [PALETTE["forest"]] * 2 + [PALETTE["ida_red"]]

fig, ax = plt.subplots(figsize=(10, 5.2))
xs = np.arange(len(labels))
bars = ax.bar(xs, vals, color=colors, edgecolor="black", linewidth=0.8)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 1.0, f"{v:.1f}%",
            ha="center", fontsize=9)
ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=45, ha="right",
                                      rotation_mode="anchor")
ax.set_ylabel("% flagged")
ax.set_ylim(0, 65)
ax.set_title("Fraction of features flagged by the BH-ranked diagnostic, read against the empirical null")
fig.subplots_adjust(bottom=0.28)
pdf, png = save_fig(fig, "F3_frac_significant_by_config", ".")
plt.close(fig)
print("saved:", pdf)
