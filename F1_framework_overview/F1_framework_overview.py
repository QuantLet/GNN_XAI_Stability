"""
Quantlet F1_framework_overview

Framework schematic: tabular + GNN -> hybrid -> SHAP on two splits -> JS test.

Datafile(s): none
Output:      F1_framework_overview.pdf

Run from inside this folder:
    python F1_framework_overview.py
or open F1_framework_overview.ipynb in Jupyter.
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


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

set_rcparams()
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

def box(x, y, w, h, label, color, text_color="white"):
    r = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                 linewidth=1.2, facecolor=color, edgecolor="black")
    ax.add_patch(r)
    ax.text(x + w/2, y + h/2, label, ha="center", va="center",
            color=text_color, fontsize=10, weight="bold")

box(0.3, 4.5, 2.2, 1.0, "Tabular\nfeatures",   PALETTE["main_blue"])
box(0.3, 2.7, 2.2, 1.0, "Graph\n(entities)",   PALETTE["grey"])
box(0.3, 0.9, 2.2, 1.0, "GraphSAGE\nembeddings", PALETTE["forest"])
box(3.2, 2.7, 2.4, 1.4, "Hybrid\nlearner",     PALETTE["ida_red"])
box(6.3, 4.0, 2.0, 1.0, "SHAP\n(random)", PALETTE["grey"], "white")
box(6.3, 1.6, 2.0, 1.0, "SHAP\n(temporal)", PALETTE["grey"], "white")
box(8.6, 2.7, 1.2, 1.4, "JS\n+ test",       PALETTE["crimson"])

for (x1, y1, x2, y2) in [(2.5,5.0,3.2,3.7),(2.5,3.2,3.2,3.4),(2.5,1.4,3.2,3.1),
                          (5.6,3.4,6.3,4.5),(5.6,3.4,6.3,2.1),
                          (8.3,4.5,8.6,3.6),(8.3,2.1,8.6,2.9)]:
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="black"))

ax.text(5, 5.7, "Per-feature SHAP attribution stability under temporal shift",
        ha="center", fontsize=12, weight="bold")

pdf, png = save_fig(fig, "F1_framework_overview", ".")
plt.close(fig)
print("saved:", pdf)
