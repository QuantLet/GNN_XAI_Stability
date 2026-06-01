"""
Quantlet F3b_persistence_bimodal

Attribution-persistence-ratio bimodal separator: real-signal hybrids in [0.87, 1.08], column-aug controls in [0.24, 0.46].

Datafile(s): T1b_attribution_persistence.csv
Output:      F3b_persistence_bimodal.pdf

Run from inside this folder:
    python F3b_persistence_bimodal.py
or open F3b_persistence_bimodal.ipynb in Jupyter.
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


import pandas as pd
import matplotlib.pyplot as plt

set_rcparams()
df = pd.read_csv("T1b_attribution_persistence.csv")

is_signal = df["config"].str.contains("sage_l1|sage_l2", regex=True)
signal = df[is_signal]; control = df[~is_signal]

fig, ax = plt.subplots(figsize=(8.5, 5))
ax.scatter(signal["persistence_ratio"], range(len(signal)),
           color=PALETTE["ida_red"], s=100, marker="o", label="real-signal hybrids")
for _, r in signal.iterrows():
    ax.annotate(r["config"], (r["persistence_ratio"], signal.index.get_loc(r.name)),
                fontsize=8, xytext=(5, 0), textcoords="offset points", va="center")
offset = len(signal)
ax.scatter(control["persistence_ratio"], [offset + i for i in range(len(control))],
           color=PALETTE["forest"], s=100, marker="s", label="column-aug. controls")
for i, (_, r) in enumerate(control.iterrows()):
    ax.annotate(r["config"], (r["persistence_ratio"], offset + i),
                fontsize=8, xytext=(5, 0), textcoords="offset points", va="center")

ax.axvline(0.65, color=PALETTE["grey"], linestyle="--", lw=1, label="separator at 0.65")
ax.set_xlabel("Attribution persistence ratio")
ax.set_yticks([])
ax.set_title("Real-signal vs control: bimodal separation of persistence")
legend_below(ax, y_offset=-0.18)
fig.subplots_adjust(bottom=0.28)
pdf, png = save_fig(fig, "F3b_persistence_bimodal", ".")
plt.close(fig)
print("saved:", pdf)
