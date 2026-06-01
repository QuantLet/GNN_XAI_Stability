"""
Quantlet T6_tuned_hybrid

Hyperparameter-tuned xgb_sage_l1_tuned via Optuna TPE (25 trials). Recovers 68% of pinned hybrid's temporal-AUC gap.

Datafile(s): T6_tuned_hybrid.csv
Output:      T6_tuned_hybrid.csv

Run from inside this folder:
    python T6_tuned_hybrid.py
or open T6_tuned_hybrid.ipynb in Jupyter.
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


import shutil
from pathlib import Path

INPUT  = "T6_tuned_hybrid.csv"
OUTPUT = "T6_tuned_hybrid.csv"

# Some tables (T6) carry section markers and uneven row widths, so render
# the file content directly rather than forcing pandas.read_csv.
text = Path(INPUT).read_text(encoding="utf-8")
print(text)

if Path(INPUT).resolve() != Path(OUTPUT).resolve():
    shutil.copyfile(INPUT, OUTPUT)
print(f"wrote: {OUTPUT}")
