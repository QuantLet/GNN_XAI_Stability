"""
Quantlet T3_robustness_controls

Three column-augmentation control configurations (random, permuted, noise-injected) per learner family.

Datafile(s): T3_robustness_controls.csv
Output:      T3_robustness_controls.csv, T3_robustness_controls.pdf, T3_robustness_controls.png

Run from inside this folder:
    python T3_robustness_controls.py
or open T3_robustness_controls.ipynb in Jupyter.
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
import matplotlib.pyplot as plt
import pandas as pd

INPUT      = "T3_robustness_controls.csv"
OUTPUT_CSV = "T3_robustness_controls.csv"
OUTPUT_PDF = "T3_robustness_controls.pdf"
OUTPUT_PNG = "T3_robustness_controls.png"

# 1. Echo the CSV content (handles T6's multi-section format too).
text = Path(INPUT).read_text(encoding="utf-8")
print(text)

# 2. Re-emit the canonical CSV (no-op when input==output).
if Path(INPUT).resolve() != Path(OUTPUT_CSV).resolve():
    shutil.copyfile(INPUT, OUTPUT_CSV)

# 3. Render the table as a matplotlib figure -> PDF + PNG.
#    Robust to multi-section CSVs: pandas.read_csv with comment="#" and
#    skip_blank_lines drops section headers; if the file still has mixed
#    schemas we fall back to a monospace text rendering.
set_rcparams()
try:
    df = pd.read_csv(INPUT, comment="#", skip_blank_lines=True)
    fig, ax = plt.subplots(figsize=(min(1.2 + 1.6 * len(df.columns), 14),
                                     0.45 + 0.32 * (len(df) + 1)))
    ax.axis("off")
    tbl = ax.table(cellText=df.round(4).astype(str).values,
                   colLabels=df.columns.tolist(),
                   cellLoc="center", colLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    tbl.scale(1.0, 1.25)
    for k, c in tbl.get_celld().items():
        c.set_edgecolor(PALETTE["light"])
        if k[0] == 0:
            c.set_facecolor(PALETTE["main_blue"]); c.set_text_props(color="white", weight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_PDF, transparent=True)
    fig.savefig(OUTPUT_PNG, transparent=True, dpi=300)
    plt.close(fig)
except Exception as exc:
    # Fallback: render the raw text as a monospace figure (covers T6).
    lines = text.splitlines()
    fig, ax = plt.subplots(figsize=(11, 0.25 * (len(lines) + 2)))
    ax.axis("off")
    ax.text(0.0, 1.0, text, family="monospace", fontsize=8,
            va="top", ha="left", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(OUTPUT_PDF, transparent=True)
    fig.savefig(OUTPUT_PNG, transparent=True, dpi=300)
    plt.close(fig)
    print(f"[fallback render: {exc.__class__.__name__}: {exc}]")

print(f"wrote: {OUTPUT_CSV}, {OUTPUT_PDF}, {OUTPUT_PNG}")
