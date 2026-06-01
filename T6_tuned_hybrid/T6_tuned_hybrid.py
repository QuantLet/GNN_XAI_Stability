"""
Quantlet T6_tuned_hybrid

Hyperparameter-tuned xgb_sage_l1_tuned via Optuna TPE (25 trials). Recovers 68% of pinned hybrid's temporal-AUC gap.

Datafile(s): T6_tuned_hybrid.csv
Output:      T6_tuned_hybrid.csv, T6_tuned_hybrid.tex

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
import io
import pandas as pd

INPUT      = "T6_tuned_hybrid.csv"
OUTPUT_CSV = "T6_tuned_hybrid.csv"
OUTPUT_TEX = "T6_tuned_hybrid.tex"

# 1. Echo the CSV content (handles T6's multi-section format).
text = Path(INPUT).read_text(encoding="utf-8")
print(text)

# 2. Re-emit the canonical CSV (no-op when input==output).
if Path(INPUT).resolve() != Path(OUTPUT_CSV).resolve():
    shutil.copyfile(INPUT, OUTPUT_CSV)

# 3. Re-emit the bundled LaTeX rendering. Split on blank lines and "# header"
#    markers so multi-section CSVs (e.g., T6) render as multiple tabulars.
NL = chr(10)
sections, current_header, current_lines = [], None, []
for raw in text.splitlines():
    if raw.startswith("#"):
        if current_lines: sections.append((current_header, current_lines)); current_lines = []
        current_header = raw.lstrip("#").strip(); continue
    if raw.strip() == "":
        if current_lines: sections.append((current_header, current_lines)); current_lines = []
        continue
    current_lines.append(raw)
if current_lines: sections.append((current_header, current_lines))

out_parts = []
for hdr, lines in sections:
    if hdr: out_parts.append(f"% --- {hdr} ---")
    try:
        df = pd.read_csv(io.StringIO(NL.join(lines)))
        out_parts.append(df.to_latex(index=False, escape=True, float_format="%.4f"))
    except Exception as e:
        out_parts.append(f"% (section could not be parsed as tabular: {e})")
        for line in lines:
            out_parts.append("% " + line)
    out_parts.append("")
Path(OUTPUT_TEX).write_text(NL.join(out_parts).rstrip() + NL, encoding="utf-8")

print(f"wrote: {OUTPUT_CSV}, {OUTPUT_TEX}")
