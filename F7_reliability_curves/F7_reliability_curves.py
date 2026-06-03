"""
Quantlet F7_reliability_curves

Reliability (calibration) curves under the temporal split for the 12 main
IEEE-CIS configurations (two tabular baselines, sage_only, three hybrids,
six controls). ECE annotations are included in the legend.

Datafile(s): calibration_{config}_temporal.parquet (12 files)
Output:      F7_reliability_curves.pdf, F7_reliability_curves.png

Run from inside this folder:
    python F7_reliability_curves.py
or open F7_reliability_curves.ipynb in Jupyter.
"""
from pathlib import Path

PALETTE = {
    "main_blue": "#003DA5",
    "ida_red":   "#C8102E",
    "forest":    "#228B22",
    "amber":     "#D49B00",
    "crimson":   "#DC143C",
    "grey":      "#777777",
    "light":     "#CCCCCC",
}
LINESTYLES = ["-", "--", ":", "-."]


def set_rcparams():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25,
        "grid.linestyle": "--", "grid.linewidth": 0.5,
        "legend.frameon": False, "legend.fontsize": 8,
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


def legend_outside_bottom(ax, ncol=3, y_offset=-0.28, fontsize=7):
    h, l = ax.get_legend_handles_labels()
    if not h:
        return
    ax.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, y_offset),
              ncol=ncol, frameon=False, fontsize=fontsize)


import pandas as pd
import matplotlib.pyplot as plt

set_rcparams()

MAIN_CONFIGS = (
    "xgb_tab", "catb_tab", "sage_only",
    "xgb_sage_l1", "xgb_sage_l2", "catb_sage_l1",
    "xgb_random_emb", "xgb_permuted_emb", "xgb_noise_inj",
    "catb_random_emb", "catb_permuted_emb", "catb_noise_inj",
    "xgb_covmatched_emb",
)
files = [Path(f"calibration_{c}_temporal.parquet") for c in MAIN_CONFIGS]
files = [f for f in files if f.exists()]

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], "--", color=PALETTE["light"], label="perfect")
for i, p in enumerate(files):
    df = pd.read_parquet(p)
    config = p.stem.removeprefix("calibration_").rsplit("_temporal", 1)[0]
    if "covmatched_emb" in config:
        color = PALETTE["amber"]
    elif any(t in config for t in ("random_emb", "permuted_emb", "noise_inj")):
        color = PALETTE["forest"]
    elif "tab" in config:
        color = PALETTE["main_blue"]
    else:
        color = PALETTE["ida_red"]
    w = df["count"] / df["count"].sum()
    ece = (w * (df["acc"] - df["conf"]).abs()).sum()
    ax.plot(df["conf"], df["acc"], "-o", color=color, markersize=3,
            linestyle=LINESTYLES[i % len(LINESTYLES)],
            label=f"{config} ECE={ece:.3f}")
ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
ax.set_title("Reliability curves (temporal split)")
legend_outside_bottom(ax, ncol=3, y_offset=-0.28, fontsize=7)
pdf, png = save_fig(fig, "F7_reliability_curves", ".")
plt.close(fig)
print("saved:", pdf)
