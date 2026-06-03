"""
Quantlet F3b_persistence_bimodal

Attribution-persistence-ratio plot per configuration across three groups:
  - real embedding (red): xgb_sage_l1/l2, catb_sage_l1, *_noise_inj
  - covariance-matched control (amber): xgb_covmatched_emb (~0.55,
    between the two clusters)
  - diagonal/permutation controls (green): xgb/catb random_emb, permuted_emb

The metric separates the diagonal-scale controls from the embedding-bearing
configurations, but the covariance-matched control lands between the two
bands, so the ratio does NOT isolate graph structure as a single-metric
separator. Reported descriptively.

Datafile(s): T1b_attribution_persistence.csv
Output:      F3b_persistence_bimodal.pdf, F3b_persistence_bimodal.png

Run from inside this folder:
    python F3b_persistence_bimodal.py
or open F3b_persistence_bimodal.ipynb in Jupyter.
"""
from pathlib import Path

PALETTE = {
    "main_blue": "#003DA5",
    "ida_red":   "#C8102E",   # real embedding
    "forest":    "#228B22",   # diagonal/permutation controls
    "amber":     "#D49B00",   # covariance-matched control
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


import pandas as pd
import matplotlib.pyplot as plt

set_rcparams()
df = pd.read_csv("T1b_attribution_persistence.csv").set_index("config")

real_configs = [c for c in ("xgb_sage_l1", "xgb_sage_l2", "catb_sage_l1",
                             "xgb_noise_inj", "catb_noise_inj")
                if c in df.index]
cov_configs  = [c for c in ("xgb_covmatched_emb",) if c in df.index]
ctrl_configs = [c for c in ("xgb_random_emb", "xgb_permuted_emb",
                             "catb_random_emb", "catb_permuted_emb")
                if c in df.index]

# Within each group, sort by ratio descending; stack groups so amber
# lands visibly between red and green.
def _by_ratio_desc(cs):
    return sorted(cs, key=lambda c: df.loc[c, "persistence_ratio"], reverse=True)

real_configs = _by_ratio_desc(real_configs)
cov_configs  = _by_ratio_desc(cov_configs)
ctrl_configs = _by_ratio_desc(ctrl_configs)

# barh draws first-passed at the BOTTOM, so feed bottom-up.
order_bottom_up = ctrl_configs + cov_configs + real_configs

fig, ax = plt.subplots(figsize=(8, 5.4))
for cfg in order_bottom_up:
    val = float(df.loc[cfg, "persistence_ratio"])
    if cfg in real_configs:
        color, label = PALETTE["ida_red"], "real embedding"
    elif cfg in cov_configs:
        color, label = PALETTE["amber"], "covariance-matched control"
    else:
        color, label = PALETTE["forest"], "diagonal/permutation control"
    ax.barh(cfg, val, color=color, label=label)
    ax.text(val + 0.01, cfg, f"{val:.2f}", va="center", fontsize=8)

ax.axvline(1.0, ls=":", color=PALETTE["light"], linewidth=0.7)

# Deduplicate legend; fix order red, amber, green.
handles, labels = ax.get_legend_handles_labels()
seen = {}
for h, l in zip(handles, labels):
    if l not in seen:
        seen[l] = h
legend_order = ["real embedding", "covariance-matched control",
                "diagonal/permutation control"]
legend_order = [l for l in legend_order if l in seen]
ax.legend([seen[l] for l in legend_order], legend_order,
          loc="lower right", frameon=False, fontsize=8)

xmax = max(df.loc[order_bottom_up, "persistence_ratio"].max(), 1.1) * 1.05
ax.set_xlim(0, xmax)
ax.set_xlabel("Attribution persistence ratio (temporal / random)")
ax.set_title("Attribution persistence ratio per configuration")
pdf, png = save_fig(fig, "F3b_persistence_bimodal", ".")
plt.close(fig)
print("saved:", pdf)
