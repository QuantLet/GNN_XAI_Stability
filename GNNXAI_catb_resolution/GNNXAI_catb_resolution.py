"""GNNXAI_catb_resolution -- the CatBoost embedding-conditioning resolution floor

Self-contained: reads only the committed CSVs shipped in this Quantlet folder.
T_ieee_catb_conditioning.csv and ece_eps_probe.csv are produced by
scripts/ieee_catb_conditioning.py and scripts/ece_eps_probe.py.

Output: GNNXAI_catb_resolution.png (+ .pdf)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
C_OBS, C_BAND, C_ALT = "#c0392b", "#7f8c8d", "#2c7fb8"


def _save(fig, name):
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"{name}.{ext}", dpi=160, bbox_inches="tight")
    print("wrote", HERE / f"{name}.png")


NAME = "GNNXAI_catb_resolution"
C = pd.read_csv(HERE / "T_ieee_catb_conditioning.csv")
E = pd.read_csv(HERE / "ece_eps_probe.csv")


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.8))

    base = float(C[C.eps == 0].mean_delta.iloc[0])
    nz = C[C.eps > 0]
    ax1.axhline(base, color=C_BAND, ls="--", lw=1.4,
                label=fr"unperturbed $\bar{{\delta}}_k$ = {base:+.4f}")
    ax1.semilogx(nz.rel_perturbation, nz.mean_delta, "o-", color=C_OBS, lw=2)
    shift = float(nz.mean_delta.iloc[0]) - base
    ax1.annotate(f"{shift:+.4f} at a\nrelative $10^{{-7}}$",
                 xy=(float(nz.rel_perturbation.iloc[0]), float(nz.mean_delta.iloc[0])),
                 xytext=(3e-7, base + 0.004), fontsize=8.5,
                 arrowprops=dict(arrowstyle="->", lw=1))
    ax1.set_xlabel("relative embedding perturbation", fontsize=9)
    ax1.set_ylabel(r"mean $\bar{\delta}_k$", fontsize=9)
    ax1.set_title("(a) below float32 epsilon, the point estimate moves", fontsize=9.5)
    ax1.legend(fontsize=8)

    r = np.arange(len(E))
    ax2.bar(r, E.diff_tab_minus_hyb, color=[C_BAND if e == 0 else C_ALT for e in E.eps])
    ax2.set_xticks(r)
    ax2.set_xticklabels([("rep." if e == 0 else f"{e:g}") for e in E.eps], fontsize=8)
    ax2.set_xlabel(r"repeat at $\epsilon=0$ / perturbation $\epsilon$", fontsize=9)
    ax2.set_ylabel("paired ECE gap (tab $-$ hyb)", fontsize=9)
    ax2.set_title("(b) the ECE gap is not resolved either", fontsize=9.5)

    _save(fig, NAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
