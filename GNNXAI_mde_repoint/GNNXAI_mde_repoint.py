"""GNNXAI_mde_repoint -- minimum detectable effect against both band edges

Self-contained: reads only the committed CSVs shipped in this Quantlet folder.
mde_repointed.csv is produced by scripts/mde_repoint.py.

Output: GNNXAI_mde_repoint.png (+ .pdf)
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


NAME = "GNNXAI_mde_repoint"
M = pd.read_csv(HERE / "mde_repointed.csv")


def main():
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    y = np.arange(len(M))
    obs = float(M.observed_delta.iloc[0])
    for i, r in M.iterrows():
        ax.barh(i, abs(r.mde_delta), height=0.42, color=C_BAND, alpha=0.55)
        ax.text(abs(r.mde_delta) + 0.0009, i,
                f"MDE {r.mde_delta:+.4f}  ({r.ratio_mde_to_observed:.1f}x observed)",
                va="center", fontsize=8.5)
    ax.axvline(abs(obs), color=C_OBS, lw=2,
               label=f"observed headline ({obs:+.4f})")
    ax.set_yticks(y)
    ax.set_yticklabels([r.reference.replace("_", " ") for _, r in M.iterrows()],
                       fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.045)
    ax.set_xlabel(r"$|\bar{\delta}_k|$ needed to clear the band edge", fontsize=9.5)
    ax.set_title("The correctly shaped reference makes the rule more sensitive\n"
                 "but leaves less headroom", fontsize=10)
    ax.legend(fontsize=8.5, loc="lower right")
    _save(fig, NAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
