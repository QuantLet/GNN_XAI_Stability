"""
Quantlet F_null_pvalue_hist

Pooled per-feature raw block-permutation p-values under the empirical
re-seeded-baseline null (xgb_tab vs re-seeded xgb_tab; quantile bins, daily
blocks, seeds 10-14 vs the seed-42 reference; 2,430 = 486 x 5 feature-level
values). A calibrated test would be uniform (dashed line); the observed
distribution concentrates near zero, and the annotated fraction below 0.05
quantifies the diagnostic's anti-conservatism.

Datafile(s): F_null_pvalue_hist.csv  (the 2,430 pooled raw p-values)
Output:      F_null_pvalue_hist.pdf

Provenance: the original per-feature quantile p-values were lost in the disk crash
(see ../../../STATUS.md); F_null_pvalue_hist.csv is regenerated from the reproducible
pipeline (scripts/null_refit_lean.py + scripts/null_analysis.py). The below-0.05
fraction is the reconstruction's; see ../../../REVISION_LOG.md.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = np.loadtxt("F_null_pvalue_hist.csv", delimiter=",", skiprows=1)
frac = float((p < 0.05).mean())

fig, ax = plt.subplots(figsize=(7.5, 5.0))
ax.hist(p, bins=20, range=(0, 1), color="#C8102E", alpha=0.8,
        edgecolor="black", linewidth=0.5)
ax.axhline(len(p) / 20, color="#003DA5", linestyle="--", lw=1.5,
           label="uniform (calibrated) reference")
ax.axvline(0.05, color="#777777", linestyle=":", lw=1.2)
ax.set_xlabel("raw block-permutation $p$-value")
ax.set_ylabel("# feature-level values")
ax.set_title("Re-seeded-baseline null: pooled per-feature raw $p$-values")
ax.text(0.5, 0.9, f"{frac*100:.1f}% below 0.05\n(n={len(p)}, seeds 10--14)",
        transform=ax.transAxes, fontsize=10)
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig("F_null_pvalue_hist.pdf")
fig.savefig("F_null_pvalue_hist.png", dpi=200)
plt.close(fig)
print(f"saved F_null_pvalue_hist.pdf; {frac*100:.1f}% below 0.05 (n={len(p)})")
