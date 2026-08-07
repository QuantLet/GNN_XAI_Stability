<div style="margin: 0; padding: 0; text-align: center; border: none;">
<a href="https://quantlet.com" target="_blank" style="text-decoration: none; border: none;">
<img src="https://github.com/StefanGam/test-repo/blob/main/quantlet_design.png?raw=true" alt="Header Image" width="100%" style="margin: 0; padding: 0; display: block; border: none;" />
</a>
</div>

```
Name of Quantlet: GNNXAI_null_bands

Published in:    Do Graph Embeddings Stabilize SHAP? A Controlled Decomposition under Distribution Shift (submitted, Expert Systems with Applications)

Description:     Leave-one-out pairwise re-seeded null-band construction and
                 the single-panel verdict figure. The empirical null is drawn
                 from all pairwise comparisons (seed_i, seed_j), i<j, among
                 re-seeded refits of the tabular baseline (seed 42 excluded),
                 so every flagged feature is a Type I error; the 95% band of
                 the per-pair flagged fraction is the null band, computed per
                 (dataset, learner) cell under quantile bins. Firmed bands:
                 IEEE-CIS XGBoost [36.0%, 44.0%], IEEE-CIS CatBoost
                 [33.7%, 46.1%], Elliptic XGBoost [26.7%, 40.6%], Elliptic
                 CatBoost [23.5%, 54.8%]. The figure plots every headline and
                 T2b flagged fraction against its OWN cell's band: on IEEE-CIS
                 the XGBoost headline (36.0%) sits at the low end of its band
                 and the CatBoost headline (29.2%) below every one of its 66
                 draws, and all T2b cross-pairs sit at the null floor --- so no
                 hybrid stabilizes above the null and no graph-information
                 channel beyond covariance is detected. The construction script
                 (null_bands_construction.py) consumes the re-seeded refit SHAP
                 caches; the figure script (GNNXAI_null_bands.py) is fully
                 self-contained from the committed band/summary CSVs.

Keywords:        null band, leave-one-out pairwise, re-seeded refit,
                 permutation diagnostic, empirical null, negative control,
                 SHAP stability, block bootstrap, Type I error

See also:        Diaconu and Pele 2026, GNN_XAI_Stability collection;
                 manuscript Appendix E (null-resampling) and Table E.14 band map

Author:          Delia Diaconu, Daniel Traian Pele

Submitted:       2026-07-13

Datafile:        null_bands_all.csv, single_set_stability.csv,
                 T8_elliptic_stability_singleset.csv (all committed; drive the
                 figure). The per-seed re-seeded refit SHAP caches that
                 null_bands_construction.py consumes to PRODUCE null_bands_all.csv
                 are several hundred MB, git-ignored, and archived externally
                 (see the paper's Data-availability statement).

Output:          GNNXAI_null_bands.png, GNNXAI_null_bands.pdf

```
