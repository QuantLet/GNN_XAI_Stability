<div style="margin: 0; padding: 0; text-align: center; border: none;">
<a href="https://quantlet.com" target="_blank" style="text-decoration: none; border: none;">
<img src="https://github.com/StefanGam/test-repo/blob/main/quantlet_design.png?raw=true" alt="Header Image" width="100%" style="margin: 0; padding: 0; display: block; border: none;" />
</a>
</div>

```
Name of Quantlet: F3_frac_significant_by_config

Published in:    Do Graph Embeddings Stabilize SHAP? A Controlled Decomposition under Distribution Shift (submitted, Expert Systems with Applications)

Description:     Per-configuration fraction of features flagged by the
                 BH-ranked block-permutation diagnostic across the six
                 canonical hybrid-vs-tabular and column-augmentation
                 control-vs-tabular comparisons. The diagnostic is read
                 against the learner-specific leave-one-out pairwise
                 re-seeded null bands (IEEE-CIS XGBoost [35.8%, 44.1%],
                 CatBoost [33.7%, 46.1%]; mean ~40% at nominal alpha=0.05)
                 rather than as a calibrated false-discovery rate. Under
                 those bands the XGBoost headline (36.0%) sits at the low
                 end and the CatBoost headline (29.2%) below all draws ---
                 neither hybrid stabilizes above the null. The metric captures per-feature
                 distributional change but does not isolate graph
                 information; residual real-embedding effects are
                 examined directly via the T2b cross-pair contrasts in
                 T3_robustness_controls.

Keywords:        bar chart, BH-ranked diagnostic, block bootstrap,
                 configuration sweep, empirical null

See also:        Diaconu and Pele 2026, GNN_XAI_Stability collection

Author:          Delia Diaconu, Daniel Traian Pele

Submitted:       2026-06-03

Datafile:        stability_xgb_tab_vs_xgb_sage_l1_block_time.parquet,
                 stability_xgb_tab_vs_xgb_sage_l2_block_time.parquet,
                 stability_xgb_tab_vs_xgb_noise_inj_block_time.parquet,
                 stability_xgb_tab_vs_xgb_random_emb_block_time.parquet,
                 stability_xgb_tab_vs_xgb_permuted_emb_block_time.parquet,
                 stability_catb_tab_vs_catb_sage_l1_block_time.parquet

Output:          F3_frac_significant_by_config.pdf,
                 F3_frac_significant_by_config.png

```
