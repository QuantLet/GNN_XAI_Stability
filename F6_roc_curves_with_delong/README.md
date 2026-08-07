<div style="margin: 0; padding: 0; text-align: center; border: none;">
<a href="https://quantlet.com" target="_blank" style="text-decoration: none; border: none;">
<img src="https://github.com/StefanGam/test-repo/blob/main/quantlet_design.png?raw=true" alt="Header Image" width="100%" style="margin: 0; padding: 0; display: block; border: none;" />
</a>
</div>

```
Name of Quantlet: F6_roc_curves_with_delong

Published in:    Do Graph Embeddings Stabilize SHAP? A Controlled Decomposition under Distribution Shift (submitted, Expert Systems with Applications)

Description:     ROC curves under the temporal split for the 13 main
                 IEEE-CIS tree-learner configurations: two tabular
                 baselines, sage_only, three hybrids, six diagonal-scale
                 controls, and the covariance-matched control
                 xgb_covmatched_emb (drawn in amber). Labels carry AUC
                 only; formal within-split DeLong comparisons use
                 same-learner references (XGBoost vs xgb_tab, CatBoost
                 vs catb_tab) and live in T1_performance, not in this
                 figure.

Keywords:        ROC, AUC, temporal split, multi-configuration,
                 covariance-matched control

See also:        Diaconu and Pele 2026, GNN_XAI_Stability collection

Author:          Delia Diaconu, Daniel Traian Pele

Submitted:       2026-06-03

Datafile:        preds_{config}_temporal.parquet, 13 files for the main
                 IEEE-CIS configurations

Output:          F6_roc_curves_with_delong.pdf,
                 F6_roc_curves_with_delong.png

```
