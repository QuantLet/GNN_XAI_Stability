<div style="margin: 0; padding: 0; text-align: center; border: none;">
<a href="https://quantlet.com" target="_blank" style="text-decoration: none; border: none;">
<img src="https://github.com/StefanGam/test-repo/blob/main/quantlet_design.png?raw=true" alt="Header Image" width="100%" style="margin: 0; padding: 0; display: block; border: none;" />
</a>
</div>

```
Name of Quantlet: T8_elliptic_stability

Published in:    Do Graph Embeddings Stabilize SHAP? A Controlled Decomposition under Distribution Shift (submitted, Expert Systems with Applications)

Description:     Elliptic Bitcoin replication (single-set, quantile bins),
                 read against the Elliptic XGBoost leave-one-out pairwise
                 null band [26.7%, 40.6%]. XGBoost headline xgb_tab vs
                 xgb_sage_l1: delta_k=-0.0080 (32.1% flagged); diagonal-scale
                 T2b (xgb_random_emb vs xgb_sage_l1): delta_k=-0.0380 (36.4%);
                 covariance-matched T2b (xgb_covmatched_emb vs xgb_sage_l1):
                 delta_k=-0.0214 (40.0%). On the flagged-fraction axis nothing
                 clears the null; on the effect-size axis both T2b contrasts
                 fall below the null lower edge (-0.0154) --- a benchmark-specific,
                 effect-size-only residual, the opposite direction to IEEE-CIS.
                 The CatBoost arm is inconclusive: headline delta_k=+0.0425
                 (29.7%) and covariance-matched T2b delta_k=+0.0143 (32.1%) sit
                 inside a wide regenerated null, and the sign is fragile to the
                 GraphSAGE initialization seed.

Keywords:        Elliptic, Bitcoin, cross-dataset replication

See also:        Diaconu and Pele 2026, GNN_XAI_Stability collection

Author:          Delia Diaconu, Daniel Traian Pele

Submitted:       2026-06-01

Datafile:        T8_elliptic_stability.csv

Output:          T8_elliptic_stability.csv, T8_elliptic_stability.tex

```
