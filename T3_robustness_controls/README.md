<div style="margin: 0; padding: 0; text-align: center; border: none;">
<a href="https://quantlet.com" target="_blank" style="text-decoration: none; border: none;">
<img src="https://github.com/StefanGam/test-repo/blob/main/quantlet_design.png?raw=true" alt="Header Image" width="100%" style="margin: 0; padding: 0; display: block; border: none;" />
</a>
</div>

```
Name of Quantlet: T3_robustness_controls

Published in:    Do Graph Embeddings Stabilize SHAP? A Controlled Decomposition under Distribution Shift (submitted, Expert Systems with Applications)

Description:     Four robustness control configurations against
                 xgb_tab: random embedding (diagonal scale), permuted
                 embedding (diagonal scale, column marginals preserved),
                 noise-injected embedding (small inference-time
                 perturbation of real L1), and covariance-matched
                 embedding (N(0, Sigma_hat) matching the full 64x64
                 covariance of the real layer-1 embedding). The
                 covariance-matched control is the decisive comparison:
                 it reproduces a larger stability gain than the real
                 sage_l1 hybrid and absorbs the diagonal-scale residual
                 in the cross-pair T2b diagnostic.

Keywords:        robustness controls, random, permuted, noise injection,
                 covariance-matched control, negative control

See also:        Diaconu and Pele 2026, GNN_XAI_Stability collection

Author:          Delia Diaconu, Daniel Traian Pele

Submitted:       2026-06-02

Datafile:        T3_robustness_controls.csv

Output:          T3_robustness_controls.csv, T3_robustness_controls.tex

```
