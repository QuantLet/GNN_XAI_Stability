<div style="margin: 0; padding: 0; text-align: center; border: none;">
<a href="https://quantlet.com" target="_blank" style="text-decoration: none; border: none;">
<img src="https://github.com/StefanGam/test-repo/blob/main/quantlet_design.png?raw=true" alt="Header Image" width="100%" style="margin: 0; padding: 0; display: block; border: none;" />
</a>
</div>

```
Name of Quantlet:GNNXAI_estimator_adjudication

Published in:    Do Graph Embeddings Stabilize SHAP? A Controlled Decomposition under
                 Distribution Shift (submitted, Expert Systems with Applications)

Description:     Bin-free estimator adjudication and the attribution-scale confound. Under
                 equal-width bins the IEEE-CIS XGBoost headline clears its null, and so do
                 two estimators that have no bins at all (1-Wasserstein 61.7% against
                 [44.1%, 50.7%]; Cramer/energy 57.6% against [41.5%, 49.0%]), so the
                 disagreement cannot be dismissed as a binning artifact. What separates the
                 estimators is scale sensitivity: the embedding columns absorb 37-38% of
                 attribution mass, shrinking the hybrid's tabular attributions to a median
                 0.627 of the baseline's, and a pure rescaling of that size accounts for
                 111% of the observed mean effect, correlates with it at +0.837, and --
                 once removed by normalising each configuration by its own mean |phi| --
                 turns the share of features with delta_k < 0 from 72.5% into 45.9%, a coin
                 flip. The controls confirm the mechanism with a dose-response. The primary
                 specification is therefore chosen on an identified defect of the
                 alternatives, not on taste.

Keywords:        estimator invariance, Wasserstein, Cramer distance, energy distance, scale
                 equivariance, attribution dilution, binning, specification sensitivity

See also:        GNNXAI_scale_invariance; manuscript Section 5.7

Author:          Delia Diaconu, Daniel Traian Pele

Submitted:       2026-08-01

Datafile:        binning_adjudication.csv, scale_confound.csv (both committed; drive the
                 figure). Produced by scripts/binning_adjudication.py and
                 scripts/scale_confound.py from the per-feature SHAP caches.

Output:          GNNXAI_estimator_adjudication.png, GNNXAI_estimator_adjudication.pdf

```
