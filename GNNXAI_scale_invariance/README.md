<div style="margin: 0; padding: 0; text-align: center; border: none;">
<a href="https://quantlet.com" target="_blank" style="text-decoration: none; border: none;">
<img src="https://github.com/StefanGam/test-repo/blob/main/quantlet_design.png?raw=true" alt="Header Image" width="100%" style="margin: 0; padding: 0; display: block; border: none;" />
</a>
</div>

```
Name of Quantlet:GNNXAI_scale_invariance

Published in:    Do Graph Embeddings Stabilize SHAP? A Controlled Decomposition under
                 Distribution Shift (submitted, Expert Systems with Applications)

Description:     Direct scale-invariance test of the primary estimator. The quantile-bin
                 diagnostic sets its bin edges from the pooled sample, so shrinking one arm
                 does perturb them and exact invariance is not guaranteed. Rescaling the
                 hybrid's tabular attributions to the baseline's magnitude moves the
                 quantile flagged fraction by at most 1.9 points and the panel mean by at
                 most 0.00025, against -17.7 points and a sign reversal for equal-width;
                 and it moves quantile-JS up while moving equal-width sharply down, whereas
                 an estimator manufacturing stabilization from dilution must lose it when
                 the dilution is removed. The quantile specification is therefore invariant
                 enough, and in the wrong direction, to be the artifact.

Keywords:        scale invariance, quantile binning, equal-width binning, positive control,
                 attribution dilution, estimator diagnostics

See also:        GNNXAI_estimator_adjudication; manuscript Supplementary Section S6

Author:          Delia Diaconu, Daniel Traian Pele

Submitted:       2026-08-01

Datafile:        quantile_scale_invariance.csv (committed; drives the figure). Produced by
                 scripts/quantile_scale_invariance.py from the per-feature SHAP caches.

Output:          GNNXAI_scale_invariance.png, GNNXAI_scale_invariance.pdf

```
