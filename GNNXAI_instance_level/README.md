<div style="margin: 0; padding: 0; text-align: center; border: none;">
<a href="https://quantlet.com" target="_blank" style="text-decoration: none; border: none;">
<img src="https://github.com/StefanGam/test-repo/blob/main/quantlet_design.png?raw=true" alt="Header Image" width="100%" style="margin: 0; padding: 0; display: block; border: none;" />
</a>
</div>

```
Name of Quantlet:GNNXAI_instance_level

Published in:    Do Graph Embeddings Stabilize SHAP? A Controlled Decomposition under
                 Distribution Shift (submitted, Expert Systems with Applications)

Description:     Per-row explanation agreement against both nulls. The marginal diagnostic
                 is necessary but not sufficient for explanation comparability, and because
                 both configurations are explained on the same evaluation rows the gap can
                 be measured. The mean per-row top-10 overlap is 0.686 (random) and 0.677
                 (temporal) with Spearman 0.86/0.84, against a re-seeded null where two
                 fits of the same configuration agree at 0.761/0.769 -- a clear excess, so
                 a null-level marginal verdict does not license the claim that individual
                 explanations are reproducible. Against the shape-matched null most of the
                 movement is attributable to adding columns at all, and on the random split
                 the observed value falls inside the band; on the temporal split it does
                 not, with the overlap below [0.716, 0.737] and Spearman below [0.854,
                 0.870]. This is the one place in the study where the graph embedding
                 separates from a covariance-matched control, and it is a reshuffling of
                 per-row ordering, not a stability gain.

Keywords:        instance-level explanations, top-k overlap, Spearman correlation,
                 explanation reproducibility, model risk, SR 11-7, covariance-matched
                 control

See also:        GNNXAI_null_bands_matched; manuscript Section 5.5

Author:          Delia Diaconu, Daniel Traian Pele

Submitted:       2026-08-01

Datafile:        instance_level_agreement.csv, instance_level_null_band_matched.csv (both
                 committed; drive the figure). Produced by scripts/instance_level_check.py
                 from the per-row SHAP caches.

Output:          GNNXAI_instance_level.png, GNNXAI_instance_level.pdf

```
