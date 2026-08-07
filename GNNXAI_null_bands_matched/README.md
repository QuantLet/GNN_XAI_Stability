<div style="margin: 0; padding: 0; text-align: center; border: none;">
<a href="https://quantlet.com" target="_blank" style="text-decoration: none; border: none;">
<img src="https://github.com/StefanGam/test-repo/blob/main/quantlet_design.png?raw=true" alt="Header Image" width="100%" style="margin: 0; padding: 0; display: block; border: none;" />
</a>
</div>

```
Name of Quantlet:GNNXAI_null_bands_matched

Published in:    Do Graph Embeddings Stabilize SHAP? A Controlled Decomposition under
                 Distribution Shift (submitted, Expert Systems with Applications)

Description:     Shape-matched re-seeded null bands. The committed pairwise band re-seeds a
                 486-column baseline against itself, but the headline contrast compares 486
                 columns with 550 and the T2b contrast compares two 550-column
                 configurations. Under a strict 'the graph adds nothing' null there is no
                 reason those shapes share an expected flagged fraction, so the references
                 are built directly by re-seeding the covariance-matched control, whose 64
                 extra columns match the real embedding's full covariance and carry no
                 graph signal. The 486-vs-550 null is materially lower ([29.9%, 38.0%] over
                 66 off-diagonal draws), moving the XGBoost headline from its band floor to
                 the 77th percentile, while the 550-vs-550 null is nearly identical to the
                 same-dimension band, so every T2b reading is unaffected. Bands are
                 reported on the full 486 support and on the live common support.

Keywords:        null band, shape matching, covariance-matched control, re-seeded refit,
                 leave-one-out pairwise, dimensionality, SHAP stability

See also:        GNNXAI_null_bands; GNNXAI_degeneracy_support; manuscript Appendix E

Author:          Delia Diaconu, Daniel Traian Pele

Submitted:       2026-08-01

Datafile:        null_bands_matched.csv, null_bands_all_regen.csv (both committed; drive
                 the figure). Produced by scripts/null_refit_covmatched.py and
                 scripts/null_bands_matched.py from the re-seeded refit caches, which are
                 git-ignored and archived externally.

Output:          GNNXAI_null_bands_matched.png, GNNXAI_null_bands_matched.pdf

```
