<div style="margin: 0; padding: 0; text-align: center; border: none;">
<a href="https://quantlet.com" target="_blank" style="text-decoration: none; border: none;">
<img src="https://github.com/StefanGam/test-repo/blob/main/quantlet_design.png?raw=true" alt="Header Image" width="100%" style="margin: 0; padding: 0; display: block; border: none;" />
</a>
</div>

```
Name of Quantlet: F1_framework_overview

Published in:    Do Graph Embeddings Stabilize SHAP? A Controlled Decomposition under Distribution Shift (submitted, Expert Systems with Applications)

Description:     Schematic of the audit design, in two panels. (a) The
                 inferential unit is the COMPLETE REFIT: a single paired fit of
                 baseline against hybrid yields one delta_bar, but that number
                 carries ~40% training-RNG noise and is a screen only; every
                 stage (holdout, features, leakage-controlled graph, GraphSAGE,
                 both tree fits) is re-drawn per seed, each refit contributes one
                 paired contrast, and the distribution supports ordinary interval
                 estimation across independent units -- no empirical band and no
                 per-feature p-value. (b) The negative controls form a ladder:
                 each rung adds one property of the real embedding (dimension,
                 per-column marginals, second-order covariance, attribution
                 dilution), so a residual against any rung is still attributable
                 to everything below it, and only the top rung adds graph
                 structure. Replaces an earlier single-fit pipeline schematic,
                 which depicted precisely the object the paper argues is
                 uninterpretable on its own, and whose red/green pair was
                 indistinguishable under deuteranopia (OKLab dE 3.8 against a
                 floor of 8). The palette is an Okabe-Ito subset with worst
                 pairwise CVD separation 11.2.

Keywords:        audit design, inferential unit, complete refit, negative control
                 ladder, re-seeded null, attribution dilution, colour-vision
                 accessibility, SHAP stability

See also:        Diaconu and Pele 2026, GNN_XAI_Stability collection

Author:          Delia Diaconu, Daniel Traian Pele

Submitted:       2026-06-01

Datafile:        (none -- standalone)

Output:          F1_framework_overview.pdf

```
