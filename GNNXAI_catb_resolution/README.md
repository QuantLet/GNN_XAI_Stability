<div style="margin: 0; padding: 0; text-align: center; border: none;">
<a href="https://quantlet.com" target="_blank" style="text-decoration: none; border: none;">
<img src="https://github.com/StefanGam/test-repo/blob/main/quantlet_design.png?raw=true" alt="Header Image" width="100%" style="margin: 0; padding: 0; display: block; border: none;" />
</a>
</div>

```
Name of Quantlet:GNNXAI_catb_resolution

Published in:    Do Graph Embeddings Stabilize SHAP? A Controlled Decomposition under
                 Distribution Shift (submitted, Expert Systems with Applications)

Description:     The CatBoost embedding-conditioning resolution floor. Perturbing the
                 frozen embedding by a relative 1e-7 -- below float32 machine epsilon, i.e.
                 below the precision at which the embedding is stored -- moves the CatBoost
                 mean delta_k by +0.0137, comparable to the point estimate itself (+0.0141)
                 and to a quarter of the width of its own null band. The effective
                 resolution of a CatBoost effect size is therefore of order 0.014, and the
                 paper interprets every CatBoost effect size at two decimals rather than
                 the four it deposits. The same probe on the calibration axis shows the
                 paired ECE gap is likewise unresolved, including across repeat runs at
                 zero perturbation. The XGBoost cells do not carry this caveat: the
                 amplifier is CatBoost's discontinuous split selection, which the probe
                 localizes.

Keywords:        numerical conditioning, float32 epsilon, CatBoost, split selection,
                 reproducibility, expected calibration error, resolution floor

See also:        GNNXAI_null_bands; manuscript Section 5.1 and Supplementary Section S5

Author:          Delia Diaconu, Daniel Traian Pele

Submitted:       2026-08-01

Datafile:        T_ieee_catb_conditioning.csv, ece_eps_probe.csv (both committed; drive the
                 figure). Produced by scripts/ieee_catb_conditioning.py and
                 scripts/ece_eps_probe.py, which refit CatBoost on the cached inputs.

Output:          GNNXAI_catb_resolution.png, GNNXAI_catb_resolution.pdf

```
