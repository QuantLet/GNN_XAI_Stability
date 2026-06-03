# GNN_XAI_Stability --- Quantlet collection

[![Quantlet](https://github.com/QuantLet/Styleguide-and-FAQ/raw/master/pictures/qloqo.png)](http://quantlet.de/)

```
Name of QuantletCollection: GNN_XAI_Stability
Published in:               Expert Systems with Applications (submitted)
Description:                Do Graph Embeddings Stabilize SHAP? A Controlled
                            Decomposition under Distribution Shift.
                            Per-feature SHAP attribution stability
                            diagnostic --- Jensen-Shannon divergence with
                            block-bootstrap intervals, paired
                            block-permutation, BH adjustment, and an
                            explicit null-resampling baseline (~35% rejection
                            at nominal alpha=0.05). Nested controls
                            decompose the headline gain: diagonal-scale
                            random/permuted/noise-injected controls
                            reproduce most of the gain via column
                            augmentation; a covariance-matched control
                            absorbs the remainder on XGBoost. Validated on
                            IEEE-CIS and replicated on the Elliptic Bitcoin
                            transaction dataset (diagonal-scale controls
                            only).
Keywords:                   XAI, SHAP attribution stability, distribution
                            shift, graph neural networks, GraphSAGE,
                            permutation diagnostics, negative control,
                            null calibration, covariance-matched control,
                            fraud detection, IEEE-CIS, Elliptic, block
                            bootstrap, BH-FDR
See also:                   Diaconu and Pele 2026
Author:                     Delia Diaconu, Daniel Traian Pele
Submitted:                  2026-06-02
Datafile:                   IEEE-CIS Fraud Detection (Kaggle, public);
                            Elliptic Bitcoin transaction dataset (Kaggle)
Output:                     Quantlets producing T1-T8 tables and F1-F9
                            figures from the manuscript, plus the
                            null-resampling and covariance-matched
                            control tables (T_null_calibration,
                            T_covmatched_control)
```

## Collection layout

Strict QuantLet style. Each Quantlet folder is fully self-contained and runs
without shared infrastructure:

```
{QuantletName}/
  {QuantletName}.py        runnable Python script (helpers inlined)
  {QuantletName}.ipynb     equivalent Jupyter notebook
  Metainfo.txt             QuantLet metadata
  {QuantletName}.pdf       figure PDF (figure Quantlets F*)
  {QuantletName}.png       figure PNG (figure Quantlets F*)
  {QuantletName}.csv       table data (table Quantlets T*)
  {QuantletName}.tex       LaTeX tabular ready to \input{} (table Quantlets T*)
  <input parquet / CSV files needed by the .py>
```

All files live at the Quantlet folder root --- no subdirectories.

The 17 Quantlets:

| Quantlet | Output |
|---|---|
| F1_framework_overview         | F1 framework schematic |
| F2_js_bars_with_ci            | F2 top-30 per-feature JS bars |
| F3_frac_significant_by_config | F3 BH-flagged fraction across configurations |
| F3b_persistence_bimodal       | F3b persistence-ratio plot (real / covariance-matched / diagonal controls; no single-metric separator) |
| F6_roc_curves_with_delong     | F6 ROC curves under temporal split (AUC-only labels) |
| F7_reliability_curves         | F7 reliability curves with ECE |
| F9_layer_ablation             | F9 L1 vs L2 SAGE layer ablation |
| T1_performance                | T1 AUC / PR-AUC / Brier / ECE / DeLong (12 main configs + xgb_covmatched_emb) |
| T2_stability_summary          | T2 Algorithm 1 primary stability comparisons |
| T2b_cross_config              | T2b cross-pair diagnostic (diagonal-scale + covariance-matched) |
| T3_robustness_controls        | T3 random / permuted / noise / covariance-matched controls |
| T5_feature_class_decomposition| T5 entity vs non-entity signal effect (descriptive) |
| T6_tuned_hybrid               | T6 Optuna-tuned hybrid robustness check |
| T7_temporal_cutoff_robustness | T7 60/40, 70/30, 80/20 cutoff robustness |
| T8_elliptic_stability         | T8 Elliptic Bitcoin cross-dataset replication |
| T_null_calibration            | Null-resampling size check (xgb_tab vs xgb_tab_seedB); empirical baseline ~35% at alpha=0.05 |
| T_covmatched_control          | Covariance-matched control: xgb_tab vs xgb_covmatched_emb and xgb_covmatched_emb vs xgb_sage_l1 |

## Reproduction

```bash
python -m pip install -r requirements.txt
cd F2_js_bars_with_ci/         # or any other Quantlet folder
python F2_js_bars_with_ci.py   # or open the .ipynb in Jupyter
```

The output lands directly in the Quantlet folder root, alongside the script:
figure Quantlets re-emit `{name}.pdf` and `{name}.png`; table Quantlets
re-emit `{name}.csv` and `{name}.tex`. Every Quantlet is independent: no
shared imports, no cross-Quantlet path dependencies, no `results/`
subfolder.

## License

This collection is released under the MIT License --- a permissive open
source license that allows commercial and academic re-use, modification,
and redistribution provided the copyright notice is preserved. The full
license text is in [`LICENSE`](LICENSE). Copyright (c) 2026 Delia Diaconu
and Daniel Traian Pele.

## Citation

If you use this collection or the framework it implements, please cite:

> Diaconu, D., & Pele, D. T. (2026). *Do Graph Embeddings Stabilize
> SHAP? A Controlled Decomposition under Distribution Shift.* Expert
> Systems with Applications (submitted).

A BibTeX entry will be added to this README when the manuscript is
accepted; in the interim, repository metadata is recorded in
[`Metainfo.txt`](Metainfo.txt) and [`Quantlet.yaml`](Quantlet.yaml).
