# GNN_XAI_Stability --- Quantlet collection

[![Quantlet](https://github.com/QuantLet/Styleguide-and-FAQ/raw/master/pictures/qloqo.png)](http://quantlet.de/)

```
Name of QuantletCollection: GNN_XAI_Stability
Published in:               Expert Systems with Applications (submitted)
Description:                Testing SHAP Stability under Distribution Shift in
                            Graph-Enhanced Fraud Detection. Per-feature
                            Jensen-Shannon explanation-stability framework
                            with block-bootstrap CIs and a paired
                            block-permutation test, validated on IEEE-CIS and
                            replicated on the Elliptic Bitcoin transaction
                            dataset.
Keywords:                   XAI, SHAP, attribution stability, distribution
                            shift, Jensen-Shannon, GraphSAGE, fraud detection,
                            IEEE-CIS, Elliptic, block bootstrap,
                            permutation test, BH-FDR, regularization vs signal
See also:                   Diaconu and Pele 2026
Author:                     Delia Diaconu, Daniel Traian Pele
Submitted:                  2026-06-01
Datafile:                   IEEE-CIS Fraud Detection (Kaggle, public);
                            Elliptic Bitcoin transaction dataset (Kaggle)
Output:                     15 Quantlets producing T1-T8 tables and
                            F1-F9 figures from the manuscript
```

## Collection layout

Strict QuantLet style. Each Quantlet folder is fully self-contained and runs
without shared infrastructure:

```
{QuantletName}/
  {QuantletName}.py        runnable Python script (helpers inlined)
  {QuantletName}.ipynb     equivalent Jupyter notebook
  Metainfo.txt             QuantLet metadata
  {QuantletName}.pdf       output figure (or {QuantletName}.csv for table Quantlets)
  {QuantletName}.png       PNG mirror (figures only)
  <input parquet / CSV files needed by the .py>
```

All files live at the Quantlet folder root --- no subdirectories.

The 15 Quantlets:

| Quantlet | Output |
|---|---|
| F1_framework_overview         | F1 framework schematic |
| F2_js_bars_with_ci            | F2 top-30 per-feature JS bars |
| F3_frac_significant_by_config | F3 BH-flagged fraction across 6 configs |
| F3b_persistence_bimodal       | F3b persistence-ratio bimodal separator |
| F6_roc_curves_with_delong     | F6 ROC curves with DeLong z |
| F7_reliability_curves         | F7 calibration curves with ECE |
| F9_layer_ablation             | F9 L1 vs L2 SAGE layer ablation |
| T1_performance                | T1 AUC / PR-AUC / Brier / ECE / DeLong |
| T2_stability_summary          | T2 Algorithm 1 primary stability comparisons |
| T2b_cross_config              | T2b residual-channel cross-pair test |
| T3_robustness_controls        | T3 random / permuted / noise controls |
| T5_feature_class_decomposition| T5 entity vs non-entity signal effect |
| T6_tuned_hybrid               | T6 Optuna-tuned hybrid robustness check |
| T7_temporal_cutoff_robustness | T7 60/40, 70/30, 80/20 cutoff robustness |
| T8_elliptic_stability         | T8 Elliptic Bitcoin cross-dataset replication |

## Reproduction

```bash
python -m pip install -r requirements.txt
cd F2_js_bars_with_ci/         # or any other Quantlet folder
python F2_js_bars_with_ci.py   # or open the .ipynb in Jupyter
```

The output lands in that Quantlet's `results/` folder. Every Quantlet is
independent: no shared imports, no cross-Quantlet path dependencies.

## License

MIT --- see `LICENSE`.

## Citation

Diaconu, D., & Pele, D. T. (2026). *Testing SHAP Stability under Distribution
Shift in Graph-Enhanced Fraud Detection.* Expert Systems with Applications
(submitted).
