# Reproducing the IEEE-CIS analysis

The reference folder contains the reported seed-level results, the scale and CatBoost controls, and the run manifest. The original transactions are available after accepting the [IEEE-CIS competition rules](https://www.kaggle.com/competitions/ieee-fraud-detection/rules) on [Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data). Section 7.B restricts redistribution of those files. Download train_transaction.csv and train_identity.csv to data/raw/ for a fresh run. The manifest supplies the expected file hashes.

Run `python GNNXAI_ieee_paired.py` to reproduce the primary comparison from the deposited CSVs without downloading transactions. The `reference/predictive_by_seed.csv` file contains the held-out AUC and average precision for each XGBoost fit. The published exhibit sources are under `submission/source/`.

For a fresh run, use Python 3.13 and install requirements.txt. Run the following commands from the extracted project root, in this order:

    python scripts/check_strict_ieee.py
    python scripts/strict_ieee.py
    python scripts/strict_ieee_controls.py
    python scripts/render_strict_ieee_table.py
    python scripts/strict_ieee_normalized.py
    python scripts/plot_strict_ieee_comparisons.py
    python scripts/report_ieee_context.py

The first command checks that changing held-out data leaves the training representation unchanged. The next two fit the models. The remaining scripts regenerate the exhibits from the saved fits. Compare the resulting CSV files in results/strict_ieee/primary with reference/. The working output and reference directories are separate to allow fresh fitting. GPU and software differences can affect numerical reproducibility.

The small reference tables also permit direct inspection without retraining. To regenerate the first two figures from the reference summaries, run:

    python -c "from pathlib import Path; import pandas as pd; from scripts.plot_strict_ieee_comparisons import plot_estimates; plot_estimates(pd.read_csv('reference/paired.csv',index_col='seed'),pd.read_csv('reference/plot_comparisons_by_seed.csv',index_col='seed'),Path('submission/source/figures'))"

For R8, `reference/context_by_seed.csv` contains the two absolute JS levels on the primary graph/Gaussian pair's common bin edges, their difference, the share of absolute SHAP mass on the added 64 coordinates, forward-time AUC and the graph-minus-tabular Wasserstein contrast on a shared feature scale. JS averages 486 fields using 5,000 observations per regime. The mass denominator includes all inputs in the respective model and pools both regimes. AUC uses all 177,162 held-out temporal transactions. The summary gives means and pointwise Student intervals over the 20 seeds (19 degrees of freedom), conditional on the cutoff. All columns can be regenerated from the fitted checkpoints. To reproduce the summary and Table 2 from the public CSVs alone, run:

    python scripts/report_ieee_context.py --directory reference --summary-only --output results/check --table-dir results/check

Code for the two extra normalization contrasts and the top-ten plot is in scripts/plot_strict_ieee_comparisons.py.

To reproduce the exploratory top-three and top-five agreement table, run
`python scripts/strict_ieee_shortlists.py --directory reference --summary-only --output results/check`.
The seed-level file gives the fraction of the tabular model's leading fields retained by each alternative on the same 5,000 payments, among the original 486 fields. The script checks the ten-field result against the earlier deposit. The four shorter-list contrasts use Bonferroni simultaneous 95% intervals over paired seed differences. To recompute the overlaps from the saved SHAP arrays after fitting, omit the flags.
