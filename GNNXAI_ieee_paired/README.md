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

The first command checks that changing held-out data leaves the training representation unchanged. The next two fit the models. The last three regenerate the exhibits from the saved fits. Compare the resulting CSV files in results/strict_ieee/primary with reference/. The working output and reference directories are separate to allow fresh fitting. GPU and software differences can affect numerical reproducibility.

The small reference tables also permit direct inspection without retraining. To regenerate the first two figures from the reference summaries, run:

    python -c "from pathlib import Path; import pandas as pd; from scripts.plot_strict_ieee_comparisons import plot_estimates; plot_estimates(pd.read_csv('reference/paired.csv',index_col='seed'),pd.read_csv('reference/plot_comparisons_by_seed.csv',index_col='seed'),Path('submission/source/figures'))"

Code for the two extra normalization contrasts and the top-ten plot is in scripts/plot_strict_ieee_comparisons.py.
