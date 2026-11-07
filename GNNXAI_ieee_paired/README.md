# Reproducing the IEEE-CIS analysis

The reference folder contains the reported seed-level results, the scale and CatBoost controls, and the run manifest. The original transactions are available after accepting the [IEEE-CIS competition rules](https://www.kaggle.com/competitions/ieee-fraud-detection/rules) on [Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data). Section 7.B restricts redistribution of those files. Download train_transaction.csv and train_identity.csv to data/raw/ for a fresh run. The manifest supplies the expected file hashes.

Run `python GNNXAI_ieee_paired.py` to reproduce the primary comparison from the deposited CSVs without downloading transactions. The `reference/predictive_by_seed.csv` file contains the held-out AUC and average precision for each XGBoost fit. The published exhibit sources are under `submission/source/`.

For a fresh run, use Python 3.13 and install requirements.txt. Run the following commands from the extracted project root, in this order:

    python scripts/check_strict_ieee.py --output results/check/label_only_check.json
    python scripts/strict_ieee.py
    python scripts/strict_ieee_controls.py
    python scripts/render_strict_ieee_table.py
    python scripts/strict_ieee_normalized.py
    python scripts/plot_strict_ieee_comparisons.py
    python scripts/report_ieee_context.py

The first command reruns the graph on the same 1,200 transactions with the 360 later labels flipped and all input fields fixed (seed 201, two epochs). It checks every coordinate of the historical and held-out embeddings against the unaltered run. The aggregate result is in reference/label_only_check.json. A separate perturbation changes both labels and fields to check that the historical representation stays fixed. The next two fit the models. The remaining scripts regenerate the exhibits from the saved fits. Compare the resulting CSV files in results/strict_ieee/primary with reference/. The working output and reference directories are separate to allow fresh fitting. GPU and software differences can affect numerical reproducibility.

The small reference tables also permit direct inspection without retraining. To regenerate the first two figures from the reference summaries, run:

    python -c "from pathlib import Path; import pandas as pd; from scripts.plot_strict_ieee_comparisons import plot_estimates; plot_estimates(pd.read_csv('reference/paired.csv',index_col='seed'),pd.read_csv('reference/plot_comparisons_by_seed.csv',index_col='seed'),Path('submission/source/figures'))"

For R8, `reference/context_by_seed.csv` contains the two absolute JS levels on the primary graph/Gaussian pair's common bin edges, their difference, the share of absolute SHAP mass on the added 64 coordinates, forward-time AUC and the graph-minus-tabular Wasserstein contrast on a shared feature scale. JS averages 486 fields using 5,000 observations per regime. The mass denominator includes all inputs in the respective model and pools both regimes. AUC uses all 177,162 held-out temporal transactions. The summary gives means and pointwise Student intervals over the 20 seeds (19 degrees of freedom), conditional on the cutoff. All columns can be regenerated from the fitted checkpoints. To reproduce the summary and Table 2 from the public CSVs alone, run:

    python scripts/report_ieee_context.py --directory reference --summary-only --output results/check --table-dir results/check

Code for the two extra normalization contrasts and the top-ten plot is in scripts/plot_strict_ieee_comparisons.py.

To reproduce the exploratory top-three and top-five agreement table, run
`python scripts/strict_ieee_shortlists.py --directory reference --summary-only --output results/check`.
The seed-level file gives the fraction of the tabular model's leading fields retained by each alternative on the same 5,000 payments, among the original 486 fields. The script checks the ten-field result against the earlier deposit. The four shorter-list contrasts use Bonferroni simultaneous 95% intervals over paired seed differences. To recompute the overlaps from the saved SHAP arrays after fitting, omit the flags.

The first-field comparison uses the 486 shared inputs ranked by absolute SHAP value. The reference file display_by_seed.csv contains counts over the 5,000 later payments per fit and the mean absolute attribution mass per payment, pooled over the 5,000 random and 5,000 temporal observations. No transaction IDs are needed for the published summary. Reproduce it with:

    python scripts/strict_ieee_display.py --directory reference --summary-only --output results/check

Running the same script without flags recomputes the counts and masses from the original paired SHAP checkpoints. The first-place tie rule is the canonical input order. Ties are counted for each model and seed.

The stage-separation experiment retains the original forward holdout and splits the remaining payments at their median timestamp. Both blocks contain 206,689 transactions. All downstream trees train on the second block, using identical tabular features fitted there. The graph component is trained either in the first or second block with its own historical preprocessing. AUC and average precision cover all later payments. Local overlap uses 1,000 stratified later payments, with the same background and evaluation IDs across the three fits (the input columns depend on the model). The training design is exploratory and uses one graph initialization. The reference files stage_blocks.csv, stage_results.csv and stage_intervals.csv are sufficient to reproduce the manuscript table:

    python scripts/render_ieee_stage_table.py --directory reference --output results/check/stages.tex

To rerun from the original competition data:

    python scripts/strict_ieee_stages.py
    python scripts/strict_ieee_stage_intervals.py
    python scripts/render_ieee_stage_table.py

The intervals resample the common evaluation period in seven-calendar-day circular blocks, conditional on the fitted models. Each of 1,000 draws uses identical day weights across models. Only aggregate results are distributed.

The 1,000 paired bootstrap draws are also deposited in reference/stage_block_draws.csv. They contain AUC and average precision for each model in each draw, with no payment-level information. Regenerate the intervals from these public data with:

    python scripts/strict_ieee_stage_intervals.py --directory reference --from-draws --output results/stage_intervals_check.csv

The exploratory AUC table uses two families of contrasts. For the three XGBoost refit contrasts, Bonferroni-adjusted paired Student intervals describe training variation on the fixed cohort. For the four fixed-model contrasts, shared seven-day circular bootstrap draws give a simultaneous band using the 95th percentile of the largest absolute centered contrast, standardized by its bootstrap standard deviation. The stored CatBoost contrasts use the same day draws as the stage experiment. Run the following command to reproduce the paired intervals and the manuscript table from the aggregate CSVs:

    python scripts/strict_ieee_auc.py --reference reference --output results/check

Without arguments, the script computes CatBoost's bootstrap contrasts from the saved temporal scores, verifies day-by-day draw alignment against the archived tabular bootstrap AUC, and writes the outputs under results/strict_ieee/stages.

The wide Gaussian control is an exploratory scale check. The earlier width search in reference/legacy_scale_width_search.csv targeted the median ratio of the original fields' mean absolute SHAP contributions (graph relative to tabular). Linear interpolation between the 512- and 768-column probes at the target of 0.627 gave 703 columns after rounding. The current analysis keeps this width fixed. In reference/scale.csv, share is the sum of absolute SHAP contributions for the added coordinates divided by the sum over all coordinates, pooling 5,000 explanations from each validation scheme. The wide control received 44.6% and the graph coordinates 39.5% in that single comparison. For the separate 20-refit analysis, use reference/context_by_seed.csv.

### Successive temporal cutoffs and importance weighting

The exploratory rolling study fits one tabular model and three graph specifications at each of the 60%, 70% and 80% transaction-time quantiles. Each graph specification uses initializations 201–203. The downstream trees use the second half of the available history. All fits explain the same 1,000 later transactions using the same 100 historical background transaction IDs. Reassigned historical edges preserve entity degrees within calendar day and relation type. The 20 original random/temporal pairs are a different experiment.

From this Quantlet directory, with the original competition files in data/raw, run:

    python scripts/chronological_stress.py . results/rolling
    python scripts/rolling_contrasts.py results/rolling data/raw/train_transaction.csv results/rolling/audit
    python scripts/report_chronological.py results/rolling data/raw/train_transaction.csv results/rolling/table.tex
    python scripts/weighted_stability.py results/strict_ieee/primary

The public reference files contain per-fit metrics, numbers of reassigned links, agreement across updates, nine paired AUC contrasts, and the importance-weighted results for the original 20 pairs. The rolling AUC intervals use 12,000 paired resamples of seven-day circular blocks per window and Bonferroni adjustment across all nine contrasts, conditional on the fitted models. Raw competition files and individual predictions are obtained by downloading the original data under the Kaggle competition rules and running the code. To regenerate the printed temporal table directly from the public summaries:

    python -c "import sys,pandas as pd; sys.path.insert(0,'scripts'); from report_chronological import render; print(render(pd.read_csv('reference/rolling_fits.csv')),end='')"
