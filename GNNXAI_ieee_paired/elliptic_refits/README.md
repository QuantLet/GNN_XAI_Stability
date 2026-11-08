# Refitting a fraud model on Elliptic

Table 7 compares how often the original field with the largest absolute SHAP contribution changes for the same later transaction between two fits. Each hybrid fit uses a different GraphSAGE initialization. The matched Gaussian control redraws its coordinates using the training covariance of that fit. The tree seed and the IDs of the training, background and evaluation transactions stay fixed.

[Comparison across datasets](reference/comparison.csv) · [Elliptic seed-pair counts](reference/seed_pairs.csv) · [Run manifest and input hashes](reference/design.json)

## Reproduce from the original data

Run the commands below from the parent directory of this README's folder (`GNNXAI_ieee_paired`) in a fresh clone. Use Python 3.13 on macOS or Linux with the versions in `requirements.txt`. Download the [original Elliptic files](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) and place `elliptic_txs_features.csv`, `elliptic_txs_classes.csv` and `elliptic_txs_edgelist.csv` in `data/raw/elliptic/`.

```sh
python -m pip install -r requirements.txt
python elliptic_refits/check_time_steps.py data/raw/elliptic
python elliptic_refits/run.py . elliptic_refits/checkpoints --workers 4
python elliptic_refits/report.py .
```

The time-step check includes unlabelled transactions. The model fits and explanations use labelled transactions only. The fit script writes one atomic checkpoint per model and resumes completed fits when rerun with the same inputs. It fits both tabular seed checks and twenty hybrid/control pairs. The reporting script checks panel IDs, background IDs and training edges before computing ranks and intervals. Compare its output in `elliptic_refits/results/` with `elliptic_refits/reference/`. The `ieee_reference/` folder contains the IEEE-CIS seed-pair inputs and the shared ranking-agreement routine used for the cross-dataset comparison.

Each Elliptic pair compares the same 5,000 later transactions. The difference in Table 7 averages the hybrid-minus-Gaussian change rate over 190 unordered seed pairs. The interval removes one of the twenty matched seeds and all its pairs in turn. It describes variation across initializations for this fixed panel.
