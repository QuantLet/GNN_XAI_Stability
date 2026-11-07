"""Reproduce the primary JS comparison from the archived seed-level data."""
from pathlib import Path
import pandas as pd
from scipy import stats
import numpy as np


def main():
    ref = Path(__file__).resolve().parent / "reference"
    paired = pd.read_csv(ref / "paired.csv").set_index("seed")
    if list(paired.index) != list(range(201, 221)):
        raise ValueError("Expected the 20 paired seeds in order")
    if not (paired[["p", "n_shap_random", "n_shap_temporal"]] == [486, 5000, 5000]).all().all():
        raise ValueError("Unexpected feature or evaluation-sample count")
    metric = "js_hyb_minus_cov"
    differences = paired[metric]
    if not np.isfinite(differences).all():
        raise ValueError("Non-finite paired contrast")
    mean = differences.mean()
    half = stats.t.ppf(0.975, len(differences) - 1) * differences.sem()
    archived = pd.read_csv(ref / "intervals.csv").set_index("metric").loc[metric]
    np.testing.assert_allclose([mean, mean - half, mean + half],
                               archived[["mean", "lower", "upper"]].to_numpy(dtype=float))
    print(f"Graph minus Gaussian, mean JS over {len(differences)} refits: "
          f"{mean:+.5f} nats (95% CI {mean-half:+.5f} to {mean+half:+.5f})")


if __name__ == "__main__":
    main()
