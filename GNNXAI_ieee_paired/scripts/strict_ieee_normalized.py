"""Post-hoc Wasserstein scale check on saved IEEE-CIS attributions, without refitting."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]


def distances(random, temporal):
    # Equal empirical weights and equal sample sizes give an exact sorted-sample W1.
    if random.shape != temporal.shape or random.shape != (5000, 486):
        raise ValueError("Expected two 5,000 by 486 matrices of shared SHAP values")
    if not np.isfinite(random).all() or not np.isfinite(temporal).all():
        raise ValueError("Non-finite SHAP attribution")
    gap = np.abs(np.sort(random, axis=0) - np.sort(temporal, axis=0)).mean(axis=0)
    mass = (np.abs(random).mean(axis=0) + np.abs(temporal).mean(axis=0)) / 2
    normalized = np.divide(gap, mass, out=np.zeros_like(gap), where=mass > 0)
    return gap.mean(), normalized.mean()


def write_scale_table(data, table_dir, common=None):
    lines = [r"\begin{tabular}{lrrr}", r"\toprule", r"Scale and comparator & Mean difference & 95\% interval & Positive runs\\", r"\midrule"]
    for scale in ("raw", "normalized"):
        for arm, label in (("tab", "tabular"), ("cov", "Gaussian")):
            delta = data[f"{scale}_hyb"] - data[f"{scale}_{arm}"]
            mean = delta.mean()
            half = stats.t.ppf(.975, len(delta)-1) * delta.sem()
            digits = 5 if scale == "raw" else 4
            line = (f"{scale.capitalize()}, {label} & {mean:+.{digits}f} & "
                    f"$[{mean-half:+.{digits}f},{mean+half:+.{digits}f}]$ & "
                    f"{(delta > 0).sum()}/{len(delta)}" + r"\\")
            lines.append(line)
            print(line, 'negative runs', (delta < 0).sum())
    if common is not None:
        if not common.index.equals(data.index):
            raise ValueError("Mismatched common-scale seeds")
        mean = common.mean()
        half = stats.t.ppf(.975, len(common) - 1) * common.sem()
        lines.append(f"Common scale, tabular & {mean:+.4f} & "
                     f"$[{mean-half:+.4f},{mean+half:+.4f}]$ & "
                     f"{(common > 0).sum()}/{len(common)}" + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (table_dir / "strict_ieee_scale.tex").write_text("\n".join(lines) + "\n")


def run(directory, table_dir):
    paired = pd.read_csv(directory / "paired.csv").set_index("seed")
    if paired.index.tolist() != list(range(201, 221)):
        raise ValueError("The analysis requires all 20 planned pairs")
    rows = []
    for seed, original in paired.iterrows():
        row = {"seed": seed}
        with np.load(directory / f"{seed}_random.npz") as r, np.load(directory / f"{seed}_temporal.npz") as t:
            for arm in ("tab", "hyb", "cov"):
                raw, norm = distances(r[arm][:, :486], t[arm][:, :486])
                row[f"raw_{arm}"] = raw
                row[f"normalized_{arm}"] = norm
        for arm in ("hyb", "cov"):
            np.testing.assert_allclose(row[f"raw_{arm}"] - row["raw_tab"],
                                       original[f"wasserstein_{arm}_minus_tab"], atol=1e-10)
        rows.append(row)
    scales = pd.read_csv(directory / "scale.csv").set_index("arm")
    with np.load(directory / "scale_random.npz") as r, np.load(directory / "scale_temporal.npz") as t:
        for arm, reference in scales.iterrows():
            np.testing.assert_allclose(distances(r[arm][:, :486], t[arm][:, :486]),
                                       reference[["wasserstein", "normalized"]], atol=1e-10)
    data = pd.DataFrame(rows).set_index("seed")
    data.to_csv(directory / "wasserstein_by_seed.csv")
    write_scale_table(data, table_dir)
    figure_dir = table_dir.parent / "figures"
    figure_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(6.25, 3.8), sharex=True, layout="constrained")
    for ax, scale, label in zip(axes, ("raw", "normalized"),
                                ("Raw difference (SHAP units)", "Normalized difference")):
        difference = data[f"{scale}_hyb"] - data[f"{scale}_tab"]
        ax.axhline(0, color="0.45", lw=0.8)
        ax.scatter(data.index, difference, color="#204b65", s=18)
        ax.set_title(label, loc="left", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.margins(y=0.23)
    axes[-1].set_xticks([201, 205, 210, 215, 220])
    axes[-1].set_xlabel("Training seed", fontsize=9)
    fig.savefig(figure_dir / "wasserstein_by_seed.pdf")
    plt.close(fig)


if __name__ == "__main__":
    run(ROOT / "results/strict_ieee/primary", ROOT / "submission/source/sections")
