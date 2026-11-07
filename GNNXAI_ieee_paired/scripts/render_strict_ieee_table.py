"""Generate the main results table from the completed IEEE-CIS run."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/strict_ieee/primary"
ROWS = [
    ("js_hyb_minus_cov", "JS, graph minus covariance control", 5),
    ("js_hyb_minus_tab", "JS, graph minus tabular", 5),
    ("wasserstein_hyb_minus_tab", "Raw Wasserstein, graph minus tabular", 5),
    ("top10_random", "Top-ten agreement difference, random", 4),
    ("top10_temporal", "Top-ten agreement difference, temporal", 4),
    ("spearman_random", "Spearman agreement difference, random", 4),
    ("spearman_temporal", "Spearman agreement difference, temporal", 4),
]


def main():
    with (RESULTS / "paired.csv").open() as source:
        seeds = [int(row["seed"]) for row in csv.DictReader(source)]
    if seeds != list(range(201, 221)):
        raise ValueError("The manuscript requires all 20 planned seeds")
    with (RESULTS / "intervals.csv").open() as source:
        estimates = {row["metric"]: row for row in csv.DictReader(source)}
    lines = [r"\begin{tabular}{lrr}", r"\toprule",
             r"Comparison & Mean & 95\% interval\\", r"\midrule"]
    for metric, label, digits in ROWS:
        row = estimates[metric]
        mean, lo, hi = (float(row[k]) for k in ("mean", "lower", "upper"))
        if int(row["n"]) != len(seeds) or not lo <= mean <= hi:
            raise ValueError(f"Invalid interval for {metric}")
        cells = (f"{mean:+.{digits}f}", f"[{lo:+.{digits}f},{hi:+.{digits}f}]")
        lines.append(label + " & " + " & ".join("$" + v + "$" for v in cells) + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    target = ROOT / "submission/source/sections/strict_ieee_rows.tex"
    target.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
