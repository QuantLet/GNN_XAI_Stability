"""Recover JS levels for the primary comparison and context for the paired estimates."""
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config, stability
from scripts.strict_ieee_normalized import write_scale_table


def read_seeds(path):
    table = pd.read_csv(path).set_index("seed")
    if table.index.tolist() != list(range(201, 221)):
        raise ValueError(f"Expected the 20 planned seed pairs in {path}")
    return table


def from_checkpoints(directory):
    paired = read_seeds(directory / "paired.csv")
    scales = read_seeds(directory / "plot_comparisons_by_seed.csv")
    rows = []
    for seed, archived in paired.iterrows():
        with np.load(directory / f"{seed}_random.npz") as r, np.load(directory / f"{seed}_temporal.npz") as t:
            p = int(r["p"])
            cov_r, cov_t, graph_r, graph_t = r["cov"], t["cov"], r["hyb"], t["hyb"]
            samples = (cov_r, cov_t, graph_r, graph_t)
            if p != 486 or int(t["p"]) != p or any(a.shape != (5000, p + 64) for a in samples):
                raise ValueError(f"Unexpected SHAP denominator at seed {seed}")
            if not all(np.isfinite(a).all() for a in samples):
                raise ValueError(f"Non-finite SHAP values at seed {seed}")
            levels = np.empty((p, 2))
            for k in range(p):
                edges = stability.bin_edges(np.concatenate([a[:, k] for a in samples]),
                                            config.N_BINS, config.BIN_TYPE)
                levels[k, 0] = stability.js_divergence(graph_r[:, k], graph_t[:, k], edges)
                levels[k, 1] = stability.js_divergence(cov_r[:, k], cov_t[:, k], edges)
            js_graph, js_gaussian = levels.mean(axis=0)
            if (levels < -1e-12).any() or (levels > np.log(2) + 1e-12).any():
                raise ValueError("Invalid JS divergence in nats")
            np.testing.assert_allclose(js_graph - js_gaussian, archived.js_hyb_minus_cov, atol=1e-12)
            row = dict(seed=seed, js_graph=js_graph, js_gaussian=js_gaussian,
                       js_difference=js_graph - js_gaussian)
            for arm, random, temporal in [("cov", cov_r, cov_t), ("hyb", graph_r, graph_t)]:
                pooled = np.abs(np.concatenate([random, temporal]))
                row[f"share_{arm}"] = pooled[:, p:].sum() / pooled.sum()
                np.testing.assert_allclose(row[f"share_{arm}"], archived[f"absorbed_{arm}"], atol=1e-8)
                row[f"auc_{arm}"] = float(t[f"auc_{arm}"])
            if int(t["test_n"]) != 177162:
                raise ValueError("Unexpected temporal AUC denominator")
            row["auc_tab"] = float(t["auc_tab"])
            tab_r, tab_t = r["tab"], t["tab"]
            if tab_r.shape != (5000, p) or tab_t.shape != tab_r.shape:
                raise ValueError("Unexpected tabular SHAP shape")
            gaps, masses = [], []
            for random, temporal in [(graph_r[:, :p], graph_t[:, :p]), (tab_r, tab_t)]:
                gaps.append(np.abs(np.sort(random, axis=0) - np.sort(temporal, axis=0)).mean(axis=0))
                masses.append((np.abs(random).mean(axis=0) + np.abs(temporal).mean(axis=0)) / 2)
            common = (masses[0] + masses[1]) / 2
            row["common_scale"] = np.divide(gaps[0] - gaps[1], common,
                                             out=np.zeros(p), where=common > 0).mean()
            np.testing.assert_allclose(row["common_scale"], scales.loc[seed, "common_scale"], atol=1e-10)
            rows.append(row)
    return pd.DataFrame(rows).set_index("seed")


def summarize(table):
    if not np.isfinite(table.to_numpy()).all():
        raise ValueError("Incomplete seed-level results")
    for col in ["share_cov", "share_hyb", "auc_cov", "auc_hyb", "auc_tab"]:
        if not table[col].between(0, 1).all():
            raise ValueError(f"Invalid proportion in {col}")
    mean = table.mean()
    half = stats.t.ppf(.975, len(table) - 1) * table.sem()
    return pd.DataFrame(dict(n=len(table), mean=mean, lower=mean-half, upper=mean+half))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "results/strict_ieee/primary")
    parser.add_argument("--output", type=Path, help="Output directory (defaults to the input directory)")
    parser.add_argument("--summary-only", action="store_true", help="Use deposited seed-level CSV instead of checkpoints")
    parser.add_argument("--table-dir", type=Path, default=ROOT / "submission/source/sections")
    args = parser.parse_args()
    out = args.output or args.directory
    out.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        table = read_seeds(args.directory / "context_by_seed.csv")
    else:
        table = from_checkpoints(args.directory)
        table.to_csv(out / "context_by_seed.csv")
    result = summarize(table)
    result.to_csv(out / "context_summary.csv", index_label="metric")
    args.table_dir.mkdir(parents=True, exist_ok=True)
    write_scale_table(read_seeds(args.directory / "wasserstein_by_seed.csv"), args.table_dir, table.common_scale)
    print(result.to_string())
