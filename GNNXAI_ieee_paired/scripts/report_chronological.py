"""Recompute the per-period manuscript table from the completed temporal experiment."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ARMS = ("same", "separate", "separate_rewired")
LABELS = ("Tabular", "Same period", "Earlier period", "Earlier period, reassigned links")


def top_three(phi):
    return np.argsort(-np.abs(phi[:, :486]), axis=1, kind="stable")[:, :3]


def compute(results, raw):
    design = json.loads((results / "design.json").read_text())
    cutoffs, seeds = design["cutoffs"], design["seeds"]
    if cutoffs != [60, 70, 80] or seeds != [201, 202, 203]:
        raise ValueError("Unexpected design")
    required = {f"c{c}_tab" for c in cutoffs}
    required.update(f"c{c}_{s}_{a}" for c in cutoffs for s in seeds for a in ARMS)
    found = {f.stem for f in results.glob("c*.npz") if not f.name.endswith(".partial.npz")}
    if found != required:
        raise ValueError(f"{len(found & required)}/{len(required)} planned fits available")
    fields = ["TransactionID", "TransactionDT", "isFraud"]
    transactions = pd.read_csv(raw, usecols=fields).sort_values(["TransactionDT", "TransactionID"])
    if not transactions.TransactionID.is_unique:
        raise ValueError("Duplicate transaction IDs")
    times = transactions.TransactionDT
    by_id = transactions.set_index("TransactionID")
    for key, size in (("panel_ids", 1000), ("bg_ids", 100)):
        if len(design[key]) != size or len(set(design[key])) != size:
            raise ValueError(f"Unexpected size for {key}")
    if (by_id.loc[design["panel_ids"], "TransactionDT"] < times.quantile(.8)).any():
        raise ValueError("Explanation panel includes an earlier transaction")
    if (by_id.loc[design["bg_ids"], "TransactionDT"] >= times.quantile(.2)).any():
        raise ValueError("Background includes a later transaction")
    records = []
    for cutoff in cutoffs:
        wave = transactions[(times >= times.quantile(cutoff / 100)) &
                            (times < times.quantile((cutoff + 10) / 100))]
        with np.load(results / f"c{cutoff}_tab.npz") as tab:
            reference = top_three(tab["phi"])
        for arm in ("tab", *ARMS):
            for seed in ([None] if arm == "tab" else seeds):
                name = f"c{cutoff}_tab" if seed is None else f"c{cutoff}_{seed}_{arm}"
                with np.load(results / f"{name}.npz") as fit:
                    np.testing.assert_array_equal(np.sort(fit["wave_ids"]), np.sort(wave.TransactionID))
                    np.testing.assert_array_equal(fit["y"], by_id.loc[fit["wave_ids"], "isFraud"])
                    for key in ("bg_ids", "panel_ids"):
                        np.testing.assert_array_equal(fit[key], design[key])
                    rankings = top_three(fit["phi"])
                    count = (reference[:, :, None] == rankings[:, None, :]).any(axis=2).sum(axis=1)
                    records.append(dict(cutoff=cutoff, arm=arm, seed=seed, model=name,
                                        n_wave=len(wave), n_fraud=int(fit["y"].sum()),
                                        auc=roc_auc_score(fit["y"], fit["scores"]),
                                        ap=average_precision_score(fit["y"], fit["scores"]),
                                        top3_vs_tab=count.mean()))
    frame = pd.DataFrame(records)
    metrics = ["n_wave", "n_fraud", "auc", "ap", "top3_vs_tab"]
    original = pd.read_csv(results / "descriptive.csv").set_index("model")
    np.testing.assert_allclose(frame[metrics], original.loc[frame.model, metrics], atol=1e-10, rtol=0)
    return frame


def render(frame):
    lines = [r"\begin{tabular}{llrrr}", r"\toprule",
             r"Cutoff & Model & AUC & AP & Top three retained \\", r"\midrule"]
    for cutoff, group in frame.groupby("cutoff"):
        for i, (arm, label) in enumerate(zip(("tab", *ARMS), LABELS)):
            row = group[group.arm == arm].mean(numeric_only=True)
            text = f"{cutoff}\\%" if i == 0 else ""
            lines.append(f"{text} & {label} & {row.auc:.4f} & {row.ap:.4f} & {row.top3_vs_tab:.2f}" + r" \\")
        lines.append(r"\addlinespace")
    return "\n".join(lines[:-1] + [r"\bottomrule", r"\end{tabular}", ""])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("transaction_csv", type=Path)
    parser.add_argument("tex_output", type=Path)
    args = parser.parse_args()
    table = compute(args.results, args.transaction_csv)
    args.tex_output.write_text(render(table))
