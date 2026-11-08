"""Summarise the matched Elliptic refits and compare them with the IEEE-CIS refits."""
import argparse
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import t


def original_fields(path, ids, background, width):
    with np.load(path) as raw:
        np.testing.assert_array_equal(raw["ids"], ids)
        np.testing.assert_array_equal(raw["bg_ids"], background)
        values = raw["phi"]
        if values.shape != (len(ids), width) or not np.isfinite(values).all():
            raise ValueError(f"Invalid attribution matrix in {path}")
        return values


def top_fields(values):
    magnitude = np.abs(values)
    order = np.argsort(-magnitude, axis=1, kind="stable")[:, :10]
    first_two = np.take_along_axis(magnitude, order[:, :2], axis=1)
    return order, first_two[:, 0] == first_two[:, 1]


def distribution_metrics(left, right, config, stability):
    divergences = []
    for a, b in zip(left.T, right.T):
        edges = stability.bin_edges(np.r_[a, b], config.N_BINS, config.BIN_TYPE)
        divergences.append(stability.js_divergence(a, b, edges))
    w1 = np.abs(np.sort(left, axis=0) - np.sort(right, axis=0)).mean()
    mae = np.abs(left - right).mean()
    assert np.isfinite(divergences).all() and 0 <= min(divergences) <= max(divergences) <= np.log(2) + 1e-12
    assert 0 <= w1 <= mae + 1e-12
    return {"mean_js": np.mean(divergences), "mean_w1": w1, "mean_absolute_change_same_payment": mae}


def summarize(pairs, seeds):
    rows = []
    disjoint = pairs[(pairs.a % 2 == 1) & (pairs.b == pairs.a + 1)]
    assert len(disjoint) == len(seeds) // 2
    for name in ["graph_change", "gaussian_change", "graph_minus_gaussian", "graph_top3", "gaussian_top3",
                 "graph_top10", "gaussian_top10", "graph_against_tabular", "untied_difference"]:
        value = pairs[name].mean()
        # Delete a matched fit and every pair involving it; transactions remain fixed.
        leave_one_out = np.array([pairs.loc[(pairs.a != s) & (pairs.b != s), name].mean() for s in seeds])
        se = np.sqrt((len(seeds) - 1) * np.mean((leave_one_out - leave_one_out.mean()) ** 2))
        half_width = t.ppf(.975, len(seeds) - 1) * se
        rows.append(dict(metric=name, estimate=value, lower=value-half_width, upper=value+half_width,
                         disjoint_mean=disjoint[name].mean()))
    return pd.DataFrame(rows).set_index("metric")


def audit_design(root, design, raw_dir):
    panels = {}
    with np.load(root / "inputs.npz") as d:
        ids, y, steps = d["ids"], d["y"], d["steps"]
        assert len(np.unique(ids)) == len(y) == design["labeled"]
        assert set(np.unique(y)) == {0, 1} and y.sum() == design["illicit"]
        temporal = d["temporal"]
        np.testing.assert_array_equal(temporal, steps < design["cut"])
        saved_edges = d["edge"]
        raw_edges = pd.read_csv(raw_dir / "elliptic_txs_edgelist.csv", usecols=["txId1","txId2"])
        positions = pd.Index(ids)
        source = positions.get_indexer(raw_edges.txId1)
        destination = positions.get_indexer(raw_edges.txId2)
        known = (source >= 0) & (destination >= 0)
        source, destination = source[known], destination[known]
        historical = temporal[source] & temporal[destination]
        np.testing.assert_array_equal(saved_edges, np.stack([source[historical], destination[historical]]))
        assert int(known.sum()) == design["labeled_edges"]
        assert int((temporal[source] & ~temporal[destination]).sum()) == design["history_to_future"] == 0
        assert saved_edges.shape == (2, design["training_edges"])
        for split, metadata in design["splits"].items():
            train, ev, bg = d[split], d[split + "_ev"], d[split + "_bg"]
            assert len(np.unique(ev)) == metadata["explained"] == 5000
            assert len(np.unique(bg)) == metadata["background"] == 100
            assert train[bg].all() and not train[ev].any() and y[ev].sum() == metadata["explained_illicit"]
            assert (~train).sum() == metadata["test"] and y[~train].sum() == metadata["test_illicit"]
            assert train.sum() == metadata["train"]
            panels[split] = ids[ev], ids[bg]
    return panels


def run(project, root, ieee, out):
    sys.path.insert(0, str(project.resolve()))
    from src import config as C, stability
    sys.path.insert(0, str(ieee.resolve()))
    from graph_refit import agreement

    design = json.loads((root / "design.json").read_text())
    seeds = design["seeds"]
    assert seeds == list(range(201, 221))
    panels = audit_design(root, design, C.RAW_DIR / "elliptic")
    width = design["features"]
    tabular_checks = []
    for split, (ids, bg) in panels.items():
        path = root / f"tab_{split}.npz"
        values = original_fields(path, ids, bg, width)
        with np.load(path) as raw:
            tabular_checks.append(dict(split=split, same_trees=bool(raw["same_trees"]),
                                       same_predictions=bool(raw["same_predictions"]),
                                       max_shap_difference=float(raw["max_delta"]),
                                       leading_changed=int(raw["changed"]), explained=len(ids)))
        if split == "temporal":
            base, _ = top_fields(values)

    ids, bg = panels["temporal"]
    ranks, ties, against = {}, {}, {}
    unique_hash = set()
    distributions = []
    previous = {}
    for seed in seeds:
        graph_hash = {}
        for arm in ("hyb", "cov"):
            path = root / f"{seed}_{arm}.npz"
            with np.load(path) as f:
                assert int(f["seed"]) == seed
                graph_hash[arm] = str(f["graph_hash"])
            values = original_fields(path, ids, bg, width)
            ranks[arm, seed], ties[arm, seed] = top_fields(values)
            if seed % 2:
                previous[arm] = values
            else:
                distributions.append(dict(a=seed-1, b=seed, arm=arm,
                    **distribution_metrics(previous[arm], values, C, stability)))
        assert graph_hash["hyb"] == graph_hash["cov"]
        unique_hash.add(graph_hash["hyb"])
        against[seed] = agreement(ranks["hyb", seed], base)["fraction"]
    assert len(unique_hash) == len(seeds)

    rows = []
    for a, b in combinations(seeds, 2):
        row = {"a": a, "b": b, "n": len(ids), "graph_against_tabular": .5 * (against[a] + against[b])}
        no_ties = np.ones(len(ids), dtype=bool)
        changes = {}
        for arm, label in (("hyb", "graph"), ("cov", "gaussian")):
            ra, rb = ranks[arm, a], ranks[arm, b]
            m = agreement(ra, rb)
            changes[arm] = ra[:, 0] != rb[:, 0]
            assert m["leading_changed"] == changes[arm].sum()
            no_ties &= ~(ties[arm, a] | ties[arm, b])
            row.update({label+"_changed_n": m["leading_changed"], label+"_change": m["fraction"],
                        label+"_top3": m["top3_overlap"], label+"_top10": m["top10_overlap"]})
        row["graph_minus_gaussian"] = row["graph_change"] - row["gaussian_change"]
        row["untied_n"] = no_ties.sum()
        if not no_ties.any():
            raise ValueError("No unambiguous leading fields for sensitivity check")
        row["untied_difference"] = changes["hyb"][no_ties].mean() - changes["cov"][no_ties].mean()
        rows.append(row)
    pairs = pd.DataFrame(rows)
    assert len(pairs) == 190
    # Independent recount: each field selected c times contributes c*(c-1)/2 agreements.
    for arm, label in (("hyb", "graph"), ("cov", "gaussian")):
        counts = np.zeros((len(ids), width), dtype=int)
        for seed in seeds:
            counts[np.arange(len(ids)), ranks[arm, seed][:, 0]] += 1
        agreeing = (counts * (counts - 1) // 2).sum()
        assert pairs[label+"_changed_n"].sum() == len(pairs)*len(ids) - agreeing
    summary = summarize(pairs, seeds)
    reference = pd.read_csv(ieee / "graph_summary.csv").set_index("metric")
    ieee_pairs = pd.read_csv(ieee / "graph_pairs.csv")
    assert list(zip(ieee_pairs.a, ieee_pairs.b)) == list(zip(pairs.a, pairs.b))
    compare = []
    for dataset, frame, names in [
        ("IEEE-CIS", reference, ["fraction","gaussian_fraction","graph_minus_gaussian_reseed"]),
        ("Elliptic", summary, ["graph_change","gaussian_change","graph_minus_gaussian"])]:
        if dataset == "IEEE-CIS":
            for name in names:
                np.testing.assert_allclose(ieee_pairs[name].mean(), frame.loc[name,"estimate"])
        g, c, diff = (frame.loc[n] for n in names)
        np.testing.assert_allclose(g.estimate-c.estimate, diff.estimate)
        compare.append(dict(dataset=dataset, graph=g.estimate, gaussian=c.estimate, difference=diff.estimate,
                            lower=diff.lower, upper=diff.upper, disjoint_difference=diff.disjoint_mean))
    out.mkdir(exist_ok=True, parents=True)
    for name, frame in [("seed_pairs",pairs), ("summary",summary.reset_index()), ("comparison",pd.DataFrame(compare)),
                        ("tabular_checks",pd.DataFrame(tabular_checks)), ("distributions",pd.DataFrame(distributions))]:
        frame.to_csv(out / f"{name}.csv", index=False)
    print(pd.DataFrame(compare).to_string(index=False))
    print(pd.DataFrame(tabular_checks).to_string(index=False))
    print("Minimum common unambiguous transactions per seed pair:", pairs.untied_n.min())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("project", type=Path)
    here = Path(__file__).resolve().parent
    p.add_argument("--checkpoints", type=Path, default=here / "checkpoints")
    p.add_argument("--ieee", type=Path, default=here / "ieee_reference")
    p.add_argument("--out", type=Path, default=here / "results")
    args = p.parse_args()
    run(args.project, args.checkpoints, args.ieee, args.out)
