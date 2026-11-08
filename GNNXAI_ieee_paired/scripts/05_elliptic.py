"""Step 5 -- Elliptic replication (Section 5.8) -> T8_elliptic_stability.

XGBoost-only replication of the headline pair plus the diagonal-scale and
covariance-matched controls, on the homogeneous Elliptic transaction graph with
the native time-step block axis.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# `src` must be imported BEFORE torch: on macOS its __init__ pre-loads xgboost to
# keep a single OpenMP runtime in the process (see src/__init__.py).
from src import config as C
from src import elliptic, gnn, embeddings, models, shap_engine, stability as S

import torch
from torch_geometric.nn import SAGEConv
from torch import nn
import torch.nn.functional as F


class HomoSAGE(nn.Module):
    """Two-layer homogeneous GraphSAGE for the Elliptic transaction graph."""

    def __init__(self, in_dim, hidden=C.EMBED_DIM):
        super().__init__()
        self.c1 = SAGEConv(in_dim, hidden, aggr="mean")
        self.c2 = SAGEConv(hidden, hidden, aggr="mean")
        self.head = nn.Linear(hidden, 2)

    def encode(self, x, ei):
        z1 = F.relu(self.c1(x, ei))
        z2 = F.relu(self.c2(z1, ei))
        return z1, z2

    def forward(self, x, ei):
        z1, z2 = self.encode(x, ei)
        return self.head(z2), z1, z2


def train_extract(data, device="cpu", seed=C.SEED):
    # `seed` controls the GraphSAGE weight init + training RNG only; the default
    # reproduces the headline fit. The graph-seed sweep varies it (see
    # scripts/elliptic_catb_seedcheck.py).
    C.seed_everything(seed)
    m = HomoSAGE(data.x.size(1)).to(device)
    data = data.to(device)
    opt = torch.optim.Adam(m.parameters(), lr=C.SAGE_PARAMS["lr"],
                           weight_decay=C.SAGE_PARAMS["weight_decay"])
    y, tm = data.y, data.train_mask
    best, wait, state = 1e9, 0, None
    for _ in range(C.SAGE_PARAMS["max_epochs"]):
        m.train(); opt.zero_grad()
        logits, _, _ = m(data.x, data.edge_index)
        loss = F.cross_entropy(logits[tm], y[tm])
        loss.backward(); opt.step()
        if float(loss) < best - 1e-5:
            best, wait, state = float(loss), 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            wait += 1
            if wait >= C.SAGE_PARAMS["patience"]:
                break
    if state:
        m.load_state_dict(state)
    m.eval()
    with torch.no_grad():
        z1, z2 = m.encode(data.x, data.edge_index)
    return z1.cpu().numpy().astype(np.float32), z2.cpu().numpy().astype(np.float32)


def shap_for(emb_kind, df, X, z1, z2, train_mask, seed=C.SEED, learner="xgb"):
    emb = embeddings.embedding_for_variant(emb_kind, z1, z2, seed=seed)
    Xin = models.build_input(X, emb)
    y = df["y"].values
    tr, val = train_mask, ~train_mask
    model = models.fit_learner(learner, Xin[tr], y[tr], Xin[val], y[val])
    sv, idx = shap_engine.compute_shap(model, Xin[tr], y[tr], Xin[val], y[val])
    # tabular columns only (for cross-config comparison) + eval-row indices into val
    return sv[:, : X.shape[1]], idx


def main(device: str = "cpu") -> int:
    df, X, y = elliptic.prepare_elliptic()
    print(f"Elliptic labelled: {len(df)} tx, illicit rate {y.mean():.4f}")

    time_step = df["time_step"].values
    results, blocks = {}, {}
    for split in ("random", "temporal"):
        if split == "random":
            rng = np.random.default_rng(C.SEED)
            mask = np.zeros(len(df), bool)
            mask[rng.choice(len(df), int(0.7 * len(df)), replace=False)] = True
        else:
            mask = elliptic.temporal_mask(df)
        data = elliptic.build_graph(df, X, mask)
        z1, z2 = train_extract(data, device=device)
        val = ~mask
        # XGBoost arm (tab, l1, random, covmatched) + CatBoost arm (tab, l1, covmatched)
        specs = [("tab", None, "xgb"), ("l1", "l1", "xgb"),
                 ("random", "random", "xgb"), ("covmatched", "covmatched", "xgb"),
                 ("catb_tab", None, "catb"), ("catb_l1", "l1", "catb"),
                 ("catb_covmatched", "covmatched", "catb")]
        out = {k: shap_for(kind, df, X, z1, z2, mask, learner=lrn) for k, kind, lrn in specs}
        results[split] = {k: v[0] for k, v in out.items()}
        # native time-step axis as the block id, aligned to the SHAP eval rows
        eval_idx = out["tab"][1]
        blocks[split] = time_step[val][eval_idx]

    rows = []
    pairs = [("tab", "l1", "headline"),
             ("random", "l1", "t2b_random"),
             ("covmatched", "l1", "t2b_covmatched"),
             # CatBoost arm on Elliptic (closes the "most informative follow-up")
             ("catb_tab", "catb_l1", "catb_headline"),
             ("catb_covmatched", "catb_l1", "catb_t2b_covmatched")]
    for a, b, lbl in pairs:
        n = min(results["random"][a].shape[0], results["temporal"][a].shape[0])
        res = S.diagnostic(
            results["random"][a][:n], results["temporal"][a][:n],
            results["random"][b][:n], results["temporal"][b][:n],
            blocks["random"][:n], blocks["temporal"][:n],
            n_bins=C.N_BINS, bin_type=C.BIN_TYPE, n_boot=300, n_perm=1000,
        )
        s = S.summarize(res); s["pair"] = lbl
        rows.append(s)
        print(f"  {lbl:22s} flagged={s['frac_flagged']*100:5.1f}%  delta={s['mean_delta']:+.4f}", flush=True)
    pd.DataFrame(rows).to_csv(C.TABLES_DIR / "T8_elliptic_stability_singleset.csv", index=False)
    print("Wrote T8_elliptic_stability_singleset.csv to results/tables/.")
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    raise SystemExit(main(**vars(ap.parse_args())))
