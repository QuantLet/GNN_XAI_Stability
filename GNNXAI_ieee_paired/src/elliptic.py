"""Elliptic Bitcoin replication (Section 5.8).

Differences from IEEE-CIS:
  * homogeneous, directed transaction-to-transaction graph;
  * 165 anonymized numeric covariates (no entity columns);
  * ~9.8% illicit base rate on the labelled subset (46,564 tx after dropping
    unknown-class rows);
  * native time-step axis (integers 1-49) replaces daily blocks as the block ID;
  * leakage controls adapt: the training graph keeps only edges with both
    endpoints in the training fold.

The XGBoost-only replication runs the headline pair plus the diagonal-scale and
covariance-matched controls.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from . import config as C

ELLIPTIC_CUTOFF_STEP_PCT = 0.70
N_TIME_STEPS = 49


@lru_cache(maxsize=1)
def load_elliptic() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three Elliptic CSVs from data/raw/elliptic/.

    Cached: the 690 MB feature CSV takes ~2.5 min to parse and every caller here
    is read-only (prepare_elliptic copies before mutating; build_graph only reads
    the edge list), so a seed sweep re-reads it once instead of once per fit.
    """
    base = C.RAW_DIR / "elliptic"
    feats = pd.read_csv(base / "elliptic_txs_features.csv", header=None)
    classes = pd.read_csv(base / "elliptic_txs_classes.csv")
    edges = pd.read_csv(base / "elliptic_txs_edgelist.csv")
    feats.columns = ["txId", "time_step"] + [f"f{i}" for i in range(feats.shape[1] - 2)]
    return feats, classes, edges


def prepare_elliptic() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return (features_df_labelled, X, y) dropping unknown-class rows."""
    feats, classes, _ = load_elliptic()
    classes = classes.rename(columns={"class": "label"})
    df = feats.merge(classes, on="txId", how="left")
    df = df[df["label"].isin(["1", "2", 1, 2])].copy()   # drop 'unknown'
    df["y"] = (df["label"].astype(str) == "1").astype(int)  # 1 = illicit
    feat_cols = [c for c in df.columns if c.startswith("f")]
    return df.reset_index(drop=True), df[feat_cols].values.astype(np.float32), df["y"].values


def temporal_mask(df: pd.DataFrame, cutoff_pct: float = ELLIPTIC_CUTOFF_STEP_PCT) -> np.ndarray:
    cutoff = np.quantile(df["time_step"].values, cutoff_pct)
    return df["time_step"].values < cutoff


def block_ids(df: pd.DataFrame) -> np.ndarray:
    """Native time-step axis as the bootstrap block ID."""
    return df["time_step"].values.astype(np.int64)


def build_graph(df: pd.DataFrame, X: np.ndarray, train_mask: np.ndarray) -> Data:
    """Homogeneous directed graph; keep only training-internal edges (leakage rule)."""
    _, _, edges = load_elliptic()
    id_to_idx = {t: i for i, t in enumerate(df["txId"].values)}
    src = edges["txId1"].map(id_to_idx)
    dst = edges["txId2"].map(id_to_idx)
    keep = src.notna() & dst.notna()
    src, dst = src[keep].astype(int).values, dst[keep].astype(int).values
    # rule: edge retained only if both endpoints are training transactions
    both_train = train_mask[src] & train_mask[dst]
    edge_index = torch.tensor(np.stack([src[both_train], dst[both_train]]), dtype=torch.long)
    data = Data(
        x=torch.tensor(X, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(df["y"].values, dtype=torch.long),
    )
    data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
    return data
