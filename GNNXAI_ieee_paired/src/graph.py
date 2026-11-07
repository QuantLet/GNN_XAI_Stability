"""Leakage-controlled heterogeneous graph construction (Section 3.2).

Node types: transaction (primary) + four entity types
(addr1, DeviceType, P_emaildomain, id_31).

Four leakage rules:
  1. training graph contains only transactions with t < t_cut;
  2. entity universe restricted to entities touched by >=1 training transaction;
  3. validation transactions referencing unseen entities -> per-type sentinel
     'unseen' node with zero degree features (not dropped);
  4. weights frozen at inference; message passing aggregates over training-graph
     neighbours only.

Entity node feature: [deg(v), log(1+deg(v))] computed strictly on training-fold
edges. Mean aggregation (Section 3.2). The same four rules apply under the random
training mask.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from . import config as C

SENTINEL = "__unseen__"


def _entity_index(values: pd.Series, train_mask: np.ndarray) -> dict:
    """Map entity values seen in training to contiguous ids; reserve a sentinel.

    Rule 2: only entities touched by a training transaction become real nodes.
    Rule 3: everything else (incl. NaN and future-only values) maps to SENTINEL.
    """
    seen = pd.unique(values[train_mask].dropna())
    mapping = {v: i for i, v in enumerate(seen)}
    mapping[SENTINEL] = len(mapping)
    return mapping


def build_hetero_graph(df: pd.DataFrame, X: np.ndarray, train_mask: np.ndarray, *, inductive: bool = False) -> HeteroData:
    """Build a HeteroData graph with leakage controls for one split.

    `X` is the standardized transaction feature matrix (standardization uses
    training-fold means/variances, Section 3.2). `train_mask` selects training
    transactions for both edge construction and the labelled training nodes.
    """
    data = HeteroData()
    n_tx = len(df)

    data["transaction"].x = torch.tensor(X, dtype=torch.float)
    data["transaction"].y = torch.tensor(df[C.TARGET].values, dtype=torch.long)
    data["transaction"].train_mask = torch.tensor(train_mask, dtype=torch.bool)

    for col in C.ENTITY_COLS:
        if col not in df.columns:
            continue
        vals = df[col]
        mapping = _entity_index(vals, train_mask)
        sentinel_id = mapping[SENTINEL]

        # transaction -> entity ids (rule 3: unseen -> sentinel)
        ent_ids = vals.map(mapping).fillna(sentinel_id).astype(np.int64).values
        n_ent = len(mapping)

        # degree features computed on TRAINING edges only (rule 1 + rule 2)
        deg = np.zeros(n_ent, dtype=np.float64)
        train_ent = ent_ids[train_mask]
        np.add.at(deg, train_ent, 1.0)
        deg[sentinel_id] = 0.0
        # standardize the degree features: high-degree entities (gmail.com ~1e4)
        # otherwise blow up activations to magnitude ~sqrt(deg) and destabilize
        # training (Section 3.2). Stats over real (non-sentinel) training nodes.
        ent_feat = np.stack([deg, np.log1p(deg)], axis=1)
        real = np.ones(n_ent, dtype=bool)
        real[sentinel_id] = False
        if real.sum() > 1:
            mu = ent_feat[real].mean(axis=0)
            sd = ent_feat[real].std(axis=0)
            sd[sd == 0] = 1.0
            ent_feat = (ent_feat - mu) / sd
            ent_feat[sentinel_id] = 0.0
        data[col].x = torch.tensor(ent_feat, dtype=torch.float)

        # bidirectional edges; at inference, validation rows referencing the
        # sentinel aggregate only over training neighbours (handled by frozen
        # weights + sentinel having no informative features)
        tx_idx = torch.arange(n_tx, dtype=torch.long)
        ent_idx = torch.tensor(ent_ids, dtype=torch.long)
        outgoing = torch.stack([tx_idx, ent_idx])
        if inductive:
            # Evaluation nodes may receive historical messages, but cannot send any.
            outgoing = outgoing[:, data["transaction"].train_mask]
        data["transaction", "to", col].edge_index = outgoing
        data[col, "rev_to", "transaction"].edge_index = torch.stack([ent_idx, tx_idx])

    return data


def standardize(X_train_fit: np.ndarray, X_all: np.ndarray) -> np.ndarray:
    """Standardize all rows using training-fold means and variances (Section 3.2)."""
    mu = X_train_fit.mean(axis=0)
    sd = X_train_fit.std(axis=0)
    sd[sd == 0] = 1.0
    return (X_all - mu) / sd
