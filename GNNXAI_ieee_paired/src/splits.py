"""Train/validation splits under the two validation regimes (Section 4.2).

random  : 30% stratified holdout on isFraud, seed 42
temporal: partition on the 70th percentile of TransactionDT (forward-time)

Both regimes receive the same leakage treatment (Section 3.2); the only
difference is the choice of training mask.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C


def random_split_mask(df: pd.DataFrame, seed: int = C.SEED) -> np.ndarray:
    """Return a boolean train mask for the stratified random split."""
    idx = np.arange(len(df))
    train_idx, _ = train_test_split(
        idx,
        test_size=C.RANDOM_HOLDOUT_FRAC,
        stratify=df[C.TARGET].values,
        random_state=seed,
    )
    mask = np.zeros(len(df), dtype=bool)
    mask[train_idx] = True
    return mask


def temporal_split_mask(df: pd.DataFrame, cutoff_pct: float = C.TEMPORAL_CUTOFF_PCT) -> np.ndarray:
    """Return a boolean train mask for the forward-time split.

    Transactions with TransactionDT below the cutoff percentile train; those
    above validate. cutoff_pct=0.70 reproduces the headline; 0.60/0.80 are the
    robustness checks of Section 5.7.
    """
    t = df[C.TIME_COL].values
    cutoff = np.quantile(t, cutoff_pct)
    return t < cutoff


def get_split_mask(df: pd.DataFrame, split: str, seed: int = C.SEED,
                   cutoff_pct: float = C.TEMPORAL_CUTOFF_PCT) -> np.ndarray:
    if split == "random":
        return random_split_mask(df, seed=seed)
    if split == "temporal":
        return temporal_split_mask(df, cutoff_pct=cutoff_pct)
    raise ValueError(f"unknown split {split!r}")


def block_ids(df: pd.DataFrame) -> np.ndarray:
    """Daily block IDs for the block bootstrap: floor(TransactionDT / 86400)."""
    return (df[C.TIME_COL].values // C.BLOCK_SECONDS).astype(np.int64)
