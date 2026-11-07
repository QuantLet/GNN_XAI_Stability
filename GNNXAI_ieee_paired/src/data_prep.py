"""IEEE-CIS preprocessing (Section 4.1).

Deterministic pipeline (seed 42) producing a p=486 tabular feature matrix:
  * merge train_transaction + train_identity on TransactionID
  * frequency-encode high-cardinality string categoricals
  * one-hot encode low-cardinality categoricals (drop first level)
  * fold-specific entity-level TransactionAmt summary statistics (no isFraud
    leakage; computed per fold)

All fold-specific statistics are computed on the training mask only and applied
to all rows, so no validation-time information crosses the train/validation
boundary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def load_raw() -> pd.DataFrame:
    """Merge the two IEEE-CIS tables on TransactionID.

    Expects train_transaction.csv and train_identity.csv under data/raw/
    (see scripts/00_prepare_data.py for download instructions).
    """
    tx = pd.read_csv(C.RAW_DIR / "train_transaction.csv")
    idf = pd.read_csv(C.RAW_DIR / "train_identity.csv")
    df = tx.merge(idf, on=C.ID_COL, how="left")
    df = df.sort_values(C.TIME_COL).reset_index(drop=True)
    return df


def _frequency_encode(df: pd.DataFrame, col: str, reference: pd.DataFrame) -> pd.Series:
    """Map each category to its reference-fold frequency (count-based)."""
    counts = reference[col].value_counts(dropna=False)
    return df[col].map(counts).fillna(0).astype(float)


def _amt_group_stats(df: pd.DataFrame, train_mask: np.ndarray, group_col: str,
                     key: pd.Series, stats_wanted: list[str]) -> pd.DataFrame:
    """Fold-specific TransactionAmt summary stats grouped by `key` (a Series).

    `group_col` names the output prefix; `key` is the grouping value per row (may
    be a frequency-encoded column). Stats computed on training rows only, mapped
    to every row. `sm_mean` is the empirical-Bayes shrunk mean.
    """
    tmp = pd.DataFrame({"k": key.values, "amt": df["TransactionAmt"].values}, index=df.index)
    train = tmp.loc[train_mask]
    grp = train.groupby("k")["amt"]
    agg = grp.agg(["mean", "std", "min", "max", "count"])
    global_mean = train["amt"].mean()
    n = agg["count"]
    agg["sm_mean"] = (agg["mean"] * n + global_mean * C.AMT_SMOOTHING) / (n + C.AMT_SMOOTHING)
    out = {}
    for stat in stats_wanted:
        out[f"{group_col}_TransactionAmt_{stat}"] = tmp["k"].map(agg[stat])
    return pd.DataFrame(out, index=df.index).fillna(0.0)


def _onehot(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """One-hot encode a low-cardinality categorical, with a MISSING level for NaN."""
    s = df[col].astype("object").where(df[col].notna(), "MISSING").astype(str)
    return pd.get_dummies(s, prefix=col).astype(float)


def build_features(df: pd.DataFrame, train_mask: np.ndarray, *, historical_frequencies: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """Return the (X, entity_feature_names) tabular matrix for a given fold mask,
    reindexed to the canonical 486-column list recovered from the deposit.

    Amount summaries always use training rows. Historical runs used whole-table
    frequencies and category support. The strict rerun fits both on training rows.
    """
    feats: dict[str, pd.Series] = {}

    # 1. frequency-encoded high-cardinality categoricals
    freq_vals: dict[str, pd.Series] = {}
    reference = df if historical_frequencies else df.loc[train_mask, df.columns.intersection(C.FREQ_ENCODE_COLS)]
    for col in C.FREQ_ENCODE_COLS:
        if col in df.columns:
            fv = _frequency_encode(df, col, reference)
            freq_vals[col] = fv
            feats[f"{col}_freq"] = fv

    # 2. one-hot low-cardinality categoricals (MISSING level for NaN)
    for col in C.ONEHOT_COLS:
        if col in df.columns:
            onehot = _onehot(df, col)
            if not historical_frequencies:
                onehot = onehot.loc[:, onehot.loc[train_mask].any(axis=0)]
            for c, ser in onehot.items():
                feats[c] = ser

    # 3. fold-specific TransactionAmt summary statistics
    for group_col, stats_wanted in C.AMT_GROUPS.items():
        base = group_col[:-5] if group_col.endswith("_freq") else group_col
        if group_col.endswith("_freq"):
            key = freq_vals.get(base)
        else:
            key = df[base] if base in df.columns else None
        if key is not None:
            stats = _amt_group_stats(df, train_mask, group_col, key, stats_wanted)
            for c in stats.columns:
                feats[c] = stats[c]

    # 4. remaining numeric columns (V/C/D, numeric id_*, raw cards/addr/dist/amount)
    encoded = set(C.FREQ_ENCODE_COLS) | set(C.ONEHOT_COLS)
    skip = encoded | {C.TARGET, C.ID_COL, C.TIME_COL} | set(feats)
    for col in df.columns:
        if col in skip:
            continue
        ser = pd.to_numeric(df[col], errors="coerce")
        if (ser if historical_frequencies else ser.loc[train_mask]).notna().any():
            feats[col] = ser.fillna(-999).astype(float)

    X = pd.DataFrame(feats, index=df.index)

    # reindex to the canonical 486-feature list: drops extras, zero-fills any
    # absent one-hot level so every split has identical columns in identical order
    canonical = _canonical_columns()
    if canonical is not None:
        X = X.reindex(columns=canonical, fill_value=0.0)
    X = X.fillna(0.0)

    entity_feats = [f for f in X.columns if is_entity_feature(f)]
    return X, entity_feats


_CANON_CACHE: list[str] | None = None


def _canonical_columns() -> list[str] | None:
    global _CANON_CACHE
    if _CANON_CACHE is None and C.FEATURE_LIST_PATH.exists():
        import json

        _CANON_CACHE = json.loads(C.FEATURE_LIST_PATH.read_text())
    return _CANON_CACHE


def is_entity_feature(name: str) -> bool:
    """Name-pattern rule for the entity-related class (Section 5.4, Appendix B)."""
    patterns = ("_freq", "DeviceType_", "TransactionAmt_")
    raw = {"card1", "card2", "card3", "card5", "addr2"}   # Appendix B group 3
    return name in raw or any(p in name for p in patterns)
