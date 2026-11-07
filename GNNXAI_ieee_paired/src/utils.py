"""Shared utilities expected by the original pipeline driver (tools/run_pipeline.py).

Provides get_logger, load_config, and seed_all. load_config returns a plain dict
overlaid from config.yaml (if present) on top of the canonical constants in
src/config.py, so both the file-driven original API and the module constants stay
in sync.
"""
from __future__ import annotations

import logging

from . import config as C

_CONFIG_CACHE: dict | None = None


def get_logger(name: str = "pipeline") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s",
                                         datefmt="%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def seed_all(seed: int = C.SEED) -> None:
    C.seed_everything(seed)


def read_preds(path):
    """Read a predictions parquet as (y_true, y_proba), tolerating both the
    deposit schema (y_true/y_proba) and the legacy (y/proba) schema."""
    import numpy as np
    import pandas as pd

    df = pd.read_parquet(path)
    y = df["y_true"] if "y_true" in df.columns else df["y"]
    p = df["y_proba"] if "y_proba" in df.columns else df["proba"]
    return np.asarray(y), np.asarray(p)


def _defaults() -> dict:
    return {
        "seed": C.SEED,
        "temporal_cutoff_pct": C.TEMPORAL_CUTOFF_PCT,
        "random_holdout_frac": C.RANDOM_HOLDOUT_FRAC,
        "shap": {
            "background_size": C.SHAP_BACKGROUND_N,
            "explain_size": C.SHAP_EVAL_N,
            "skip_configs": [],
        },
        "stability": {
            "n_bins": C.N_BINS,
            "bin_type": C.BIN_TYPE,
            "n_boot": C.N_BOOT,
            "n_perm": C.N_PERM,
            "alpha": C.ALPHA,
        },
    }


def _deep_update(base: dict, overlay: dict) -> dict:
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(reload: bool = False) -> dict:
    """Return the run config: defaults from config.py, overlaid by config.yaml."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and not reload:
        return _CONFIG_CACHE
    cfg = _defaults()
    yaml_path = C.ROOT / "config.yaml"
    if yaml_path.exists():
        try:
            import yaml

            with open(yaml_path, "r", encoding="utf-8") as f:
                _deep_update(cfg, yaml.safe_load(f) or {})
        except ImportError:
            get_logger().warning("config.yaml present but PyYAML not installed; using defaults")
    _CONFIG_CACHE = cfg
    return cfg
