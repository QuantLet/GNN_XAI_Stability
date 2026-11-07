"""Central configuration for the GNN-XAI stability pipeline.

All settings are pinned to the values reported in the paper (Appendix A,
Sections 3-4). Reconstructed from the manuscript; see STATUS.md for provenance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
MODELS_DIR = ROOT / "models"
STABILITY_DIR = ROOT / "stability"
RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

for _d in (RAW_DIR, INTERIM_DIR, MODELS_DIR, STABILITY_DIR, TABLES_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Compute device (auto-detect; the GNN uses it, with a CPU fallback on OOM)
# ----------------------------------------------------------------------------
def default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


DEVICE = default_device()


def default_torch_device() -> str:
    """Accelerator for the GNN only.

    Kept separate from DEVICE because DEVICE also feeds the tree learners, and
    neither XGBoost (`device=`) nor CatBoost (`task_type="GPU"`) accepts Apple
    Metal -- both are CUDA-only, so on an Apple-silicon box the trees must stay
    on CPU while GraphSAGE runs on "mps".
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


TORCH_DEVICE = default_torch_device()

# ----------------------------------------------------------------------------
# Determinism (Appendix A: all RNGs seeded at 42)
# ----------------------------------------------------------------------------
SEED = 42
SHAP_BACKGROUND_SEED = 42      # 2,000-row interventional background
SHAP_EVAL_SEED = 47            # 5,000-row stratified explanation subsample
NULL_SEED_B = 43               # re-seeded baseline copy (Appendix E)
NULL_BAND_SEEDS = list(range(10, 30))  # 20 seeds for the empirical null band

# ----------------------------------------------------------------------------
# IEEE-CIS data setup (Section 4)
# ----------------------------------------------------------------------------
TARGET = "isFraud"
TIME_COL = "TransactionDT"
ID_COL = "TransactionID"
TEMPORAL_CUTOFF_PCT = 0.70     # 70th percentile of TransactionDT
RANDOM_HOLDOUT_FRAC = 0.30     # stratified on isFraud, seed 42
N_TABULAR_FEATURES = 486       # p, after preprocessing

# Four relation-defining entity columns (Section 3.2)
ENTITY_COLS = ["addr1", "DeviceType", "P_emaildomain", "id_31"]

# High-cardinality string categoricals -> frequency encoded (Section 4.1)
FREQ_ENCODE_COLS = [
    "P_emaildomain", "R_emaildomain", "DeviceInfo",
    "id_30", "id_31", "id_33",
]

# Low-cardinality categoricals -> one-hot (with a MISSING level for NaN).
# Reverse-engineered from the recovered deposit's 486-feature list (Appendix B).
ONEHOT_COLS = [
    "ProductCD", "card4", "card6", "DeviceType",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "id_12", "id_15", "id_16", "id_23", "id_27", "id_28", "id_29",
    "id_34", "id_35", "id_36", "id_37", "id_38",
]

# Fold-specific TransactionAmt summary statistics, grouped by these keys.
# addr1/card1 get the full 6 stats; the freq-encoded keys get 4 (no count/sm_mean),
# matching the deposit's 20 TransactionAmt columns.
AMT_GROUPS = {
    "addr1": ["mean", "std", "min", "max", "count", "sm_mean"],
    "card1": ["mean", "std", "min", "max", "count", "sm_mean"],
    "DeviceInfo_freq": ["mean", "std", "min", "max"],
    "P_emaildomain_freq": ["mean", "std", "min", "max"],
}
AMT_SMOOTHING = 10.0

# Canonical 486-feature column list (ground truth from the recovered deposit).
FEATURE_LIST_PATH = ROOT / "src" / "feature_columns_486.json"


# ----------------------------------------------------------------------------
# Hyperparameters (Appendix A, Table A.10) -- pinned across all configurations
# ----------------------------------------------------------------------------
XGB_PARAMS = dict(
    n_estimators=1000,
    max_depth=10,
    learning_rate=0.05,
    tree_method="hist",
    device=DEVICE,                 # "cuda" when a GPU is available
    objective="binary:logistic",
    eval_metric="auc",
    random_state=SEED,
    n_jobs=-1,
)

CATB_PARAMS = dict(
    iterations=3000,
    depth=8,
    learning_rate=0.05,
    early_stopping_rounds=50,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=SEED,
    verbose=False,
    **({"task_type": "GPU", "devices": "0"} if DEVICE == "cuda" else {}),
)

SAGE_PARAMS = dict(
    hidden_dim=64,
    num_layers=2,
    aggregator="mean",
    lr=1e-3,
    weight_decay=1e-4,
    patience=50,
    max_epochs=200,
)
EMBED_DIM = 64

# ----------------------------------------------------------------------------
# SHAP / stability diagnostic (Section 3.4, Algorithm 1)
# ----------------------------------------------------------------------------
SHAP_BACKGROUND_N = 2000
SHAP_EVAL_N = 5000
N_BINS = 50                    # quantile-spaced (primary); equal-width for robustness
BIN_TYPE = "quantile"          # {"quantile", "equalwidth"}
N_BOOT = 1000                  # block-bootstrap replicates
N_PERM = 1000                  # block-permutation replicates
ALPHA = 0.05                   # BH level
BLOCK_SECONDS = 86400          # daily blocks: floor(TransactionDT / 86400)
NOISE_INJ_SIGMA = 0.1          # noise-injected control: 0.1 * sigma_hat


# ----------------------------------------------------------------------------
# Configuration matrix (Section 4.3) -- 14 variants
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Variant:
    name: str
    learner: str          # {"xgb", "catb", "sage"}
    embed: str | None     # {None, "l1", "l2", "random", "permuted", "noise", "covmatched"}
    shap: bool = True     # sage_only is excluded from SHAP analysis


VARIANTS: list[Variant] = [
    # tabular baselines
    Variant("xgb_tab", "xgb", None),
    Variant("catb_tab", "catb", None),
    # pure relational baseline (excluded from SHAP)
    Variant("sage_only", "sage", "l1", shap=False),
    # real-embedding hybrids
    Variant("xgb_sage_l1", "xgb", "l1"),
    Variant("xgb_sage_l2", "xgb", "l2"),
    Variant("catb_sage_l1", "catb", "l1"),
    # diagonal-scale + covariance-matched controls (Section 3.5)
    Variant("xgb_random_emb", "xgb", "random"),
    Variant("xgb_permuted_emb", "xgb", "permuted"),
    Variant("xgb_noise_inj", "xgb", "noise"),
    Variant("xgb_covmatched_emb", "xgb", "covmatched"),
    Variant("catb_random_emb", "catb", "random"),
    Variant("catb_permuted_emb", "catb", "permuted"),
    Variant("catb_noise_inj", "catb", "noise"),
    Variant("catb_covmatched_emb", "catb", "covmatched"),
]
VARIANTS_BY_NAME = {v.name: v for v in VARIANTS}
SPLITS = ["random", "temporal"]


def seed_everything(seed: int = SEED) -> None:
    """Seed all RNGs (numpy, random, torch CPU+CUDA) with cuDNN determinism."""
    import os
    import random
    import numpy as np

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
