"""GNN-XAI stability pipeline.

Re-implements the methodology of "Do Graph Embeddings Stabilize SHAP?
A Controlled Decomposition under Distribution Shift" (Diaconu & Pele).
See STATUS.md for reconstruction provenance.
"""
__all__ = [
    "config", "utils", "data_prep", "splits", "graph", "gnn", "embeddings",
    "models", "hybrid", "shap_engine", "stability", "metrics", "calibration",
    "nulls", "elliptic", "pipeline", "viz", "tables",
]

# --- macOS OpenMP load order (do not remove) ---------------------------------
# On macOS/arm64, libxgboost.dylib wants @rpath/libomp.dylib via an rpath that only
# exists under a Homebrew install. When torch is imported first, torch's bundled
# libomp is already resident and a SECOND OpenMP runtime gets loaded alongside it;
# the next XGBoost fit then dies with SIGSEGV and no Python traceback (verified on
# Apple silicon, torch 2.11.0 + xgboost 3.2.0). Importing xgboost first leaves a
# single OpenMP image loaded and the fit succeeds. Harmless everywhere else.
import sys as _sys

if _sys.platform == "darwin":
    try:
        import xgboost as _xgboost  # noqa: F401
    except ImportError:
        pass
