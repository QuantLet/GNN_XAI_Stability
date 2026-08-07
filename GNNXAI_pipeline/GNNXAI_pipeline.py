"""GNNXAI_pipeline -- the single end-to-end entry point that regenerates every
cache the other Quantlets consume.

Stage order (idempotent, disk-cached):
    embeddings -> hybrid -> shap -> stability -> calibration -> figures -> tables

This is a thin launcher for tools/run_pipeline.py in the GNN_XAI_Stability
repository. It is NOT self-contained: the full pipeline needs the raw IEEE-CIS
and Elliptic datasets (downloaded via scripts/00_prepare_data.py; Kaggle account
required) and produces several hundred MB of intermediate SHAP/embedding caches
that are git-ignored and archived externally (see the paper's Data-availability
statement). The downstream figure/table Quantlets, by contrast, each ship the
small committed CSV/parquet they need and run self-contained.

Usage (from a clone of the repository):
    python GNNXAI_pipeline.py                      # full pipeline
    python GNNXAI_pipeline.py --only shap,stability
    python GNNXAI_pipeline.py --configs xgb_tab,xgb_sage_l1
"""
from __future__ import annotations

import sys
from pathlib import Path

STAGES = ["embeddings", "hybrid", "shap", "stability", "calibration", "figures", "tables"]


def _find_repo_root(start: Path) -> Path | None:
    """Walk up looking for tools/run_pipeline.py (works whether this file lives in
    the repo under submission/quantlets/ or is run from a checkout)."""
    for p in [start, *start.parents]:
        if (p / "tools" / "run_pipeline.py").exists():
            return p
    return None


def main() -> int:
    root = _find_repo_root(Path(__file__).resolve())
    if root is None:
        print("GNNXAI_pipeline: repository entry point tools/run_pipeline.py not found.")
        print("This Quantlet is the orchestration driver; run it from a clone of")
        print("github.com/QuantLet/GNN_XAI_Stability. Stage order:")
        print("   " + " -> ".join(STAGES))
        print("Raw data: scripts/00_prepare_data.py (Kaggle). Large intermediate")
        print("caches are archived externally (see the Data-availability statement).")
        return 0
    sys.path.insert(0, str(root))
    from tools import run_pipeline  # noqa: E402
    print(f"GNNXAI_pipeline: dispatching to {root / 'tools' / 'run_pipeline.py'}")
    return run_pipeline.main() or 0


if __name__ == "__main__":
    raise SystemExit(main())
