"""Downstream tree learners (Section 3.3) with pinned hyperparameters.

Builds the hybrid input x~ = [x || z] (Eq. 1) and fits XGBoost or CatBoost with
the Appendix A settings, identical across baseline / hybrid / control so that
any SHAP-distribution difference isolates the embedding columns.
"""
from __future__ import annotations

import numpy as np

from . import config as C


def build_input(X_tab: np.ndarray, emb: np.ndarray | None) -> np.ndarray:
    """Concatenate tabular features with embedding columns (Eq. 1)."""
    if emb is None:
        return X_tab
    return np.concatenate([X_tab, emb], axis=1)


def feature_names(tab_names: list[str], emb: np.ndarray | None, embed_tag: str | None) -> list[str]:
    if emb is None:
        return list(tab_names)
    k = emb.shape[1]
    tag = embed_tag or "emb"
    return list(tab_names) + [f"{tag}_{i}" for i in range(k)]


def fit_xgb(X_tr, y_tr, X_val, y_val, params: dict = C.XGB_PARAMS):
    from xgboost import XGBClassifier

    model = XGBClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model


def fit_catb(X_tr, y_tr, X_val, y_val, params: dict = C.CATB_PARAMS):
    from catboost import CatBoostClassifier, Pool

    model = CatBoostClassifier(**params)
    model.fit(Pool(X_tr, y_tr), eval_set=Pool(X_val, y_val), verbose=False)
    return model


def fit_learner(learner: str, X_tr, y_tr, X_val, y_val, params: dict | None = None):
    if learner == "xgb":
        return fit_xgb(X_tr, y_tr, X_val, y_val, params or C.XGB_PARAMS)
    if learner == "catb":
        return fit_catb(X_tr, y_tr, X_val, y_val, params or C.CATB_PARAMS)
    raise ValueError(f"learner {learner!r} has no tree model (sage_only is GNN-only)")


def predict_proba(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]
