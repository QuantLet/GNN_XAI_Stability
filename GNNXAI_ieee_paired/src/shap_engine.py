"""TreeExplainer SHAP attributions (Section 3.4).

Interventional reference, raw-margin (log-odds) output for the positive class.
Background: 2,000-row stratified sample from the training fold of the same split
(seed 42). Evaluation: 5,000-row stratified subsample of the validation fold
(seed 47), preserving the ~3.5% fraud rate.
"""
from __future__ import annotations

import numpy as np

from . import config as C


def stratified_sample(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Indices of a class-stratified subsample preserving the base rate."""
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    frac = n / len(y)
    n_pos = min(len(pos), max(1, int(round(len(pos) * frac))))
    n_neg = min(len(neg), n - n_pos)
    idx = np.concatenate([rng.choice(pos, n_pos, replace=False),
                          rng.choice(neg, n_neg, replace=False)])
    rng.shuffle(idx)
    return idx


def compute_shap(model, X_train, y_train, X_eval, y_eval,
                 background_n: int = C.SHAP_BACKGROUND_N,
                 eval_n: int = C.SHAP_EVAL_N):
    """Return (shap_values [n_eval, p], eval_idx) on the raw-margin scale.

    The background includes embedding columns for hybrid/control configs because
    it is drawn from the same X_train passed in (which already carries them).
    """
    import shap

    bg_idx = stratified_sample(y_train, background_n, C.SHAP_BACKGROUND_SEED)
    ev_idx = stratified_sample(y_eval, eval_n, C.SHAP_EVAL_SEED)
    background = X_train[bg_idx]

    explainer = shap.TreeExplainer(
        model,
        data=background,
        feature_perturbation="interventional",
        model_output="raw",
    )
    sv = explainer.shap_values(X_eval[ev_idx], check_additivity=False)
    if isinstance(sv, list):          # some versions return [class0, class1]
        sv = sv[1]
    return np.asarray(sv), ev_idx


# ----------------------------------------------------------------------------
# Original pipeline API
# ----------------------------------------------------------------------------
def compute_shap_for(config_id: str, split: str, explain_size: int | None = None):
    """Stage-3 entry point: compute and cache SHAP for one (config, split).

    Reloads the fitted model and cached split, rebuilds the exact training input,
    computes interventional TreeExplainer SHAP, and writes
    stability/shap_<config_id>_<split>.parquet (tabular + embedding columns).
    Also persists the eval-row daily block ids once per split for Algorithm 1.
    """
    import pandas as pd

    from . import config as C
    from . import hybrid, models, pipeline

    bundle = hybrid.load_model(config_id, split)
    sd = pipeline.load_split(split)
    y = sd.df[C.TARGET].values
    tr, val = sd.train_mask, ~sd.train_mask

    emb = None
    if bundle["embed"] is not None:
        from . import embeddings
        emb = embeddings.embedding_for_variant(bundle["embed"], sd.z1, sd.z2)
    X = models.build_input(sd.X_tab, emb)
    names = models.feature_names(sd.tab_names, emb, bundle["embed"])

    eval_n = explain_size or C.SHAP_EVAL_N
    sv, ev_idx = compute_shap(bundle["model"], X[tr], y[tr], X[val], y[val], eval_n=eval_n)

    cols = names[: sv.shape[1]]
    pd.DataFrame(sv, columns=cols).to_parquet(
        C.STABILITY_DIR / f"shap_{config_id}_{split}.parquet")

    blk_path = C.STABILITY_DIR / f"eval_blocks_{split}.npy"
    if not blk_path.exists():
        np.save(blk_path, sd.blocks[val][ev_idx])
    return sv


def rank_correlations(config_ids: list[str], split: str = "random"):
    """Spearman/Kendall rank correlations between mean-|SHAP| feature rankings.

    A descriptive cross-config check (Section 3.5). Writes a correlation matrix
    CSV to results/tables/ and returns it.
    """
    import pandas as pd
    from scipy.stats import kendalltau, spearmanr

    from . import config as C

    rankings = {}
    for cid in config_ids:
        path = C.STABILITY_DIR / f"shap_{cid}_{split}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        rankings[cid] = df.abs().mean(axis=0)

    ids = list(rankings)
    common = set.intersection(*[set(r.index) for r in rankings.values()]) if ids else set()
    common = sorted(common)
    rows = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            ra, rb = rankings[a][common], rankings[b][common]
            rows.append(dict(
                config_a=a, config_b=b,
                spearman=float(spearmanr(ra, rb).statistic),
                kendall=float(kendalltau(ra, rb).statistic),
            ))
    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(C.TABLES_DIR / f"shap_rank_correlations_{split}.csv", index=False)
    return out
