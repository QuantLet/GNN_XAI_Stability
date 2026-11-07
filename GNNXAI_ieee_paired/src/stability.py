"""Algorithm 1: block-resampling JS stability diagnostic (Section 3.4).

Per feature k, compares random-split vs temporal-split SHAP distributions via the
Jensen-Shannon divergence, with block-bootstrap 95% intervals and a paired
block-permutation one-sided test on delta_k = JS_k^B - JS_k^A, BH-adjusted.

Read against the empirical null band of Appendix E (see nulls.py), not as a
calibrated false-discovery rate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ----------------------------------------------------------------------------
# Jensen-Shannon divergence on fixed histogram edges (Eq. 2)
# ----------------------------------------------------------------------------
def bin_edges(pooled: np.ndarray, n_bins: int, kind: str) -> np.ndarray:
    """Histogram edges from the pooled per-feature sample S_k, frozen thereafter."""
    finite = pooled[np.isfinite(pooled)]
    if finite.size == 0 or np.ptp(finite) == 0:
        # Freedman-Diaconis fallback on degenerate support
        c = finite[0] if finite.size else 0.0
        return np.linspace(c - 1e-6, c + 1e-6, n_bins + 1)
    if kind == "quantile":
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.quantile(finite, qs)
        edges = np.unique(edges)
        if edges.size < 3:  # collapsed quantiles -> equal-width fallback
            edges = np.linspace(finite.min(), finite.max(), n_bins + 1)
        return edges
    return np.linspace(finite.min(), finite.max(), n_bins + 1)


def _hist(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    h, _ = np.histogram(x[np.isfinite(x)], bins=edges)
    h = h.astype(np.float64)
    s = h.sum()
    return h / s if s > 0 else np.full_like(h, 1.0 / len(h))


def js_divergence(a: np.ndarray, b: np.ndarray, edges: np.ndarray) -> float:
    """JS divergence (base-e, range [0, ln 2]) between two samples on fixed edges."""
    p, q = _hist(a, edges), _hist(b, edges)
    m = 0.5 * (p + q)
    return float(0.5 * _kl(p, m) + 0.5 * _kl(q, m))


def _kl(p: np.ndarray, m: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / m[mask])))


# ----------------------------------------------------------------------------
# Benjamini-Hochberg
# ----------------------------------------------------------------------------
def bh_adjust(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


# ----------------------------------------------------------------------------
# Block resampling helpers
# ----------------------------------------------------------------------------
def _resample_blocks(blocks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Indices of a whole-block bootstrap resample within one split."""
    uniq = np.unique(blocks)
    chosen = rng.choice(uniq, size=len(uniq), replace=True)
    idx = [np.where(blocks == b)[0] for b in chosen]
    return np.concatenate(idx) if idx else np.arange(len(blocks))


@dataclass
class FeatureResult:
    js_a: float
    js_b: float
    ci_a: tuple[float, float]
    ci_b: tuple[float, float]
    delta: float
    p: float
    p_bh: float = np.nan
    ci_sep: bool = False
    # Opposite-tail companion of `p` (destabilisation: delta_k > 0). Counted in the
    # same permutation loop rather than inferred as 1 - p, because JS on a fixed
    # 50-bin histogram of 5,000 rows is discrete enough for ties to occur.
    p_rev: float = np.nan
    p_rev_bh: float = np.nan


def diagnostic(
    shap_a_rand: np.ndarray, shap_a_temp: np.ndarray,
    shap_b_rand: np.ndarray, shap_b_temp: np.ndarray,
    blocks_rand: np.ndarray, blocks_temp: np.ndarray,
    n_bins: int = 50, bin_type: str = "quantile",
    n_boot: int = 1000, n_perm: int = 1000, alpha: float = 0.05,
    seed: int = 42,
) -> list[FeatureResult]:
    """Run Algorithm 1 over all p features. Arrays are [n_rows, p] SHAP matrices."""
    p = shap_a_rand.shape[1]
    results: list[FeatureResult] = []
    rng = np.random.default_rng(seed)

    for k in range(p):
        ar, at = shap_a_rand[:, k], shap_a_temp[:, k]
        br, bt = shap_b_rand[:, k], shap_b_temp[:, k]
        pooled = np.concatenate([ar, at, br, bt])
        edges = bin_edges(pooled, n_bins, bin_type)

        js_a = js_divergence(ar, at, edges)
        js_b = js_divergence(br, bt, edges)
        delta = js_b - js_a

        # block bootstrap -> CIs
        boot_a = np.empty(n_boot)
        boot_b = np.empty(n_boot)
        for r in range(n_boot):
            ir, it = _resample_blocks(blocks_rand, rng), _resample_blocks(blocks_temp, rng)
            boot_a[r] = js_divergence(ar[ir], at[it], edges)
            boot_b[r] = js_divergence(br[ir], bt[it], edges)
        ci_a = (float(np.percentile(boot_a, 2.5)), float(np.percentile(boot_a, 97.5)))
        ci_b = (float(np.percentile(boot_b, 2.5)), float(np.percentile(boot_b, 97.5)))
        ci_sep = ci_b[1] < ci_a[0] or ci_a[1] < ci_b[0]

        # paired block-permutation: swap A/B labels per block, both splits
        ge = 0
        le = 0
        ur, ut = np.unique(blocks_rand), np.unique(blocks_temp)
        for r in range(n_perm):
            sr = {b: rng.random() < 0.5 for b in ur}
            st = {b: rng.random() < 0.5 for b in ut}
            ar2, br2 = _swap(ar, br, blocks_rand, sr)
            at2, bt2 = _swap(at, bt, blocks_temp, st)
            d = js_divergence(br2, bt2, edges) - js_divergence(ar2, at2, edges)
            if d <= delta:
                ge += 1
            if d >= delta:
                le += 1
        pval = (1 + ge) / (n_perm + 1)
        pval_rev = (1 + le) / (n_perm + 1)

        results.append(FeatureResult(js_a, js_b, ci_a, ci_b, delta, pval,
                                     ci_sep=ci_sep, p_rev=pval_rev))

    p_bh = bh_adjust(np.array([r.p for r in results]))
    p_rev_bh = bh_adjust(np.array([r.p_rev for r in results]))
    for r, pb, prb in zip(results, p_bh, p_rev_bh):
        r.p_bh = float(pb)
        r.p_rev_bh = float(prb)
    return results


def _swap(a: np.ndarray, b: np.ndarray, blocks: np.ndarray, flip: dict):
    """Apply a per-block A/B label swap, returning new (a, b) arrays."""
    mask = np.array([flip[bl] for bl in blocks])
    a2 = np.where(mask, b, a)
    b2 = np.where(mask, a, b)
    return a2, b2


def compare_pair(a_id: str, b_id: str, split_pair=("random", "temporal"),
                 write: bool = True) -> dict:
    """Stage-4 entry point: run Algorithm 1 on a (config A, config B) pair.

    Loads cached SHAP for both configs under both splits, restricts to the common
    tabular feature space, uses the persisted eval-row daily blocks (i.i.d. fallback
    with a warning if absent), and returns the aggregate stability summary.
    """
    import pandas as pd

    from . import config as C
    from . import utils

    log = utils.get_logger()
    cfg = utils.load_config()
    s_rand, s_temp = split_pair

    def _load(cid, split):
        p = C.STABILITY_DIR / f"shap_{cid}_{split}.parquet"
        if not p.exists():
            raise FileNotFoundError(p.name)
        return pd.read_parquet(p)

    ar, at = _load(a_id, s_rand), _load(a_id, s_temp)
    br, bt = _load(b_id, s_rand), _load(b_id, s_temp)
    feats = [c for c in ar.columns if c in set(br.columns)]   # common tabular features
    ar, at, br, bt = (d[feats].values for d in (ar, at, br, bt))

    def _blocks(split, n):
        p = C.STABILITY_DIR / f"eval_blocks_{split}.npy"
        if p.exists():
            return np.load(p)
        log.warning("  %s missing; using single-block (i.i.d.) resampling", p.name)
        return np.zeros(n, dtype=int)

    blk_r, blk_t = _blocks(s_rand, len(ar)), _blocks(s_temp, len(at))
    st = cfg["stability"]
    results = diagnostic(ar, at, br, bt, blk_r, blk_t,
                         n_bins=st["n_bins"], bin_type=st["bin_type"],
                         n_boot=st["n_boot"], n_perm=st["n_perm"], alpha=st["alpha"])
    summ = summarize(results, alpha=st["alpha"])
    summ.update(A=a_id, B=b_id)
    if write:
        # deposit-compatible schema + filename, so the recovered quantlet .py files
        # and src/viz.py read the same caches.
        per_feature = pd.DataFrame([{
            "feature": feats[i],
            "JS_A": r.js_a, "ci_low_A": r.ci_a[0], "ci_high_A": r.ci_a[1],
            "JS_B": r.js_b, "ci_low_B": r.ci_b[0], "ci_high_B": r.ci_b[1],
            "delta_obs": r.delta, "p_raw": r.p, "p_bh_fdr": r.p_bh,
            "significant_fdr": bool(r.p_bh < st["alpha"]),
            "resampling": "block_time",
        } for i, r in enumerate(results)]).set_index("feature")
        per_feature.to_parquet(
            C.STABILITY_DIR / f"stability_{a_id}_vs_{b_id}_block_time.parquet")
    return summ


def summarize(results: list[FeatureResult], alpha: float = 0.05) -> dict:
    """Aggregate stability summaries (mean JS, mean delta, flagged / CI-sep fractions)."""
    js_a = np.array([r.js_a for r in results])
    js_b = np.array([r.js_b for r in results])
    delta = np.array([r.delta for r in results])
    flagged = np.array([(r.p_bh < alpha) and (r.delta < 0) for r in results])
    ci_sep = np.array([r.ci_sep and (r.js_b < r.js_a) for r in results])
    return dict(
        n_features=len(results),
        mean_JS_A=float(js_a.mean()), mean_JS_B=float(js_b.mean()),
        median_JS_A=float(np.median(js_a)), median_JS_B=float(np.median(js_b)),
        mean_delta=float(delta.mean()), median_delta=float(np.median(delta)),
        n_flagged=int(flagged.sum()), frac_flagged=float(flagged.mean()),
        n_ci_sep=int(ci_sep.sum()), frac_ci_sep=float(ci_sep.mean()),
    )
