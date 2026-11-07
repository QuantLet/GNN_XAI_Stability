"""Predictive and calibration metrics (Sections 3.5, 5.3).

AUC, PR-AUC, Brier, ECE (15 equal-width bins), and the DeLong test for paired
AUC comparison against a same-learner tabular baseline.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """ECE with equal-width probability bins (Guo et al., 2017)."""
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        conf, acc = p[m].mean(), y[m].mean()
        ece += (m.sum() / len(p)) * abs(conf - acc)
    return float(ece)


def predictive_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return dict(
        auc=float(roc_auc_score(y, p)),
        pr_auc=float(average_precision_score(y, p)),
        brier=float(brier_score_loss(y, p)),
        ece=expected_calibration_error(y, p),
    )


# ----------------------------------------------------------------------------
# DeLong test for two correlated ROC AUCs
# ----------------------------------------------------------------------------
def _compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    x_sorted = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and x_sorted[j] == x_sorted[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def delong_test(y: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> tuple[float, float]:
    """Return (z, two-sided p) for AUC(p1) - AUC(p2) on the same labels y."""
    from scipy import stats

    pos = p1[y == 1], p2[y == 1]
    neg = p1[y == 0], p2[y == 0]
    m, n = len(pos[0]), len(neg[0])
    preds = [np.concatenate([pos[k], neg[k]]) for k in range(2)]

    aucs, v01, v10 = [], [], []
    for pr in preds:
        tx = _compute_midrank(pr[:m])
        ty = _compute_midrank(pr[m:])
        tz = _compute_midrank(pr)
        auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
        aucs.append(auc)
        v01.append((tz[:m] - tx) / n)
        v10.append(1.0 - (tz[m:] - ty) / m)
    v01, v10 = np.array(v01), np.array(v10)
    s01 = np.cov(v01)
    s10 = np.cov(v10)
    s = s01 / m + s10 / n
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    if var <= 0:
        return 0.0, 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    pval = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(pval)
