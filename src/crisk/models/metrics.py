"""Discrimination and calibration metrics, pooled and per group (vintage, grade, term)."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def ks_statistic(y: np.ndarray, pd_hat: np.ndarray) -> float:
    """Kolmogorov-Smirnov: max gap between the score distributions of bads and goods."""
    fpr, tpr, _ = roc_curve(y, pd_hat)
    return float(np.max(tpr - fpr))


def binary_metrics(y, pd_hat) -> dict[str, float]:
    y = np.asarray(y)
    pd_hat = np.asarray(pd_hat)
    auc = roc_auc_score(y, pd_hat)
    return {
        "n": len(y),
        "default_rate": float(y.mean()),
        "mean_pd": float(pd_hat.mean()),
        "auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": ks_statistic(y, pd_hat),
        "pr_auc": float(average_precision_score(y, pd_hat)),
        "brier": float(brier_score_loss(y, pd_hat)),
    }


def grouped_metrics(df: pd.DataFrame, by: str, y: str = "default", p: str = "pd") -> pd.DataFrame:
    """Metrics per group; groups with a single class get NaN discrimination metrics."""
    rows = []
    for key, g in df.groupby(by, observed=True):
        if g[y].nunique() < 2:
            continue
        rows.append({by: key, **binary_metrics(g[y], g[p])})
    return pd.DataFrame(rows)
