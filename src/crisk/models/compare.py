"""Challenger vs incumbent on the same loans: AUC lift with a paired bootstrap interval."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def auc_lift_ci(y, pd_challenger, pd_incumbent, n_boot: int = 200, seed: int = 0) -> dict:
    """AUC(challenger) - AUC(incumbent), with a 95% percentile interval.

    Paired: each resample draws the same loans for both models, so the interval reflects
    the difference, not two independent uncertainties.
    """
    y = np.asarray(y)
    a = np.asarray(pd_challenger)
    b = np.asarray(pd_incumbent)
    rng = np.random.default_rng(seed)
    lifts = np.empty(n_boot)
    n = len(y)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        lifts[i] = roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx])
    low, high = np.percentile(lifts, [2.5, 97.5])
    return {
        "auc_challenger": float(roc_auc_score(y, a)),
        "auc_incumbent": float(roc_auc_score(y, b)),
        "lift": float(roc_auc_score(y, a) - roc_auc_score(y, b)),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_boot": n_boot,
    }


def compare(
    scored: pd.DataFrame, pairs: list[tuple[str, str]], splits=("valid", "test"), n_boot=200
) -> pd.DataFrame:
    """`scored` has one PD column per model plus `split` and `default`."""
    rows = []
    for challenger, incumbent in pairs:
        for split in splits:
            s = scored[scored["split"] == split]
            stats = auc_lift_ci(s["default"], s[challenger], s[incumbent], n_boot=n_boot)
            rows.append({"challenger": challenger, "incumbent": incumbent, "split": split, **stats})
    return pd.DataFrame(rows)
