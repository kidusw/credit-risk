"""PD calibration (Platt vs isotonic) and calibration metrics.

Both calibrators are fitted on validation vintages only. The method is chosen by 5-fold
cross-validation *within* validation (out-of-fold Brier), then refitted on all of it, so
test plays no part in the choice.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold

METHODS = ("platt", "isotonic")
EPS = 1e-6


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


@dataclass
class Calibrator:
    method: str
    model: object = None

    def fit(self, p, y) -> "Calibrator":
        y = np.asarray(y)
        if self.method == "platt":
            self.model = LogisticRegression(penalty=None).fit(logit(p).reshape(-1, 1), y)
        elif self.method == "isotonic":
            self.model = IsotonicRegression(y_min=EPS, y_max=1 - EPS, out_of_bounds="clip")
            self.model.fit(np.asarray(p, dtype=float), y)
        else:
            raise ValueError(f"unknown calibration method {self.method!r}")
        return self

    def transform(self, p) -> np.ndarray:
        if self.method == "platt":
            return self.model.predict_proba(logit(p).reshape(-1, 1))[:, 1]
        return self.model.predict(np.asarray(p, dtype=float))


def select_method(p, y, folds: int = 5, seed: int = 0) -> tuple[str, pd.DataFrame]:
    """Out-of-fold Brier and ECE for each method on the calibration set; lowest Brier wins."""
    p, y = np.asarray(p, dtype=float), np.asarray(y)
    rows = []
    for method in METHODS:
        oof = np.empty_like(p)
        for fit_idx, eval_idx in StratifiedKFold(folds, shuffle=True, random_state=seed).split(
            p, y
        ):
            oof[eval_idx] = Calibrator(method).fit(p[fit_idx], y[fit_idx]).transform(p[eval_idx])
        rows.append({"method": method, **calibration_metrics(y, oof)})
    table = pd.DataFrame(rows).set_index("method")
    return str(table["brier"].idxmin()), table


def expected_calibration_error(y, p, bins: int = 20) -> float:
    """ECE over equal-frequency bins (PDs cluster at 5-30%, so equal-width bins are empty)."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    g = df.groupby("bin").agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    return float((g["n"] / g["n"].sum() * (g["y"] - g["p"]).abs()).sum())


def calibration_metrics(y, p) -> dict[str, float]:
    """Brier, ECE, calibration-in-the-large, and the slope/intercept of y ~ logit(p)."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    fit = LogisticRegression(penalty=None).fit(logit(p).reshape(-1, 1), y)
    return {
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "mean_pd": float(p.mean()),
        "default_rate": float(y.mean()),
        "citl": float(p.mean() - y.mean()),  # calibration-in-the-large: + over-predicts
        "slope": float(fit.coef_[0][0]),  # 1 = well spread; < 1 = PDs too extreme
        "intercept": float(fit.intercept_[0]),
    }


def reliability_table(y, p, bins: int = 10) -> pd.DataFrame:
    """Mean predicted vs observed default rate per equal-frequency PD bin."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    return (
        df.groupby("bin")
        .agg(mean_pd=("p", "mean"), default_rate=("y", "mean"), n=("y", "size"))
        .reset_index()
    )
