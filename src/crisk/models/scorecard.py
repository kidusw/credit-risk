"""WoE + logistic regression scorecard, scaled to points like a bank's.

1. Monotonic optimal binning per feature (optbinning); WoE per bin; information value (IV).
2. Screening: IV >= MIN_IV, then drop the weaker of any pair of WoE features with
   |correlation| > MAX_CORR (keeps coefficients stable and interpretable).
3. Logistic regression on WoE. Any feature whose coefficient has the wrong sign (riskier bins
   scored as safer) is dropped and the model refit, until every sign is right.
4. Points: score = OFFSET + FACTOR * ln(good:bad odds), so BASE_POINTS at BASE_ODDS and
   +PDO points each time the odds double. Each bin gets
   points = -FACTOR * (beta * WoE + intercept / k) + OFFSET / k  for k features, rounded.
"""

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from optbinning import BinningProcess
from sklearn.linear_model import LogisticRegression

from crisk.models.features import as_binning_input, categorical

BASE_POINTS = 600
BASE_ODDS = 50  # good:bad
PDO = 20
FACTOR = PDO / math.log(2)
OFFSET = BASE_POINTS - FACTOR * math.log(BASE_ODDS)

MIN_IV = 0.02
MAX_CORR = 0.7


def iv_strength(iv: float) -> str:
    """Conventional reading of information value."""
    if iv < 0.02:
        return "useless"
    if iv < 0.1:
        return "weak"
    if iv < 0.3:
        return "medium"
    if iv < 0.5:
        return "strong"
    return "very strong (check for leakage)"


@dataclass
class Scorecard:
    features: list[str]
    min_bin_size: float = 0.05
    max_n_bins: int = 8
    binning: BinningProcess | None = None
    model: LogisticRegression | None = None
    selected: list[str] = field(default_factory=list)
    dropped: dict[str, str] = field(default_factory=dict)
    iv: pd.DataFrame | None = None

    def _woe(self, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        woe = self.binning.transform(as_binning_input(df, self.features), metric="woe")
        return pd.DataFrame(np.asarray(woe), columns=self.features, index=df.index)[features]

    def fit(self, df: pd.DataFrame, y: pd.Series) -> "Scorecard":
        cats = categorical(self.features)
        fit_params = {
            f: {"monotonic_trend": "auto_asc_desc"} for f in self.features if f not in cats
        }
        self.binning = BinningProcess(
            self.features,
            categorical_variables=cats,
            min_bin_size=self.min_bin_size,
            max_n_bins=self.max_n_bins,
            binning_fit_params=fit_params,
        )
        self.binning.fit(as_binning_input(df, self.features), y.to_numpy())

        summary = self.binning.summary().set_index("name")
        self.iv = (
            summary[["dtype", "n_bins", "iv", "js", "gini"]]
            .assign(strength=lambda s: s["iv"].map(iv_strength))
            .sort_values("iv", ascending=False)
        )
        candidates = [f for f in self.iv.index if self.iv.loc[f, "iv"] >= MIN_IV]
        for f in self.iv.index.difference(candidates):
            self.dropped[f] = f"IV {self.iv.loc[f, 'iv']:.3f} < {MIN_IV}"

        woe = self._woe(df, candidates)
        candidates = self._correlation_filter(woe, candidates)
        self.selected = self._fit_with_sign_check(woe, y, candidates)
        self.iv["selected"] = self.iv.index.isin(self.selected)
        self.iv["dropped_because"] = self.iv.index.map(self.dropped).fillna("")
        return self

    def _correlation_filter(self, woe: pd.DataFrame, candidates: list[str]) -> list[str]:
        corr = woe[candidates].corr().abs()
        kept: list[str] = []
        for f in candidates:  # in descending IV order: stronger feature wins
            clash = next((k for k in kept if corr.loc[f, k] > MAX_CORR), None)
            if clash is None:
                kept.append(f)
            else:
                self.dropped[f] = f"|corr| {corr.loc[f, clash]:.2f} with {clash}"
        return kept

    def _fit_with_sign_check(self, woe: pd.DataFrame, y: pd.Series, features: list[str]):
        # Expected sign: WoE that goes with more defaults must raise the PD.
        expected = np.sign([np.corrcoef(woe[f], y)[0, 1] for f in features])
        expected = dict(zip(features, expected, strict=True))
        features = list(features)
        while True:
            self.model = LogisticRegression(C=1.0, max_iter=1000).fit(woe[features], y)
            coef = dict(zip(features, self.model.coef_[0], strict=True))
            wrong = [f for f in features if np.sign(coef[f]) != expected[f]]
            if not wrong:
                return features
            worst = min(wrong, key=lambda f: self.iv.loc[f, "iv"])
            self.dropped[worst] = "wrong coefficient sign in the multivariate model"
            features.remove(worst)

    # --- scoring -------------------------------------------------------------------------

    def predict_pd(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._woe(df, self.selected))[:, 1]

    def points_table(self) -> pd.DataFrame:
        """One row per (feature, bin): counts, event rate, WoE and the points awarded."""
        k = len(self.selected)
        alpha = self.model.intercept_[0]
        rows = []
        for f, beta in zip(self.selected, self.model.coef_[0], strict=True):
            table = self.binning.get_binned_variable(f).binning_table.build(add_totals=False)
            for _, r in table.iterrows():
                points = -FACTOR * (beta * r["WoE"] + alpha / k) + OFFSET / k
                rows.append(
                    {
                        "feature": f,
                        "bin": _bin_label(r["Bin"]),
                        "count": int(r["Count"]),
                        "share": round(float(r["Count (%)"]), 4),
                        "default_rate": round(float(r["Event rate"]), 4),
                        "woe": round(float(r["WoE"]), 4),
                        "points": round(points),
                    }
                )
        return pd.DataFrame(rows)

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Integer score: sum of each applicant's (rounded) bin points."""
        k = len(self.selected)
        alpha = self.model.intercept_[0]
        woe = self._woe(df, self.selected).to_numpy()
        points = -FACTOR * (woe * self.model.coef_[0] + alpha / k) + OFFSET / k
        return np.rint(points).sum(axis=1).astype(int)


def score_to_pd(score) -> np.ndarray:
    """Invert the scaling: PD implied by a score (before bin-points rounding)."""
    odds = np.exp((np.asarray(score, dtype=float) - OFFSET) / FACTOR)
    return 1 / (1 + odds)


def _bin_label(b) -> str:
    """Numeric bins arrive as interval strings, categorical bins as arrays of categories."""
    if isinstance(b, str):
        return b
    return ", ".join(str(v) for v in list(b))
