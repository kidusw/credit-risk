"""Population stability: PSI of the scores and key features by issue quarter.

PSI compares a quarter's distribution with the training distribution, bin by bin:
sum((actual% - expected%) * ln(actual% / expected%)). Numeric variables use the deciles
of train as bin edges (ties merge bins), missing values get a bin of their own, and
categoricals and low-cardinality numerics use their values.
Industry bands: < 0.1 stable, 0.1-0.25 monitor, > 0.25 act.

Because the bins are train quantiles, PSI of a score is the same for the raw and the
calibrated PD (calibration is monotonic), so the raw model output is used.

The **characteristic stability report** (CSR) is the scorecard view of the same question:
for each scorecard characteristic, how many points the average applicant gains or loses
in a quarter compared with train. The characteristic shifts add up to the total score
shift, so the CSR says *which* characteristics moved the score.
"""

import numpy as np
import pandas as pd

from crisk.models.scorecard import FACTOR, OFFSET, Scorecard

BINS = 10
EPS = 1e-4  # floor for empty bins, so ln() stays finite
BANDS = ((0.1, "stable"), (0.25, "monitor"), (np.inf, "act"))


def band(value: float) -> str:
    return next(name for limit, name in BANDS if value < limit)


def _numeric_bins(expected: pd.Series, bins: int) -> np.ndarray:
    edges = np.unique(np.nanquantile(expected.astype(float), np.linspace(0, 1, bins + 1)))
    return edges[1:-1]  # interior edges; the outer bins are open-ended


def bin_shares(values: pd.Series, edges: np.ndarray | None) -> pd.Series:
    """Share of rows per bin. `edges` None: categorical levels. Missing is its own bin."""
    if edges is None:
        labels = values.astype(object).where(values.notna(), "missing").astype(str)
    else:
        idx = np.digitize(values.astype(float), edges)
        labels = pd.Series(np.where(values.isna(), "missing", idx.astype(str)), index=values.index)
    return labels.value_counts(normalize=True)


def psi(
    expected: pd.Series, actual: pd.Series, bins: int = BINS, categorical: bool = False
) -> float:
    """Population stability index of `actual` against `expected`.

    A numeric variable with no more distinct values than `bins` (term: 36 or 60) is binned
    by value, since its quantile edges would collapse into a single bin."""
    few_values = expected.nunique(dropna=True) <= bins
    edges = None if categorical or few_values else _numeric_bins(expected, bins)
    e, a = bin_shares(expected, edges), bin_shares(actual, edges)
    idx = e.index.union(a.index)
    e = e.reindex(idx, fill_value=0).clip(lower=EPS)
    a = a.reindex(idx, fill_value=0).clip(lower=EPS)
    return float(((a - e) * np.log(a / e)).sum())


def psi_by_period(
    df: pd.DataFrame,
    columns: list[str],
    period: str = "issue_quarter",
    reference: pd.Series | None = None,
    categorical: tuple[str, ...] = (),
) -> pd.DataFrame:
    """PSI of each column for each period, against the rows flagged by `reference`."""
    ref = df[reference] if reference is not None else df
    rows = []
    for p, part in df.groupby(period, observed=True, sort=True):
        row = {period: str(p), "n": len(part)}
        for c in columns:
            row[c] = psi(ref[c], part[c], categorical=c in categorical)
        rows.append(row)
    return pd.DataFrame(rows)


def characteristic_points(card: Scorecard, df: pd.DataFrame) -> pd.DataFrame:
    """Each applicant's points per scorecard characteristic (columns follow `card.selected`)."""
    k = len(card.selected)
    alpha = card.model.intercept_[0]
    woe = card._woe(df, card.selected).to_numpy()
    points = -FACTOR * (woe * card.model.coef_[0] + alpha / k) + OFFSET / k
    return pd.DataFrame(points, columns=card.selected, index=df.index)


def characteristic_report(points: pd.DataFrame, expected: pd.Series, actual: pd.Series):
    """Mean points per characteristic in `actual` rows minus `expected` rows, largest first."""
    shift = points[actual].mean() - points[expected].mean()
    out = pd.DataFrame(
        {
            "expected_points": points[expected].mean(),
            "actual_points": points[actual].mean(),
            "shift": shift,
        }
    )
    out = out.reindex(shift.abs().sort_values(ascending=False).index)
    return out.rename_axis("characteristic").reset_index()
