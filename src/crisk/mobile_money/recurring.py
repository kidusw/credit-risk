"""Project 1's recurring-payment detector, copied unchanged (txcat.insights.recurring).

Source: github.com/kidusw/transaction-categorization, `src/txcat/insights/recurring.py`.
Copied rather than imported so crisk installs and tests without the txcat package.

Recurring-payment detection: same counterparty, regular interval, stable amount.

Transactions are grouped into *series* per account by counterparty (the normalised
merchant, falling back to the category when no merchant matched, e.g. salary from an
employer outside the catalog) and direction (money in vs out). A series is recurring
when it has enough occurrences, its typical gap matches a known period, most gaps are
close to that period, and the amount is stable. Medians and MADs are used throughout so
a single missed month or an amount spike doesn't break detection.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Period name -> (typical days, accepted range of the median gap in days)
PERIODS: dict[str, tuple[int, tuple[float, float]]] = {
    "weekly": (7, (6, 8)),
    "biweekly": (14, (12, 16)),
    "monthly": (30, (26, 35)),
    "quarterly": (91, (82, 100)),
    "yearly": (365, (350, 380)),
}


@dataclass(frozen=True)
class RecurringConfig:
    min_occurrences: int = 3
    gap_tolerance: float = 0.25  # a gap is "on schedule" within ±25% of the median gap
    min_on_schedule: float = 0.75  # share of gaps that must be on schedule
    max_amount_spread: float = 0.35  # robust CV of |amount|: MAD / median


def series_key(df: pd.DataFrame) -> pd.Series:
    """Counterparty key: merchant if matched, else category; plus direction."""
    counterparty = df["merchant"].fillna(df["category"]) if "merchant" in df else df["category"]
    direction = np.where(df["amount"] > 0, "in", "out")
    return counterparty.astype(str) + " | " + direction


def _classify_period(median_gap: float) -> str | None:
    for name, (_, (low, high)) in PERIODS.items():
        if low <= median_gap <= high:
            return name
    return None


def _robust_spread(values: np.ndarray) -> float:
    median = np.median(values)
    return float(np.median(np.abs(values - median)) / median) if median else np.inf


def detect_recurring(df: pd.DataFrame, config: RecurringConfig | None = None) -> pd.DataFrame:
    """One row per recurring series: account, counterparty, period, typical amount, next due.

    `df` needs account_id, date, amount, category and optionally merchant.
    """
    config = config or RecurringConfig()
    work = df.assign(series=series_key(df)).sort_values("date")
    found = []
    for (account, series), group in work.groupby(["account_id", "series"], sort=False):
        if len(group) < config.min_occurrences:
            continue
        dates = group["date"].drop_duplicates()
        gaps = dates.diff().dt.days.dropna().to_numpy(dtype=float)
        if len(gaps) < config.min_occurrences - 1:
            continue
        median_gap = float(np.median(gaps))
        period = _classify_period(median_gap)
        if period is None:
            continue
        on_schedule = float(np.mean(np.abs(gaps - median_gap) <= config.gap_tolerance * median_gap))
        amounts = group["amount"].abs().to_numpy(dtype=float)
        spread = _robust_spread(amounts)
        if on_schedule < config.min_on_schedule or spread > config.max_amount_spread:
            continue
        counterparty, direction = series.rsplit(" | ", 1)
        last = group["date"].max()
        found.append(
            {
                "account_id": account,
                "counterparty": counterparty,
                "direction": direction,
                "category": group["category"].mode().iat[0],
                "period": period,
                "occurrences": len(group),
                "median_gap_days": median_gap,
                "on_schedule": on_schedule,
                "typical_amount": round(float(np.median(amounts)), 2),
                "amount_spread": spread,
                "first_seen": group["date"].min(),
                "last_seen": last,
                "next_expected": last + pd.Timedelta(days=round(median_gap)),
            }
        )
    columns = [
        "account_id", "counterparty", "direction", "category", "period", "occurrences",
        "median_gap_days", "on_schedule", "typical_amount", "amount_spread", "first_seen",
        "last_seen", "next_expected",
    ]  # fmt: skip
    return pd.DataFrame(found, columns=columns)


def mark_recurring(df: pd.DataFrame, recurring: pd.DataFrame) -> pd.Series:
    """Boolean per transaction: does it belong to a detected recurring series?"""
    detected = set(
        zip(
            recurring["account_id"],
            recurring["counterparty"] + " | " + recurring["direction"],
            strict=True,
        )
    )
    keys = zip(df["account_id"], series_key(df), strict=True)
    return pd.Series([k in detected for k in keys], index=df.index)
