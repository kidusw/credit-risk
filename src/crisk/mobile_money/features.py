"""Alternative-data features from a borrower's 12 months of (synthetic) wallet history.

Only transactions dated before the application are used, and only transactions: no
latent trait, rule term or demographic reaches the feature table. Amounts are mostly
scaled by the borrower's own monthly inflow, so a trader and a salaried clerk are compared
on behaviour, not size; `inflow_monthly` keeps the level.

Income regularity uses Project 1's recurring-payment detector: the share of earned inflow
that arrives in a detected recurring series (same payer, regular interval, stable amount).
The same detector finds regular equb and iddir contributions and bill payments.
"""

import numpy as np
import pandas as pd

from crisk.mobile_money.generate import HISTORY_DAYS, MONTHS
from crisk.mobile_money.recurring import detect_recurring, mark_recurring

EARNED = ("Income", "Sales", "Cash In", "Remittance In")
DETECT = ("Income", "Bills", "Equb", "Iddir")  # series worth running the detector on
LOW_BALANCE_SHARE = 0.25  # "low" = under a quarter of monthly inflow

# +1: higher value -> PD can only go up; -1: only down. Same convention as the LC model.
MONOTONE = {
    "inflow_monthly": -1,
    "inflow_cv": +1,
    "months_without_inflow": +1,
    "income_regularity": -1,
    "inflow_outflow_ratio": -1,
    "balance_mean_to_inflow": -1,
    "balance_min_to_inflow": -1,
    "low_balance_days": +1,
    "balance_trend": -1,
    "recent_inflow_change": -1,
    "recent_income_dip": -1,
    "loan_to_inflow": +1,
    "airtime_gap_cv": +1,
    "equb_member": -1,
    "iddir_member": -1,
    "p2p_out_contacts": -1,
    "p2p_in_contacts": -1,
}
FEATURES = [
    "inflow_monthly",
    "inflow_cv",
    "months_without_inflow",
    "income_regularity",
    "inflow_outflow_ratio",
    "balance_mean_to_inflow",
    "balance_min_to_inflow",
    "low_balance_days",
    "balance_volatility",
    "balance_trend",
    "recent_inflow_change",
    "recent_income_dip",
    "cash_out_share",
    "loan_amount",
    "loan_to_inflow",
    "airtime_per_month",
    "airtime_gap_cv",
    "airtime_share",
    "regular_bills",
    "equb_member",
    "iddir_member",
    "p2p_out_contacts",
    "p2p_in_contacts",
    "txn_per_month",
]


def recurring_series(tx: pd.DataFrame) -> pd.DataFrame:
    """Project 1's detector on the series that can be recurring (not P2P, merchants, cash)."""
    work = tx[tx["category"].isin(DETECT)]
    work = pd.DataFrame(
        {
            "account_id": work["borrower_id"].to_numpy(),
            "date": work["date"].to_numpy(),
            "amount": work["amount"].to_numpy(dtype=float),
            "category": work["category"].astype(str).to_numpy(),
            "merchant": work["counterparty"].astype(str).to_numpy(),
        }
    )
    return work, detect_recurring(work)


def _daily_balances(tx: pd.DataFrame, ids: np.ndarray, opening: pd.Series) -> np.ndarray:
    """End-of-day balance for each borrower and history day (rows follow `ids`).

    `tx` is sorted by borrower and time; `day` counts days since the history started."""
    pos = pd.Series(np.arange(len(ids)), index=ids)
    row = pos.reindex(tx["borrower_id"]).to_numpy()
    day = tx["day"].to_numpy()
    grid = np.full((len(ids), HISTORY_DAYS), np.nan, dtype="float64")
    last = ~pd.DataFrame({"r": row, "d": day}).duplicated(keep="last").to_numpy()
    grid[row[last], day[last]] = tx["balance"].to_numpy()[last]
    grid[:, 0] = np.where(np.isnan(grid[:, 0]), opening.reindex(ids).to_numpy(), grid[:, 0])
    return pd.DataFrame(grid).ffill(axis=1).to_numpy()


def build(tx: pd.DataFrame, borrowers: pd.DataFrame) -> pd.DataFrame:
    """One row per borrower: id, the FEATURES, and nothing hidden."""
    tx = tx.sort_values(["borrower_id", "date", "amount"], ascending=[True, True, False])
    ids = borrowers["borrower_id"].to_numpy()
    b = borrowers.set_index("borrower_id")
    n_ids = pd.Index(ids)
    month = (tx["day"] // 30).clip(upper=MONTHS - 1).to_numpy()
    cat = tx["category"].astype(str)
    amt = tx["amount"].astype(float)

    is_earned = cat.isin(EARNED).to_numpy()
    earned = tx[is_earned]
    monthly = (
        earned.groupby([earned["borrower_id"], month[is_earned]])["amount"]
        .sum()
        .unstack(fill_value=0)
        .reindex(index=n_ids, columns=range(MONTHS), fill_value=0)
    )
    inflow_m = monthly.mean(axis=1)
    f = pd.DataFrame(index=n_ids)
    f["inflow_monthly"] = inflow_m
    f["inflow_cv"] = monthly.std(axis=1) / inflow_m.replace(0, np.nan)
    f["months_without_inflow"] = (monthly <= 0).sum(axis=1)
    recent = monthly.iloc[:, -3:].mean(axis=1)
    prior = monthly.iloc[:, :-3].mean(axis=1)
    f["recent_inflow_change"] = recent / prior.replace(0, np.nan) - 1
    # A shock is one bad month: the worst of the last three vs the typical earlier month.
    typical = monthly.iloc[:, :-3].median(axis=1).replace(0, np.nan)
    f["recent_income_dip"] = (monthly.iloc[:, -3:].min(axis=1) / typical).clip(upper=3)

    work, series = recurring_series(tx)
    is_rec = mark_recurring(work, series).to_numpy()
    rec_in = work[is_rec & (work["amount"] > 0) & (work["category"] == "Income")]
    rec_income = rec_in.groupby("account_id")["amount"].sum().reindex(n_ids, fill_value=0)
    total_earned = (inflow_m * MONTHS).replace(0, np.nan)
    f["income_regularity"] = (rec_income / total_earned).fillna(0).clip(0, 1)
    found = series.groupby(["account_id", "category"]).size().unstack(fill_value=0)
    found = found.reindex(index=n_ids, fill_value=0)
    for c in ("Equb", "Iddir", "Bills"):
        if c not in found:
            found[c] = 0
    f["equb_member"] = (found["Equb"] > 0).astype(int)
    f["iddir_member"] = (found["Iddir"] > 0).astype(int)
    f["regular_bills"] = found["Bills"]

    inflow_all = amt.clip(lower=0).groupby(tx["borrower_id"]).sum().reindex(n_ids)
    outflow_all = (-amt).clip(lower=0).groupby(tx["borrower_id"]).sum().reindex(n_ids)
    f["inflow_outflow_ratio"] = inflow_all / outflow_all.replace(0, np.nan)
    cash_out = (-amt[(cat == "Cash Out").to_numpy()]).groupby(tx["borrower_id"]).sum()
    f["cash_out_share"] = (cash_out.reindex(n_ids, fill_value=0) / outflow_all).fillna(0)

    grid = _daily_balances(tx, ids, b["opening_balance"])
    scale = inflow_m.replace(0, np.nan).to_numpy()
    f["balance_mean_to_inflow"] = grid.mean(axis=1) / scale
    f["balance_min_to_inflow"] = grid.min(axis=1) / scale
    f["low_balance_days"] = (grid < LOW_BALANCE_SHARE * scale[:, None]).mean(axis=1)
    f["balance_volatility"] = grid.std(axis=1) / np.maximum(grid.mean(axis=1), 1)
    f["balance_trend"] = (grid[:, -90:].mean(axis=1) - grid[:, :90].mean(axis=1)) / scale

    air = tx[(cat == "Airtime").to_numpy()]
    f["airtime_per_month"] = air.groupby("borrower_id").size().reindex(n_ids, fill_value=0) / MONTHS
    gaps = air.groupby("borrower_id")["day"].diff().dropna()
    gstats = gaps.groupby(air.loc[gaps.index, "borrower_id"]).agg(["mean", "std"])
    f["airtime_gap_cv"] = (gstats["std"] / gstats["mean"].replace(0, np.nan)).reindex(n_ids)
    f["airtime_share"] = (-air["amount"].astype(float)).groupby(air["borrower_id"]).sum().reindex(
        n_ids, fill_value=0
    ) / (inflow_m * MONTHS).replace(0, np.nan)

    for col, c in (("p2p_out_contacts", "P2P Out"), ("p2p_in_contacts", "Remittance In")):
        sub = tx[(cat == c).to_numpy()]
        f[col] = sub.groupby("borrower_id")["counterparty"].nunique().reindex(n_ids, fill_value=0)
    f["txn_per_month"] = tx.groupby("borrower_id").size().reindex(n_ids, fill_value=0) / MONTHS

    f["loan_amount"] = b["loan_amount"].reindex(n_ids)
    f["loan_to_inflow"] = f["loan_amount"] / inflow_m.replace(0, np.nan)
    out = f[FEATURES].astype("float32").rename_axis("borrower_id").reset_index()
    out["source"] = "synthetic"
    return out
