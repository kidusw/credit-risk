"""Loan economics from realised outcomes: profit, loss given default, expected loss.

Outcome columns (payments, recoveries) are post-origination. They are used here only to
*measure* what a decision would have earned or lost, and to estimate LGD from charged-off
train loans, never as model inputs.

Simplifications (stated in the README): no servicing fee, no discounting of cash flows,
EAD = the funded amount (the plan's definition), LGD as a share of that amount.
"""

import numpy as np
import pandas as pd


def loan_economics(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Per-loan realised profit and principal loss.

    profit         = everything received (principal, interest, late fees, recoveries)
                     minus collection fees minus the amount lent
    principal_loss = for defaults, principal never repaid net of recoveries; 0 otherwise
    lgd            = principal_loss / funded_amnt (defaults only)
    """
    o = outcomes
    net_recoveries = o["recoveries"] - o["collection_recovery_fee"]
    unpaid = (o["funded_amnt"] - o["total_rec_prncp"] - net_recoveries).clip(lower=0)
    out = pd.DataFrame(
        {
            "id": o["id"],
            "funded_amnt": o["funded_amnt"].astype("float64"),
            "profit": (o["total_pymnt"] - o["collection_recovery_fee"] - o["funded_amnt"]).astype(
                "float64"
            ),
            "principal_loss": np.where(o["default"] == 1, unpaid, 0.0),
        }
    )
    out["lgd"] = np.where(o["default"] == 1, out["principal_loss"] / out["funded_amnt"], np.nan)
    return out


def lgd_table(train: pd.DataFrame) -> pd.Series:
    """Mean LGD of defaulted train loans by (grade, term); `train` has grade, term, lgd."""
    bad = train[train["default"] == 1]
    return bad.groupby(["grade", "term_months"], observed=True)["lgd"].mean()


def assign_lgd(df: pd.DataFrame, table: pd.Series, fallback: float) -> np.ndarray:
    keys = pd.MultiIndex.from_arrays([df["grade"].astype(str), df["term_months"]])
    table = table.copy()
    table.index = table.index.set_levels(table.index.levels[0].astype(str), level=0)
    return table.reindex(keys).fillna(fallback).to_numpy()


def expected_loss(pd_hat, lgd, ead) -> np.ndarray:
    return np.asarray(pd_hat) * np.asarray(lgd) * np.asarray(ead)
