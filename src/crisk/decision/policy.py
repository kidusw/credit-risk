"""Approval policies and how much each would have earned.

Every policy here *declines a subset of loans Lending Club approved*: outcomes exist only
for approved loans, so "approve all" is Lending Club's actual book and the baseline.
Thresholds are chosen on validation vintages and applied unchanged to test.
"""

import numpy as np
import pandas as pd

APPROVAL_RATES = np.round(np.arange(0.01, 1.0001, 0.01), 2)
GRADES = list("ABCDEFG")


def evaluate(approve: np.ndarray, df: pd.DataFrame) -> dict[str, float]:
    """Book-level results for a boolean approval mask. `df` has profit, funded_amnt, default."""
    approve = np.asarray(approve, dtype=bool)
    a = df[approve]
    lent = float(a["funded_amnt"].sum())
    profit = float(a["profit"].sum())
    return {
        "approval_rate": float(approve.mean()),
        "loans": int(approve.sum()),
        "lent": lent,
        "profit": profit,
        "return_on_lent": profit / lent if lent else 0.0,
        "bad_rate": float(a["default"].mean()) if len(a) else 0.0,
    }


def sweep(pd_hat, df: pd.DataFrame, rates=APPROVAL_RATES) -> pd.DataFrame:
    """Approve the lowest-PD share r of applicants, for every r: the profit curve."""
    order = np.argsort(np.asarray(pd_hat), kind="stable")
    p = np.asarray(pd_hat)[order]
    profit = df["profit"].to_numpy()[order].cumsum()
    lent = df["funded_amnt"].to_numpy()[order].cumsum()
    bads = df["default"].to_numpy()[order].cumsum()
    n = len(p)
    rows = []
    for r in rates:
        k = max(1, int(np.ceil(r * n)))
        rows.append(
            {
                "approval_rate": float(r),
                "pd_cutoff": float(p[k - 1]),
                "loans": k,
                "lent": float(lent[k - 1]),
                "profit": float(profit[k - 1]),
                "return_on_lent": float(profit[k - 1] / lent[k - 1]),
                "bad_rate": float(bads[k - 1] / k),
            }
        )
    return pd.DataFrame(rows)


# Two lenders: one maximises total profit (capital is free and unlimited), one maximises
# return per dollar lent (capital is the constraint).
OBJECTIVES = {"profit": "profit", "return": "return_on_lent"}


def best_cutoff(curve: pd.DataFrame, objective: str = "profit") -> float:
    """PD cutoff that maximises the objective (ties -> the more inclusive cutoff)."""
    col = OBJECTIVES[objective]
    best = curve.loc[curve[col] == curve[col].max()].iloc[-1]
    return float(best["pd_cutoff"])


def grade_policy(df: pd.DataFrame) -> pd.DataFrame:
    """Approve grades A..g for each g: Lending Club's own grades as a cutoff policy."""
    grade = df["grade"].astype(str)
    rows = [
        {"worst_grade": g, **evaluate(grade.isin(GRADES[: i + 1]).to_numpy(), df)}
        for i, g in enumerate(GRADES)
    ]
    return pd.DataFrame(rows)


def margins(train: pd.DataFrame) -> pd.DataFrame:
    """Per sub-grade, from 36-month train loans: return on good loans, loss on bad ones.

    Both are realised and include interest received, so the rule below prices the whole
    loan, not just principal.
    """
    t = train[train["term_months"] == 36].assign(ret=lambda d: d["profit"] / d["funded_amnt"])
    good = t[t["default"] == 0].groupby(t["sub_grade"].astype(str))["ret"].mean()
    bad = -t[t["default"] == 1].groupby(t["sub_grade"].astype(str))["ret"].mean()
    return pd.DataFrame({"good_return": good, "bad_loss": bad})


def expected_profit_rule(pd_hat, df: pd.DataFrame, table: pd.DataFrame) -> np.ndarray:
    """Approve when (1 - PD) * good return > PD * bad loss for the loan's sub-grade.

    No tuning on validation or test: margins come from train, PDs from the calibrated model.
    """
    sub = df["sub_grade"].astype(str)
    good = sub.map(table["good_return"]).fillna(table["good_return"].mean()).to_numpy()
    bad = sub.map(table["bad_loss"]).fillna(table["bad_loss"].mean()).to_numpy()
    p = np.asarray(pd_hat)
    return (1 - p) * good - p * bad > 0
