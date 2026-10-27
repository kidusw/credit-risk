"""Group fairness of an approval policy on the attributes Lending Club publishes.

Lending Club's file has no sex, race or age, so none of those is measured or inferred here.
What it does have: state, home ownership, employment length, income and the 3-digit zip
(a known proxy for race, kept out of the model and used only for this audit).

Three standard views, per group, on a fixed approval mask:

- **Demographic parity:** approval rate, and its ratio to the most-approved group (the
  "adverse impact ratio"; US practice flags ratios below 0.8, the four-fifths rule).
- **Equal opportunity:** the approval rate among loans that were repaid (the true-positive
  rate, with "good loan approved" as the positive), and its gap to the best group.
- **Calibration by group:** mean PD minus the realised default rate. If the PDs are right
  within every group, a difference in approval rates reflects a difference in risk.

When default rates differ between groups, these three can't all hold at once, so the
tables report all three rather than optimising one.
"""

import numpy as np
import pandas as pd

MIN_GROUP = 1000  # smaller groups are reported but never used as the reference group
INCOME_BANDS = [0, 40_000, 60_000, 90_000, 150_000, np.inf]
INCOME_LABELS = ["< 40k", "40-60k", "60-90k", "90-150k", ">= 150k"]


def income_band(annual_inc: pd.Series) -> pd.Series:
    band = pd.cut(annual_inc, INCOME_BANDS, labels=INCOME_LABELS, right=False).astype(str)
    return band.where(annual_inc.notna(), "missing")


def employment_band(years: pd.Series) -> pd.Series:
    band = pd.cut(years, [-1, 0.5, 4.5, 9.5, np.inf], labels=["< 1", "1-4", "5-9", "10+"])
    return band.astype(str).where(years.notna(), "missing")


def attributes(df: pd.DataFrame) -> pd.DataFrame:
    """The audited attributes as plain strings, one column each. `df` has `zip_code`."""
    return pd.DataFrame(
        {
            "addr_state": df["addr_state"].astype(str),
            "home_ownership": df["home_ownership"].astype(str),
            "employment_length": employment_band(df["emp_length_years"]),
            "income_band": income_band(df["annual_inc"]),
            "zip3": df["zip_code"].astype(str),
        },
        index=df.index,
    )


def group_table(group: pd.Series, approve, y, pd_hat, min_group: int = MIN_GROUP) -> pd.DataFrame:
    """Approval, equal-opportunity and calibration metrics for each level of `group`."""
    frame = pd.DataFrame(
        {
            "group": group.to_numpy(),
            "approve": np.asarray(approve, dtype=bool),
            "y": np.asarray(y),
            "pd": np.asarray(pd_hat, dtype=float),
        }
    )
    frame["good_approved"] = frame["approve"] & (frame["y"] == 0)
    g = frame.groupby("group", observed=True)
    t = pd.DataFrame(
        {
            "n": g.size(),
            "default_rate": g["y"].mean(),
            "mean_pd": g["pd"].mean(),
            "approval_rate": g["approve"].mean(),
            "tpr": g["good_approved"].sum() / g["y"].apply(lambda s: (s == 0).sum()),
        }
    )
    t["share"] = t["n"] / t["n"].sum()
    t["citl"] = t["mean_pd"] - t["default_rate"]
    big = t[t["n"] >= min_group]
    ref = big if len(big) else t
    t["air"] = t["approval_rate"] / ref["approval_rate"].max()
    t["tpr_gap"] = t["tpr"] - ref["tpr"].max()
    t["reference"] = t.index == ref["approval_rate"].idxmax()
    return t.rename_axis("group").reset_index()


def summary(table: pd.DataFrame, attribute: str, min_group: int = MIN_GROUP) -> dict:
    """One line per attribute: the worst ratio and gaps among groups of at least min_group."""
    big = table[table["n"] >= min_group]
    worst = big.loc[big["air"].idxmin()]
    return {
        "attribute": attribute,
        "groups": len(table),
        "groups_used": len(big),
        "min_air": float(worst["air"]),
        "min_air_group": worst["group"],
        "groups_below_0_8": int((big["air"] < 0.8).sum()),
        "max_tpr_gap": float(big["tpr_gap"].min()),
        "max_abs_citl": float(big["citl"].abs().max()),
        "approval_spread": float(big["approval_rate"].max() - big["approval_rate"].min()),
        "default_rate_spread": float(big["default_rate"].max() - big["default_rate"].min()),
    }
