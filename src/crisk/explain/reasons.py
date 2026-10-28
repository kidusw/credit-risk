"""Adverse-action reason codes: the top reasons a declined applicant's PD is high.

Each feature belongs to one reason, as on a bank's adverse-action notice: inquiries in the
last 6 months, months since the latest inquiry and inquiries in the last 12 months are all
"too many recent credit inquiries". A reason's weight for one applicant is the sum of its
features' SHAP values, which is exact because SHAP is additive. Only reasons that pushed
the PD *up* (weight > 0) can be stated, the largest first, at most four.

The base value is the average applicant in train. A declined applicant's PD is above the
cutoff, the cutoff is above the average PD, so the weights sum to more than zero and at least
one reason always exists; `reason_codes` checks this rather than assuming it.

Two kinds of reason are flagged `actionable=False`. "Lending Club's risk grade" is the
lender's own model output, so citing it doesn't tell the applicant anything. "Listing and
application type" is about how the loan was sold, not the borrower. They stay in the output,
because an adverse-action notice must give the real principal reasons, but the README counts
how often they come first.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

MAX_REASONS = 4


@dataclass(frozen=True)
class Reason:
    code: str
    text: str
    features: tuple[str, ...]
    actionable: bool = True


REASONS = [
    Reason("R01", "Credit score too low", ("fico_range_low", "sec_app_fico_range_low")),
    Reason(
        "R02",
        "Delinquency on credit accounts",
        (
            "delinq_2yrs",
            "mths_since_last_delinq",
            "acc_now_delinq",
            "delinq_amnt",
            "num_accts_ever_120_pd",
            "num_tl_120dpd_2m",
            "num_tl_30dpd",
            "num_tl_90g_dpd_24m",
            "pct_tl_nvr_dlq",
            "mths_since_recent_bc_dlq",
            "mths_since_recent_revol_delinq",
            "mths_since_last_major_derog",
            "sec_app_mths_since_last_major_derog",
        ),
    ),
    Reason(
        "R03",
        "Public records, collections or charge-offs on file",
        (
            "pub_rec",
            "mths_since_last_record",
            "pub_rec_bankruptcies",
            "tax_liens",
            "collections_12_mths_ex_med",
            "chargeoff_within_12_mths",
            "tot_coll_amt",
            "sec_app_chargeoff_within_12_mths",
            "sec_app_collections_12_mths_ex_med",
        ),
    ),
    Reason("R04", "Debt-to-income ratio too high", ("dti", "dti_joint")),
    Reason(
        "R05",
        "Loan amount too high for income",
        ("loan_amnt", "loan_to_income", "installment"),
    ),
    Reason(
        "R06",
        "Income too low or not verified",
        ("annual_inc", "annual_inc_joint", "verification_status", "verification_status_joint"),
    ),
    Reason(
        "R07",
        "Revolving credit utilisation too high",
        (
            "revol_util",
            "bc_util",
            "percent_bc_gt_75",
            "bc_open_to_buy",
            "all_util",
            "il_util",
            "revol_bal",
            "revol_bal_joint",
            "max_bal_bc",
            "sec_app_revol_util",
        ),
    ),
    Reason(
        "R08",
        "Too many recent credit inquiries",
        (
            "inq_last_6mths",
            "mths_since_recent_inq",
            "inq_fi",
            "inq_last_12m",
            "sec_app_inq_last_6mths",
        ),
    ),
    Reason(
        "R09",
        "Too many accounts opened recently",
        (
            "acc_open_past_24mths",
            "num_tl_op_past_12m",
            "mo_sin_rcnt_tl",
            "mo_sin_rcnt_rev_tl_op",
            "mths_since_recent_bc",
            "mths_since_rcnt_il",
            "open_acc_6m",
            "open_il_12m",
            "open_il_24m",
            "open_rv_12m",
            "open_rv_24m",
        ),
    ),
    Reason(
        "R10",
        "Length of credit history too short",
        (
            "credit_history_months",
            "mo_sin_old_il_acct",
            "mo_sin_old_rev_tl_op",
            "sec_app_credit_history_months",
        ),
    ),
    Reason(
        "R11",
        "Number or mix of credit accounts",
        (
            "open_acc",
            "total_acc",
            "num_actv_bc_tl",
            "num_actv_rev_tl",
            "num_bc_sats",
            "num_bc_tl",
            "num_il_tl",
            "num_op_rev_tl",
            "num_rev_accts",
            "num_rev_tl_bal_gt_0",
            "num_sats",
            "open_act_il",
            "sec_app_open_acc",
            "sec_app_open_act_il",
            "sec_app_num_rev_accts",
        ),
    ),
    Reason(
        "R12",
        "Level of existing balances and credit limits",
        (
            "tot_cur_bal",
            "avg_cur_bal",
            "total_bal_ex_mort",
            "total_bal_il",
            "tot_hi_cred_lim",
            "total_rev_hi_lim",
            "total_bc_limit",
            "total_il_high_credit_limit",
            "total_cu_tl",
        ),
    ),
    Reason(
        "R13", "No mortgage or home ownership", ("home_ownership", "mort_acc", "sec_app_mort_acc")
    ),
    Reason("R14", "Length of employment too short", ("emp_length_years",)),
    Reason("R15", "Loan term too long", ("term_months",)),
    Reason("R16", "Loan purpose", ("purpose",)),
    Reason("R17", "State of residence", ("addr_state",)),
    Reason(
        "R18",
        "Lending Club's risk grade and interest rate for the loan",
        ("grade", "sub_grade", "int_rate"),
        actionable=False,
    ),
    Reason(
        "R19",
        "Listing and application type",
        ("initial_list_status", "application_type", "disbursement_method"),
        actionable=False,
    ),
]
BY_CODE = {r.code: r for r in REASONS}
FEATURE_REASON = {f: r.code for r in REASONS for f in r.features}


def feature_reason(reasons: list[Reason]) -> dict[str, str]:
    return {f: r.code for r in reasons for f in r.features}


def code_map(reasons: list[Reason] = REASONS) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "code": r.code,
                "text": r.text,
                "actionable": r.actionable,
                "features": " ".join(r.features),
            }
            for r in reasons
        ]
    )


def grouped(phi: pd.DataFrame, reasons: list[Reason] = REASONS) -> pd.DataFrame:
    """Per-loan reason weights: the sum of each reason's features' SHAP values."""
    mapping = feature_reason(reasons)
    unmapped = sorted(set(phi.columns) - set(mapping))
    if unmapped:
        raise KeyError(f"features without a reason code: {unmapped}")
    return phi.T.groupby(phi.columns.map(mapping)).sum().T


def reason_codes(
    phi: pd.DataFrame, max_reasons: int = MAX_REASONS, reasons: list[Reason] = REASONS
) -> pd.DataFrame:
    """Top reasons (largest positive weight first) for every row of `phi`.

    Returns one row per loan: reason_1..k codes and weights, plus `n_reasons`. Call it on
    declined applicants; it raises if any row has no reason pushing its PD up.
    """
    weights = grouped(phi, reasons)
    w = weights.to_numpy()
    order = np.argsort(-w, axis=1, kind="stable")[:, :max_reasons]
    top = np.take_along_axis(w, order, axis=1)
    codes = weights.columns.to_numpy()[order]
    codes = np.where(top > 0, codes, None)
    n_reasons = (top > 0).sum(axis=1)
    if (n_reasons == 0).any():
        raise ValueError(f"{int((n_reasons == 0).sum())} rows have no adverse reason")
    out = pd.DataFrame(index=phi.index)
    for k in range(max_reasons):  # fewer reasons than ranks (small models): pad with None
        has = k < top.shape[1]
        out[f"reason_{k + 1}"] = codes[:, k] if has else None
        out[f"weight_{k + 1}"] = np.where(top[:, k] > 0, top[:, k], np.nan) if has else np.nan
    out["n_reasons"] = n_reasons
    return out


def describe(codes: pd.DataFrame, reasons: list[Reason] = REASONS) -> pd.DataFrame:
    """Reason codes as plain-language text, one column per rank."""
    by_code = {r.code: r for r in reasons}
    out = pd.DataFrame(index=codes.index)
    for col in [c for c in codes.columns if c.startswith("reason_")]:
        out[col] = codes[col].map(lambda c: by_code[c].text if pd.notna(c) else None)
    return out


def frequency(codes: pd.DataFrame, reasons: list[Reason] = REASONS) -> pd.DataFrame:
    """How often each reason is given first, and anywhere in the top reasons."""
    n = len(codes)
    rank_cols = [c for c in codes.columns if c.startswith("reason_")]
    first = codes["reason_1"].value_counts()
    anywhere = pd.Series(codes[rank_cols].to_numpy().ravel()).dropna().value_counts()
    table = pd.DataFrame(
        [
            {
                "code": r.code,
                "text": r.text,
                "actionable": r.actionable,
                "first_reason_share": first.get(r.code, 0) / n,
                "in_top_reasons_share": anywhere.get(r.code, 0) / n,
            }
            for r in reasons
        ]
    )
    return table.sort_values("first_reason_share", ascending=False, ignore_index=True)
