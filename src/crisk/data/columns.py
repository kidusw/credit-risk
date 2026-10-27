"""Leakage audit: every column of the Lending Club accepted-loans file, classified.

A model may only use what was known when the loan was approved. Anything that describes
how the loan behaved afterwards (payments, recoveries, hardship plans, refreshed FICO) leaks
the outcome and is kept out of the modelling table. `tests/test_columns.py` enforces this,
and `crisk data audit` prints the table used in the README.
"""

from dataclasses import dataclass
from enum import StrEnum


class Role(StrEnum):
    KEY = "key"  # identifier, used for joins only
    DATE = "date"  # issue date: defines vintages and the time split, never a feature
    TARGET = "target"  # source of the default label
    FEATURE = "feature"  # application-time borrower / bureau / loan data: model input
    PRICING = "pricing"  # Lending Club's own grade and price: application-time, flagged
    EXPOSURE = "exposure"  # amount actually issued: EAD for expected loss, not a feature
    OUTCOME = "outcome"  # post-origination: kept for LGD / profit only, never a feature
    AUDIT = "audit"  # kept for the fairness audit only, never a feature
    DROP = "drop"  # not read at all


@dataclass(frozen=True)
class Column:
    role: Role
    reason: str


_BUREAU = "credit bureau attribute pulled at application"
_JOINT = "co-borrower attribute from the joint application"
_PAYMENT = "post-origination: payment history after the loan was issued"
_HARDSHIP = "post-origination: hardship plan agreed after the borrower got into trouble"
_SETTLEMENT = "post-origination: debt settlement after charge-off"


def _many(role: Role, reason: str, *names: str) -> dict[str, Column]:
    return {name: Column(role, reason) for name in names}


# In the file's column order.
COLUMNS: dict[str, Column] = {
    "id": Column(Role.KEY, "loan identifier"),
    "member_id": Column(Role.DROP, "empty in the public file"),
    "loan_amnt": Column(Role.FEATURE, "amount the borrower applied for"),
    "funded_amnt": Column(Role.EXPOSURE, "amount issued; EAD for expected loss (Phase 4)"),
    "funded_amnt_inv": Column(
        Role.DROP, "investors' funding, decided after the application; near-copy of loan_amnt"
    ),
    "term": Column(Role.FEATURE, "36 or 60 months, chosen at application"),
    "int_rate": Column(Role.PRICING, "rate Lending Club set from its own grade"),
    "installment": Column(Role.PRICING, "monthly payment implied by amount, term and rate"),
    "grade": Column(Role.PRICING, "Lending Club's own risk grade at origination"),
    "sub_grade": Column(Role.PRICING, "Lending Club's own risk sub-grade at origination"),
    "emp_title": Column(Role.DROP, "free text job title, ~500k distinct values"),
    "emp_length": Column(Role.FEATURE, "self-reported employment length"),
    "home_ownership": Column(Role.FEATURE, "self-reported at application"),
    "annual_inc": Column(Role.FEATURE, "self-reported income at application"),
    "verification_status": Column(Role.FEATURE, "income verification done before issuance"),
    "issue_d": Column(Role.DATE, "month the loan was issued"),
    "loan_status": Column(Role.TARGET, "outcome at the data cut; becomes the default label"),
    "pymnt_plan": Column(Role.DROP, "post-origination: payment plan put in place later"),
    "url": Column(Role.DROP, "listing URL, an identifier"),
    "desc": Column(Role.DROP, "free text, empty for almost every loan after 2013"),
    "purpose": Column(Role.FEATURE, "loan purpose chosen at application"),
    "title": Column(Role.DROP, "free text loan title, duplicates purpose"),
    "zip_code": Column(
        Role.AUDIT, "3-digit zip is a known proxy for race; kept only for the fairness audit"
    ),
    "addr_state": Column(Role.FEATURE, "borrower's state at application"),
    "dti": Column(Role.FEATURE, "debt-to-income at application"),
    "delinq_2yrs": Column(Role.FEATURE, _BUREAU),
    "earliest_cr_line": Column(Role.FEATURE, "bureau date; becomes credit history length"),
    "fico_range_low": Column(Role.FEATURE, "FICO at application (lower end of the range)"),
    "fico_range_high": Column(Role.DROP, "fico_range_low + 4 (+5 for 441 loans): redundant"),
    **_many(
        Role.FEATURE,
        _BUREAU,
        "inq_last_6mths",
        "mths_since_last_delinq",
        "mths_since_last_record",
        "open_acc",
        "pub_rec",
        "revol_bal",
        "revol_util",
        "total_acc",
    ),
    "initial_list_status": Column(Role.FEATURE, "whole or fractional listing, set at listing"),
    **_many(
        Role.DROP,
        _PAYMENT,
        "out_prncp",
        "out_prncp_inv",
        "total_pymnt_inv",
        "total_rec_late_fee",
        "last_pymnt_amnt",
        "next_pymnt_d",
    ),
    **_many(
        Role.OUTCOME,
        _PAYMENT + "; kept for LGD and profit",
        "total_pymnt",
        "total_rec_prncp",
        "total_rec_int",
        "last_pymnt_d",
    ),
    **_many(
        Role.OUTCOME,
        "post-origination: recovered after charge-off; kept for LGD",
        "recoveries",
        "collection_recovery_fee",
    ),
    "last_credit_pull_d": Column(Role.DROP, "post-origination: date of the latest bureau pull"),
    **_many(
        Role.DROP,
        "post-origination: FICO refreshed after issuance, falls as borrowers default",
        "last_fico_range_high",
        "last_fico_range_low",
    ),
    "collections_12_mths_ex_med": Column(Role.FEATURE, _BUREAU),
    "mths_since_last_major_derog": Column(Role.FEATURE, _BUREAU),
    "policy_code": Column(Role.DROP, "constant (1) in this file"),
    "application_type": Column(Role.FEATURE, "individual or joint application"),
    **_many(
        Role.FEATURE,
        _JOINT,
        "annual_inc_joint",
        "dti_joint",
        "verification_status_joint",
    ),
    **_many(
        Role.FEATURE,
        _BUREAU,
        "acc_now_delinq",
        "tot_coll_amt",
        "tot_cur_bal",
        "open_acc_6m",
        "open_act_il",
        "open_il_12m",
        "open_il_24m",
        "mths_since_rcnt_il",
        "total_bal_il",
        "il_util",
        "open_rv_12m",
        "open_rv_24m",
        "max_bal_bc",
        "all_util",
        "total_rev_hi_lim",
        "inq_fi",
        "total_cu_tl",
        "inq_last_12m",
        "acc_open_past_24mths",
        "avg_cur_bal",
        "bc_open_to_buy",
        "bc_util",
        "chargeoff_within_12_mths",
        "delinq_amnt",
        "mo_sin_old_il_acct",
        "mo_sin_old_rev_tl_op",
        "mo_sin_rcnt_rev_tl_op",
        "mo_sin_rcnt_tl",
        "mort_acc",
        "mths_since_recent_bc",
        "mths_since_recent_bc_dlq",
        "mths_since_recent_inq",
        "mths_since_recent_revol_delinq",
        "num_accts_ever_120_pd",
        "num_actv_bc_tl",
        "num_actv_rev_tl",
        "num_bc_sats",
        "num_bc_tl",
        "num_il_tl",
        "num_op_rev_tl",
        "num_rev_accts",
        "num_rev_tl_bal_gt_0",
        "num_sats",
        "num_tl_120dpd_2m",
        "num_tl_30dpd",
        "num_tl_90g_dpd_24m",
        "num_tl_op_past_12m",
        "pct_tl_nvr_dlq",
        "percent_bc_gt_75",
        "pub_rec_bankruptcies",
        "tax_liens",
        "tot_hi_cred_lim",
        "total_bal_ex_mort",
        "total_bc_limit",
        "total_il_high_credit_limit",
    ),
    **_many(
        Role.FEATURE,
        _JOINT,
        "revol_bal_joint",
        "sec_app_fico_range_low",
    ),
    "sec_app_fico_range_high": Column(Role.DROP, "sec_app_fico_range_low + 4: redundant"),
    **_many(
        Role.FEATURE,
        _JOINT,
        "sec_app_earliest_cr_line",
        "sec_app_inq_last_6mths",
        "sec_app_mort_acc",
        "sec_app_open_acc",
        "sec_app_revol_util",
        "sec_app_open_act_il",
        "sec_app_num_rev_accts",
        "sec_app_chargeoff_within_12_mths",
        "sec_app_collections_12_mths_ex_med",
        "sec_app_mths_since_last_major_derog",
    ),
    **_many(
        Role.DROP,
        _HARDSHIP,
        "hardship_flag",
        "hardship_type",
        "hardship_reason",
        "hardship_status",
        "deferral_term",
        "hardship_amount",
        "hardship_start_date",
        "hardship_end_date",
        "payment_plan_start_date",
        "hardship_length",
        "hardship_dpd",
        "hardship_loan_status",
        "orig_projected_additional_accrued_interest",
        "hardship_payoff_balance_amount",
        "hardship_last_payment_amount",
    ),
    "disbursement_method": Column(Role.FEATURE, "cash or direct pay, chosen at issuance"),
    **_many(
        Role.DROP,
        _SETTLEMENT,
        "debt_settlement_flag",
        "debt_settlement_flag_date",
        "settlement_status",
        "settlement_date",
        "settlement_amount",
        "settlement_percentage",
        "settlement_term",
    ),
}

# Post-origination columns named in the plan; a test checks none reaches the model table.
KNOWN_LEAKS = tuple(
    name for name, col in COLUMNS.items() if col.reason.startswith("post-origination")
)

# Text columns (everything else is numeric).
STRING_COLUMNS = frozenset(
    {
        "id",
        "term",
        "grade",
        "sub_grade",
        "emp_length",
        "home_ownership",
        "verification_status",
        "issue_d",
        "loan_status",
        "purpose",
        "zip_code",
        "addr_state",
        "earliest_cr_line",
        "initial_list_status",
        "application_type",
        "verification_status_joint",
        "sec_app_earliest_cr_line",
        "disbursement_method",
        "last_pymnt_d",
    }
)


def names(*roles: Role) -> list[str]:
    return [name for name, col in COLUMNS.items() if col.role in roles]


# Columns read from the CSV into the raw parquet (everything not dropped).
READ_COLUMNS = [name for name, col in COLUMNS.items() if col.role is not Role.DROP]


def audit_markdown() -> str:
    """The leakage audit as a markdown table: every column that is not a model input."""
    lines = ["| Column | Role | Why it is not a model input |", "|---|---|---|"]
    for name, col in COLUMNS.items():
        if col.role not in (Role.FEATURE, Role.PRICING):
            lines.append(f"| `{name}` | {col.role} | {col.reason} |")
    return "\n".join(lines)
