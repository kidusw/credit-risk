"""Pandera schema for the modelling table.

`strict=True`: a column that is not declared here fails validation, so a post-origination
column cannot slip into the table without someone adding it to the schema (and the audit).
"""

import pandera.pandas as pa

from crisk.data.build import CATEGORICAL, DATA_CUT, FEATURES, SPLITS

_RANGES = {
    "loan_amnt": (500, 40_000),
    "int_rate": (0, 40),
    "installment": (0, 2_000),
    "fico_range_low": (300, 850),
    "sec_app_fico_range_low": (300, 850),
    "annual_inc": (0, 2e8),  # self-reported; a handful of rows claim > $10M
    "dti": (-1, 999),  # -1 and 999 are sentinel values in the source; cleaned in Phase 2
    "revol_util": (0, 1_000),
    "emp_length_years": (0, 10),
}
_INTEGER = {"term_months"}
_GRADES = list("ABCDEFG")


def _feature(name: str) -> pa.Column:
    if name in _INTEGER:
        return pa.Column("int16", pa.Check.isin([36, 60]))
    if name in CATEGORICAL:
        checks = [pa.Check.isin(_GRADES)] if name == "grade" else []
        return pa.Column("category", checks, nullable=True)
    checks = []
    if name in _RANGES:
        low, high = _RANGES[name]
        checks.append(pa.Check.in_range(low, high))
    # A few bureau fields are missing for every loan before 2012-2016; that is expected.
    return pa.Column("float32", checks, nullable=name != "loan_amnt")


MODEL_TABLE = pa.DataFrameSchema(
    {
        "id": pa.Column("int64", unique=True),
        "issue_d": pa.Column("datetime64[ns]", pa.Check.in_range(SPLITS["train"][0], DATA_CUT)),
        "issue_quarter": pa.Column("category"),
        "split": pa.Column("category", pa.Check.isin(list(SPLITS))),
        "default": pa.Column("int8", pa.Check.isin([0, 1])),
        **{name: _feature(name) for name in FEATURES},
    },
    strict=True,
    ordered=True,
    coerce=False,
)


def validate(model_table):
    return MODEL_TABLE.validate(model_table, lazy=True)
