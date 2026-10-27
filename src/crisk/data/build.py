"""Raw parquet -> modelling table (application-time features + target + split) and outcomes.

Steps, each logged with the rows it removes:
1. resolved loans only: Fully Paid / Charged Off / Default (Current and Late are unresolved)
2. matured vintages only: issue month + term <= DATA_CUT, so every loan had its full term
   to default (otherwise recent vintages look artificially safe)
3. time split by issue date: train (oldest) / valid / test (most recent matured)

Processed in row-group batches with pyarrow-backed strings to stay well inside 8 GB of RAM.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from crisk.data.columns import Role, names

# Last month covered by the file: statuses and payments run to March 2019.
DATA_CUT = pd.Timestamp("2019-03-31")

# Loan status -> default label. Anything not listed here is an error, not a silent drop.
STATUS_TARGET: dict[str, int | None] = {
    "Fully Paid": 0,
    "Does not meet the credit policy. Status:Fully Paid": 0,
    "Charged Off": 1,
    "Default": 1,
    "Does not meet the credit policy. Status:Charged Off": 1,
    # Unresolved at the data cut: excluded.
    "Current": None,
    "In Grace Period": None,
    "Late (16-30 days)": None,
    "Late (31-120 days)": None,
}

# Time split on issue date (inclusive month starts). Chosen from the vintage table so each
# split has enough loans and defaults; see README "Time split".
SPLITS = {
    "train": (pd.Timestamp("2007-01-01"), pd.Timestamp("2014-06-01")),
    "valid": (pd.Timestamp("2014-07-01"), pd.Timestamp("2015-03-01")),
    "test": (pd.Timestamp("2015-04-01"), pd.Timestamp("2016-03-01")),
}

# Source column -> derived model column.
DERIVED = {
    "term": "term_months",
    "emp_length": "emp_length_years",
    "earliest_cr_line": "credit_history_months",
    "sec_app_earliest_cr_line": "sec_app_credit_history_months",
}
CATEGORICAL = [
    "grade",
    "sub_grade",
    "home_ownership",
    "verification_status",
    "purpose",
    "addr_state",
    "initial_list_status",
    "application_type",
    "verification_status_joint",
    "disbursement_method",
]
META = ["id", "issue_d", "issue_quarter", "split", "default"]
FEATURES = [DERIVED.get(c, c) for c in names(Role.FEATURE, Role.PRICING)]
MODEL_COLUMNS = META + FEATURES
OUTCOME_COLUMNS = [
    "id",
    "issue_d",
    "term_months",
    "loan_status",
    "default",
    *names(Role.EXPOSURE, Role.OUTCOME, Role.AUDIT),
]


@dataclass
class BuildReport:
    filters: list[dict] = field(default_factory=list)
    vintages: pd.DataFrame | None = None
    splits: pd.DataFrame | None = None
    by_grade: pd.DataFrame | None = None
    missingness: pd.DataFrame | None = None
    model_table_mb: float = 0.0


def _month(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype("string"), format="%b-%Y", errors="coerce")


def _months_between(start: pd.Series, end: pd.Series) -> pd.Series:
    months = (end.dt.year - start.dt.year) * 12 + (end.dt.month - start.dt.month)
    return months.astype("float32")


def _emp_years(s: pd.Series) -> pd.Series:
    years = s.str.replace("< 1 year", "0").str.extract(r"(\d+)")[0]
    return pd.to_numeric(years, errors="coerce").astype("float32")


def label(status: pd.Series) -> pd.Series:
    unknown = set(status.dropna().unique()) - set(STATUS_TARGET)
    if unknown or status.isna().any():
        raise ValueError(f"unmapped loan_status values: {sorted(unknown)} (or missing)")
    return status.map(STATUS_TARGET).astype("Float32").astype("Int8")


def assign_split(issue_d: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=issue_d.index, dtype="string")
    for name, (start, end) in SPLITS.items():
        out[(issue_d >= start) & (issue_d <= end)] = name
    return out


def transform(raw: pd.DataFrame) -> pd.DataFrame:
    """One batch of raw rows -> typed rows with target, maturity flag and derived columns."""
    df = raw.copy()
    df["id"] = pd.to_numeric(df["id"]).astype("int64")
    df["issue_d"] = _month(df["issue_d"])
    df["term_months"] = df["term"].str.extract(r"(\d+)")[0].astype("float32").astype("Int16")
    df["default"] = label(df["loan_status"])
    df["matured"] = _matured(df["issue_d"], df["term_months"])
    df["issue_quarter"] = df["issue_d"].dt.to_period("Q").astype("string")
    df["emp_length_years"] = _emp_years(df["emp_length"])
    df["credit_history_months"] = _months_between(_month(df["earliest_cr_line"]), df["issue_d"])
    df["sec_app_credit_history_months"] = _months_between(
        _month(df["sec_app_earliest_cr_line"]), df["issue_d"]
    )
    df["last_pymnt_d"] = _month(df["last_pymnt_d"])
    return df


def _matured(issue_d: pd.Series, term_months: pd.Series) -> pd.Series:
    """True when the loan's last scheduled payment month is on or before DATA_CUT."""
    end_month = issue_d.dt.year * 12 + issue_d.dt.month - 1 + term_months.astype("int64")
    cut_month = DATA_CUT.year * 12 + DATA_CUT.month - 1
    return (end_month <= cut_month).astype(bool)


def _arrow_strings(batch: pa.RecordBatch) -> pd.DataFrame:
    return batch.to_pandas(types_mapper={pa.string(): pd.StringDtype("pyarrow")}.get)


def build(raw_path: Path, out_dir: Path, batch_size: int = 200_000) -> BuildReport:
    report = BuildReport()
    source = pq.ParquetFile(raw_path)
    kept, outcomes, vintage_parts = [], [], []
    n_raw = n_resolved = n_matured = 0
    for batch in source.iter_batches(batch_size=batch_size):
        df = transform(_arrow_strings(batch))
        n_raw += len(df)
        resolved = df[df["default"].notna()]
        n_resolved += len(resolved)
        vintage_parts.append(
            resolved.groupby(["issue_quarter", "term_months", "matured"], observed=True)
            .agg(loans=("id", "size"), defaults=("default", "sum"))
            .reset_index()
        )
        matured = resolved[resolved["matured"]].copy()
        n_matured += len(matured)
        matured["split"] = assign_split(matured["issue_d"])
        matured = matured[matured["split"].notna()]
        kept.append(matured[MODEL_COLUMNS])
        outcomes.append(matured[OUTCOME_COLUMNS])

    model = pd.concat(kept, ignore_index=True)
    for col in [*CATEGORICAL, "split", "issue_quarter"]:
        model[col] = model[col].astype("category")
    model["default"] = model["default"].astype("int8")
    model["term_months"] = model["term_months"].astype("int16")
    model = model.sort_values(["issue_d", "id"], ignore_index=True)
    outcome = pd.concat(outcomes, ignore_index=True).sort_values(["issue_d", "id"])

    report.filters = [
        {"step": "raw loans", "rows": n_raw, "removed": 0},
        {"step": "resolved (paid off / charged off)", "rows": n_resolved},
        {"step": "matured vintage", "rows": n_matured},
        {"step": "inside a split window", "rows": len(model)},
    ]
    for prev, cur in zip(report.filters, report.filters[1:], strict=False):
        cur["removed"] = prev["rows"] - cur["rows"]

    v = pd.concat(vintage_parts).groupby(["issue_quarter", "term_months", "matured"]).sum()
    v = v.reset_index()
    v["default_rate"] = (v["defaults"] / v["loans"]).round(4)
    report.vintages = v.sort_values(["issue_quarter", "term_months"], ignore_index=True)
    report.splits = (
        model.groupby("split", observed=True)
        .agg(
            first_issue=("issue_d", "min"),
            last_issue=("issue_d", "max"),
            loans=("id", "size"),
            defaults=("default", "sum"),
            default_rate=("default", "mean"),
            share_60m=("term_months", lambda t: (t == 60).mean()),
        )
        .reindex(list(SPLITS))
        .round({"default_rate": 4, "share_60m": 4})
        .reset_index()
    )
    report.by_grade = (
        model.pivot_table(
            index="grade", columns="split", values="default", aggfunc="mean", observed=True
        )[list(SPLITS)]
        .round(4)
        .reset_index()
    )
    report.missingness = (
        model[FEATURES]
        .isna()
        .groupby(model["split"], observed=True)
        .mean()
        .T[list(SPLITS)]
        .round(4)
        .rename_axis("feature")
        .reset_index()
        .sort_values("train", ascending=False, ignore_index=True)
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    model.to_parquet(out_dir / "model_table.parquet", index=False)
    outcome.to_parquet(out_dir / "outcomes.parquet", index=False)
    report.model_table_mb = round(model.memory_usage(deep=True).sum() / 2**20, 1)
    return report


def write_reports(report: BuildReport, reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report.filters).to_csv(reports_dir / "filters.csv", index=False)
    report.vintages.to_csv(reports_dir / "vintages.csv", index=False)
    report.splits.to_csv(reports_dir / "splits.csv", index=False)
    report.by_grade.to_csv(reports_dir / "default_rate_by_grade.csv", index=False)
    report.missingness.to_csv(reports_dir / "missingness.csv", index=False)


def vintage_wide(vintages: pd.DataFrame) -> pd.DataFrame:
    """Default rate by issue quarter, one column per term, matured or not (for the README)."""
    v = vintages.copy()
    v["term"] = v["term_months"].astype(str) + "m" + np.where(v["matured"], "", " (immature)")
    return v.pivot_table(index="issue_quarter", columns="term", values="default_rate")
