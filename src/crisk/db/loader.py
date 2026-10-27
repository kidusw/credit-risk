"""Load the processed loans and vintage table into Postgres.

Replace-load inside one transaction (truncate + COPY), so re-running is idempotent and a
failed load leaves the previous data in place. COPY streams ~1.3M rows in batches.
"""

from dataclasses import dataclass

import pandas as pd
from sqlalchemy import Engine, text

from crisk.db.models import Loan, Vintage

LOAN_COLUMNS = [c.name for c in Loan.__table__.columns]
VINTAGE_COLUMNS = [c.name for c in Vintage.__table__.columns]


@dataclass
class LoadReport:
    loans: int
    vintages: int


def loan_rows(model: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Join the modelling table with outcomes into the `loans` table's columns."""
    df = model[
        [
            "id",
            "issue_d",
            "issue_quarter",
            "split",
            "term_months",
            "loan_amnt",
            "int_rate",
            "installment",
            "grade",
            "sub_grade",
            "purpose",
            "home_ownership",
            "addr_state",
            "annual_inc",
            "dti",
            "fico_range_low",
            "default",
        ]
    ].merge(
        outcomes[["id", "funded_amnt", "loan_status", "total_pymnt", "recoveries"]],
        on="id",
        validate="one_to_one",
    )
    df["source"] = "lending_club"
    df["is_default"] = df.pop("default").astype(bool)
    df["issue_d"] = df["issue_d"].dt.date
    for col in ["loan_amnt", "funded_amnt", "installment", "total_pymnt", "recoveries"]:
        df[col] = df[col].astype("float64").round(2)
    return df[LOAN_COLUMNS]


def _copy(conn, table: str, columns: list[str], df: pd.DataFrame, batch: int = 50_000) -> None:
    raw = conn.connection.dbapi_connection  # psycopg 3 connection
    cols = ", ".join(columns)
    with raw.cursor() as cur, cur.copy(f"COPY {table} ({cols}) FROM STDIN") as copy:
        for start in range(0, len(df), batch):
            part = df.iloc[start : start + batch].astype(object)
            for row in part.itertuples(index=False, name=None):
                copy.write_row([None if pd.isna(v) else v for v in row])


def load(engine: Engine, loans: pd.DataFrame, vintages: pd.DataFrame) -> LoadReport:
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE loans, vintages"))
        _copy(conn, "loans", LOAN_COLUMNS, loans)
        _copy(conn, "vintages", VINTAGE_COLUMNS, vintages[VINTAGE_COLUMNS])
        n_loans = conn.execute(text("SELECT count(*) FROM loans")).scalar_one()
        n_vintages = conn.execute(text("SELECT count(*) FROM vintages")).scalar_one()
    return LoadReport(loans=n_loans, vintages=n_vintages)
