"""Integration tests against a real PostgreSQL (skipped when TEST_DATABASE_URL is unset)."""

import pandas as pd
import pytest
from sqlalchemy import inspect, text

from crisk.data.build import build
from crisk.data.ingest import csv_to_parquet
from crisk.db.loader import load, loan_rows

pytestmark = pytest.mark.postgres


def test_migration_creates_tables(migrated):
    assert {"loans", "vintages", "alembic_version"} <= set(inspect(migrated).get_table_names())


def test_load_is_idempotent(migrated, lc_csv, tmp_path):
    csv_to_parquet(lc_csv, tmp_path / "raw.parquet")
    report = build(tmp_path / "raw.parquet", tmp_path)
    loans = loan_rows(
        pd.read_parquet(tmp_path / "model_table.parquet"),
        pd.read_parquet(tmp_path / "outcomes.parquet"),
    )
    first = load(migrated, loans, report.vintages)
    second = load(migrated, loans, report.vintages)
    assert first == second
    assert first.loans == 8
    with migrated.connect() as conn:
        defaults = conn.execute(text("SELECT count(*) FROM loans WHERE is_default")).scalar_one()
        sources = conn.execute(text("SELECT DISTINCT source FROM loans")).scalars().all()
    assert defaults == 4
    assert sources == ["lending_club"]
