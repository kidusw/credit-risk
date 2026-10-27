import pandas as pd
import pandera.errors
import pytest

from crisk.data.build import MODEL_COLUMNS, build, label, write_reports
from crisk.data.columns import KNOWN_LEAKS
from crisk.data.ingest import SourceSchemaError, check_header, csv_to_parquet, read_header
from crisk.data.schema import validate
from tests.conftest import LOANS


@pytest.fixture
def built(lc_csv, tmp_path):
    raw = tmp_path / "interim" / "accepted.parquet"
    ingest = csv_to_parquet(lc_csv, raw, chunksize=5)
    report = build(raw, tmp_path / "processed", batch_size=4)
    model = pd.read_parquet(tmp_path / "processed" / "model_table.parquet")
    outcomes = pd.read_parquet(tmp_path / "processed" / "outcomes.parquet")
    return ingest, report, model, outcomes


def test_ingest_streams_in_chunks_and_drops_footer(built):
    ingest, *_ = built
    assert ingest.rows_written == len(LOANS)
    assert ingest.footer_rows == 2
    assert ingest.chunks == 3


def test_header_change_fails_loudly(lc_csv):
    header = read_header(lc_csv)
    check_header(header)
    with pytest.raises(SourceSchemaError, match="new_column"):
        check_header([*header, "new_column"])


def test_model_table_has_exactly_the_audited_columns(built):
    *_, model, _ = built
    assert list(model.columns) == MODEL_COLUMNS
    assert not set(KNOWN_LEAKS) & set(model.columns)


def test_only_resolved_matured_loans_are_kept_in_their_split(built):
    _, report, model, outcomes = built
    expected = {1000 + i: split for i, (*_, split) in enumerate(LOANS)}
    kept = dict(zip(model["id"], model["split"], strict=True))
    assert kept == {i: s for i, s in expected.items() if s in ("train", "valid", "test")}
    assert set(outcomes["id"]) == set(kept)
    assert [f["rows"] for f in report.filters] == [12, 10, 8, 8]


def test_target(built):
    *_, model, _ = built
    by_id = model.set_index("id")["default"]
    assert by_id[1001] == 1 and by_id[1002] == 1 and by_id[1006] == 1  # off, policy, Default
    assert by_id[1000] == 0
    with pytest.raises(ValueError, match="Paid Off"):
        label(pd.Series(["Fully Paid", "Paid Off"]))


def test_derived_features(built):
    *_, model, _ = built
    row = model.set_index("id").loc[1000]
    assert row["term_months"] == 36
    assert row["credit_history_months"] > 0


def test_vintage_table_includes_immature_loans(built):
    _, report, *_ = built
    v = report.vintages
    assert not v.loc[v["issue_quarter"] == "2016Q2", "matured"].item()
    assert v["loans"].sum() == 10


def test_schema_accepts_model_table_and_rejects_leaks(built):
    *_, model, outcomes = built
    validate(model)
    leaked = model.assign(total_pymnt=outcomes["total_pymnt"].to_numpy())
    with pytest.raises(pandera.errors.SchemaErrors):
        validate(leaked)


def test_reports_written(built, tmp_path):
    _, report, *_ = built
    write_reports(report, tmp_path / "reports")
    assert {p.name for p in (tmp_path / "reports").iterdir()} == {
        "filters.csv",
        "vintages.csv",
        "splits.csv",
        "default_rate_by_grade.csv",
        "missingness.csv",
    }
