"""The leakage audit, enforced."""

import pytest

from crisk.data.build import FEATURES, MODEL_COLUMNS, OUTCOME_COLUMNS
from crisk.data.columns import COLUMNS, KNOWN_LEAKS, STRING_COLUMNS, Role, audit_markdown, names

# Post-origination columns called out in the plan (prefix match).
PLAN_LEAKS = (
    "total_pymnt",
    "total_rec_",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d",
    "last_fico_",
    "last_credit_pull_d",
    "out_prncp",
    "hardship_",
    "debt_settlement_",
    "settlement_",
    "pymnt_plan",
    "funded_amnt_inv",
)


def test_every_source_column_is_classified_once():
    assert len(COLUMNS) == 151
    assert all(col.reason for col in COLUMNS.values())


@pytest.mark.parametrize("prefix", PLAN_LEAKS)
def test_plan_leaks_are_never_model_inputs(prefix):
    matches = [c for c in COLUMNS if c.startswith(prefix)]
    assert matches, f"no source column matches {prefix}"
    for name in matches:
        assert COLUMNS[name].role not in (Role.FEATURE, Role.PRICING), name


def test_model_table_has_no_post_origination_or_audit_columns():
    forbidden = set(KNOWN_LEAKS) | set(
        names(Role.OUTCOME, Role.DROP, Role.AUDIT, Role.EXPOSURE, Role.TARGET)
    )
    assert not forbidden & set(MODEL_COLUMNS)
    assert not forbidden & set(FEATURES)


def test_outcomes_never_feed_features():
    assert set(OUTCOME_COLUMNS) & set(FEATURES) == {"term_months"}  # join key for LGD by term


def test_string_columns_are_known():
    assert set(COLUMNS) >= STRING_COLUMNS


def test_audit_markdown_lists_every_non_input():
    md = audit_markdown()
    for name in names(Role.OUTCOME, Role.DROP, Role.AUDIT, Role.EXPOSURE):
        assert f"`{name}`" in md
    assert "`dti`" not in md
