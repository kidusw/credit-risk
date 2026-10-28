"""Phase 6: the SYNTHETIC mobile-money track (generator, features, pipeline, scoring)."""

import numpy as np
import pandas as pd
import pytest

from crisk.explain import reasons
from crisk.mobile_money import run as mm
from crisk.mobile_money.features import FEATURES, MONOTONE, build
from crisk.mobile_money.generate import HISTORY_DAYS, RULE, generate, rule_terms

N = 3000


@pytest.fixture(scope="module")
def book():
    return generate(n=N, seed=7)


@pytest.fixture(scope="module")
def features(book):
    return build(book.transactions, book.borrowers)


def test_generator_is_seeded(book):
    again = generate(n=N, seed=7)
    pd.testing.assert_frame_equal(book.borrowers, again.borrowers)
    assert len(book.transactions) == len(again.transactions)
    other = generate(n=N, seed=8)
    assert not book.borrowers["default"].equals(other.borrowers["default"])


def test_every_row_is_labelled_synthetic(book, features):
    for df in (book.borrowers, book.transactions, features):
        assert (df["source"] == "synthetic").all()


def test_history_is_before_the_application_and_never_negative(book):
    b, tx = book.borrowers.set_index("borrower_id"), book.transactions
    app = b.loc[tx["borrower_id"], "application_date"].to_numpy()
    assert (tx["date"].to_numpy() < app).all()
    assert (tx["date"].to_numpy() >= app - np.timedelta64(HISTORY_DAYS, "D")).all()
    assert (tx["balance"] >= 0).all()
    # The lowest balance, counting the opening one, is the borrower's buffer by construction.
    low = np.minimum(tx.groupby("borrower_id")["balance"].min(), b["opening_balance"])
    buffer = b["latent_buffer_months"] * b["latent_income"]
    np.testing.assert_allclose(low, buffer.loc[low.index], rtol=1e-3, atol=1)


def test_label_comes_from_the_documented_rule(book):
    b = book.borrowers
    assert set(rule_terms(b).columns) == set(RULE)
    assert 0.08 < b["default"].mean() < 0.2
    # Demographics are not in the rule: the rule terms explain the true PD on their own.
    assert not any(c in RULE for c in ("gender", "region", "urban_rural"))


def test_features_hold_nothing_hidden(features):
    assert list(features.columns) == ["borrower_id", *FEATURES, "source"]
    assert set(MONOTONE) <= set(FEATURES)
    hidden = [c for c in features if c.startswith(("latent_", "rule_")) or c == "true_pd"]
    assert not hidden
    assert not {"gender", "region", "urban_rural", "occupation", "default"} & set(features)


def test_recurring_detector_finds_salaries(book, features):
    f = features.merge(book.borrowers[["borrower_id", "occupation"]], on="borrower_id")
    reg = f.groupby("occupation")["income_regularity"].median()
    assert reg["salaried"] > 0.8
    assert reg["trader"] < 0.05 and reg["farmer"] < 0.05
    # Equb members are found from their regular contributions.
    found = f["equb_member"].to_numpy() == 1
    truth = book.borrowers.set_index("borrower_id").loc[f["borrower_id"], "latent_equb"]
    assert (found == truth.to_numpy()).mean() > 0.95


def test_every_feature_has_one_reason_and_one_driver_at_most():
    mapped = [f for r in mm.REASONS for f in r.features]
    assert sorted(mapped) == sorted(FEATURES)
    assert set(mm.FEATURE_DRIVER) <= set(FEATURES)
    assert set(mm.FEATURE_DRIVER.values()) == set(RULE)


@pytest.fixture(scope="module")
def result(book, features):
    return mm.run(features, book.borrowers, search=mm.SEARCH[:2], min_group=50)


def test_pipeline_end_to_end(result):
    m = result.metrics.set_index(["model", "split"])
    assert m.loc[("gbm", "test"), "auc"] > 0.6
    assert m.loc[("scorecard", "test"), "auc"] > 0.6
    assert result.champion in ("gbm", "scorecard")
    assert set(result.fairness_summary["attribute"]) == set(mm.ATTRIBUTES)
    shares = result.recovery[["true_share", "model_share"]].sum()
    np.testing.assert_allclose(shares, 1, atol=1e-6)
    # Every decline got at least one reason (reason_codes raises otherwise).
    assert result.reason_counts["n_reasons"].min() >= 1


def test_score_matches_the_policy(result, book, features):
    test_ids = book.borrowers.loc[book.borrowers["cohort_month"] >= 15, "borrower_id"]
    rows = features[features["borrower_id"].isin(test_ids)].head(200).reset_index(drop=True)
    out = mm.score(result.bundle, rows)
    assert (
        out["decision"] == np.where(out["pd"] <= result.bundle["cutoff"], "approve", "decline")
    ).all()
    declined = out[out["decision"] == "decline"]
    assert declined["reason_1"].notna().all()
    texts = {r.text for r in mm.REASONS}
    assert set(declined["reason_1"]) <= texts
    assert out.loc[out["decision"] == "approve", "reason_1"].isna().all()


def test_reason_codes_accept_a_custom_reason_list():
    phi = pd.DataFrame({"a": [0.5, -0.1], "b": [0.2, 0.3]})
    custom = [reasons.Reason("X1", "A", ("a",)), reasons.Reason("X2", "B", ("b",))]
    codes = reasons.reason_codes(phi, max_reasons=2, reasons=custom)
    assert codes["reason_1"].tolist() == ["X1", "X2"]
    assert reasons.describe(codes, custom)["reason_1"].tolist() == ["A", "B"]
