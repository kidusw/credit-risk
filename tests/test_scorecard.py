import math

import numpy as np
import pandas as pd
import pytest

from crisk.data.columns import COLUMNS, Role
from crisk.models.features import PRICING, clean, feature_sets
from crisk.models.metrics import binary_metrics, grouped_metrics, ks_statistic
from crisk.models.scorecard import (
    BASE_ODDS,
    BASE_POINTS,
    FACTOR,
    OFFSET,
    PDO,
    Scorecard,
    score_to_pd,
)


@pytest.fixture(scope="module")
def applicants():
    """Known drivers: lower FICO and higher DTI raise risk, purpose matters, noise doesn't."""
    rng = np.random.default_rng(7)
    n = 6000
    fico = rng.normal(700, 30, n)
    dti = rng.uniform(0, 40, n)
    purpose = rng.choice(["debt_consolidation", "credit_card", "small_business"], n)
    logit = (
        -1.8
        - 0.03 * (fico - 700)
        + 0.04 * (dti - 20)
        + np.where(purpose == "small_business", 0.9, 0)
    )
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame(
        {
            "fico_range_low": fico.astype("float32"),
            "sec_app_fico_range_low": (fico + rng.normal(0, 3, n)).astype("float32"),  # copy
            "dti": dti.astype("float32"),
            "purpose": pd.Categorical(purpose),
            "tot_coll_amt": rng.uniform(0, 1, n).astype("float32"),  # noise
        }
    )
    return df, pd.Series(y)


@pytest.fixture(scope="module")
def card(applicants):
    df, y = applicants
    return Scorecard(list(df.columns)).fit(df, y)


def test_selects_true_drivers_and_drops_noise_and_duplicates(card):
    assert set(card.selected) == {"fico_range_low", "dti", "purpose"}
    assert card.dropped["tot_coll_amt"].startswith("IV")
    assert "corr" in card.dropped["sec_app_fico_range_low"]
    assert card.iv.loc["fico_range_low", "iv"] > card.iv.loc["tot_coll_amt", "iv"]


def test_bins_are_monotonic(card):
    table = card.points_table()
    fico = table[(table["feature"] == "fico_range_low") & (table["count"] > 0)]
    fico = fico[~fico["bin"].isin(["Special", "Missing"])]
    assert fico["default_rate"].is_monotonic_decreasing
    assert fico["points"].is_monotonic_increasing  # safer bins earn more points
    purpose_bins = table.loc[table["feature"] == "purpose", "bin"]
    assert any("small_business" in b and "Array" not in b for b in purpose_bins)


def test_points_scaling():
    assert OFFSET + FACTOR * math.log(BASE_ODDS) == pytest.approx(BASE_POINTS)
    assert OFFSET + FACTOR * math.log(2 * BASE_ODDS) == pytest.approx(BASE_POINTS + PDO)
    assert score_to_pd(BASE_POINTS) == pytest.approx(1 / (1 + BASE_ODDS))


def test_score_matches_model_pd(card, applicants):
    df, _ = applicants
    p = card.predict_pd(df)
    exact = OFFSET + FACTOR * np.log((1 - p) / p)
    # Each of k bins is rounded to whole points: at most k/2 points of drift.
    assert np.max(np.abs(card.score(df) - exact)) <= len(card.selected) / 2 + 1e-6
    assert np.corrcoef(card.score(df), np.log(p / (1 - p)))[0, 1] < -0.99  # linear in log-odds


def test_scorecard_discriminates(card, applicants):
    df, y = applicants
    assert binary_metrics(y, card.predict_pd(df))["auc"] > 0.75


def test_metrics():
    y = np.array([0, 0, 1, 1])
    assert ks_statistic(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    m = binary_metrics(y, np.array([0.1, 0.6, 0.4, 0.9]))
    assert m["auc"] == 0.75
    assert m["gini"] == pytest.approx(0.5)
    groups = pd.DataFrame({"g": ["a", "a", "b", "b"], "default": [0, 1, 1, 1], "pd": [0.1] * 4})
    assert list(grouped_metrics(groups, "g")["g"]) == ["a"]  # single-class group skipped


def test_feature_sets_and_cleaning():
    df = pd.DataFrame(
        {
            "dti": [-1.0, 15.0, 999.0],
            "loan_amnt": [10_000.0] * 3,
            "annual_inc": [50_000.0, 0.0, 100_000.0],
        }
    )
    out = clean(df)
    assert out["dti"].isna().tolist() == [True, False, True]
    assert out["loan_to_income"][0] == pytest.approx(0.2)
    assert np.isnan(out["loan_to_income"][1])
    for source in ("loan_amnt", "annual_inc"):  # derived ratio uses application data only
        assert COLUMNS[source].role is Role.FEATURE


def test_sub_grade_baseline_uses_train_rates():
    from crisk.models.train import SubGradeBaseline

    train = pd.DataFrame({"sub_grade": pd.Categorical(["A1", "A1", "C3", "C3"])})
    base = SubGradeBaseline().fit(train, pd.Series([0, 0, 1, 0]))
    new = pd.DataFrame({"sub_grade": pd.Categorical(["C3", "A1", "G5"])})
    assert base.predict_pd(new).tolist() == [0.5, 0.0, 0.25]  # unseen grade -> overall rate


def test_no_pricing_variant_excludes_lending_club_grades(lc_csv, tmp_path):
    from crisk.data.build import build
    from crisk.data.ingest import csv_to_parquet

    csv_to_parquet(lc_csv, tmp_path / "raw.parquet")
    build(tmp_path / "raw.parquet", tmp_path)
    train = clean(pd.read_parquet(tmp_path / "model_table.parquet"))
    sets = feature_sets(train)
    assert set(PRICING) <= set(sets["with_pricing"])
    assert not set(PRICING) & set(sets["no_pricing"])
    assert "sec_app_fico_range_low" not in sets["with_pricing"]  # empty in train
