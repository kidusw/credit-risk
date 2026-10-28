"""Phase 7: PSI, the characteristic stability report, the rejected file and reject inference."""

import numpy as np
import pandas as pd
import pytest

from crisk.models.scorecard import Scorecard
from crisk.monitor import reject
from crisk.monitor import stability as st
from crisk.monitor.rejected import SCHEMA, transform


def test_psi_is_zero_for_the_same_distribution_and_grows_with_a_shift():
    rng = np.random.default_rng(0)
    base = pd.Series(rng.normal(0, 1, 50_000))
    assert st.psi(base, pd.Series(rng.normal(0, 1, 50_000))) < 0.01
    small = st.psi(base, pd.Series(rng.normal(0.25, 1, 50_000)))
    large = st.psi(base, pd.Series(rng.normal(1.0, 1, 50_000)))
    assert 0.03 < small < 0.1 < 0.25 < large
    assert st.band(small) == "stable" and st.band(0.2) == "monitor" and st.band(large) == "act"


def test_psi_handles_missing_categories_and_few_values():
    e = pd.Series(["A"] * 50 + ["B"] * 50)
    a = pd.Series(["A"] * 50 + ["C"] * 50)  # a level train never saw
    assert st.psi(e, a, categorical=True) > 1
    # Missing values are a bin of their own: a jump in missingness shows up.
    num = pd.Series(np.arange(1000, dtype=float))
    holes = num.copy()
    holes[:300] = np.nan
    assert st.psi(num, holes) > 0.25
    # Two-valued numeric (term): binned by value, not by collapsed quantile edges.
    term_e = pd.Series([36.0] * 80 + [60.0] * 20)
    term_a = pd.Series([36.0] * 100)
    assert st.psi(term_e, term_a) > 1


def test_psi_by_period_uses_the_reference_rows():
    df = pd.DataFrame({"q": ["a"] * 500 + ["b"] * 500, "x": np.r_[np.zeros(500), np.ones(500)]})
    out = st.psi_by_period(df, ["x"], period="q", reference=df["q"] == "a")
    assert out.loc[out["q"] == "a", "x"].item() == pytest.approx(0)
    assert out.loc[out["q"] == "b", "x"].item() > 1


def test_characteristic_report_adds_up_to_the_score_shift():
    rng = np.random.default_rng(1)
    n = 6000
    df = pd.DataFrame({"fico_range_low": rng.normal(700, 30, n), "dti": rng.uniform(0, 40, n)})
    z = -2 - 0.03 * (df["fico_range_low"] - 700) + 0.04 * (df["dti"] - 20)
    df["default"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    card = Scorecard(["fico_range_low", "dti"]).fit(df, df["default"])
    points = st.characteristic_points(card, df)
    # Per-characteristic points add up to the scorecard's (rounded) score.
    assert np.abs(points.sum(axis=1) - card.score(df)).max() <= len(card.selected) / 2 + 1e-9
    later = df["dti"] > 20
    rep = st.characteristic_report(points, ~later, later)
    total = points[later].sum(axis=1).mean() - points[~later].sum(axis=1).mean()
    assert rep["shift"].sum() == pytest.approx(total)
    assert rep.iloc[0]["characteristic"] == "dti"  # the characteristic that moved


def test_rejected_chunk_transform():
    raw = pd.DataFrame(
        {
            "Amount Requested": ["1000.0", "5000", "2000"],
            "Application Date": ["2013-05-26", "2015-07-01", "2017-01-03"],
            "Loan Title": ["x", "y", "z"],
            "Risk_Score": ["693.0", None, "650"],
            "Debt-To-Income Ratio": ["10%", "38.64%", "-1%"],
            "Zip Code": ["481xx", "010xx", "212xx"],
            "State": ["NM", "MA", "MD"],
            "Employment Length": ["4 years", "< 1 year", "10+ years"],
            "Policy Code": ["0", "0", "0"],
        }
    )
    out = transform(raw.drop(columns=["Loan Title", "Policy Code"]))
    assert list(out.columns) == SCHEMA.names
    assert out["split"].tolist() == ["train", "test"]  # 2017 is outside every window
    assert out["dti"].tolist() == pytest.approx([10.0, 38.64])
    assert out["emp_length_years"].tolist() == [4.0, 0.0]
    assert np.isnan(out["fico_range_low"].iloc[1])


def test_policy_rejects_are_out_of_range():
    r = pd.DataFrame(
        {
            "fico_range_low": [700, 640, np.nan, 700, 700],
            "dti": [20, 20, 20, 55, 20],
            "loan_amnt": [10_000, 10_000, 10_000, 10_000, 100_000],
        }
    )
    assert reject.in_range(r).tolist() == [True, False, False, False, False]


def test_fuzzy_and_parcel_rows():
    rej = pd.DataFrame(
        {"fico_range_low": [700.0, 680.0], "dti": [10.0, 30.0], "loan_amnt": [5e3, 2e4]}
    )
    p = np.array([0.1, 0.4])
    fz = reject.fuzzy_rows(rej, p, weight=0.5)
    assert len(fz) == 4
    # Each reject carries its full weight, split between a bad and a good copy.
    assert fz["w"].sum() == pytest.approx(1.0)
    assert fz.loc[fz["default"] == 1, "w"].tolist() == pytest.approx([0.05, 0.2])

    rng = np.random.default_rng(0)
    acc_p = rng.uniform(0, 1, 5000)
    acc_y = (rng.random(5000) < acc_p * 0.3).astype(int)
    big = pd.concat([rej] * 5000, ignore_index=True)
    labelled, bands = reject.parcel_rows(big, np.tile(p, 5000), 1.0, acc_p, acc_y, rng)
    assert (bands["parcel_bad_rate"] <= 0.95).all()
    np.testing.assert_allclose(
        bands["parcel_bad_rate"],
        np.minimum(0.95, reject.PARCEL_FACTOR * bands["accepted_bad_rate"]),
    )
    # Rejects with the higher KGB PD are labelled bad more often.
    rates = labelled.groupby(np.tile([0, 1], 5000))["default"].mean()
    assert rates[1] > rates[0]


def test_approve_share_respects_weights():
    p = np.array([0.1, 0.2, 0.3, 0.4])
    w = np.array([1.0, 1.0, 2.0, 1.0])
    appr = reject._approve_share(p, w, 0.4)
    assert appr.tolist() == [True, True, False, False]
