import numpy as np
import pandas as pd
import pytest

from crisk.decision.economics import assign_lgd, expected_loss, lgd_table, loan_economics
from crisk.decision.policy import (
    best_cutoff,
    evaluate,
    expected_profit_rule,
    grade_policy,
    sweep,
)
from crisk.models.calibration import (
    METHODS,
    Calibrator,
    calibration_metrics,
    reliability_table,
    select_method,
)


@pytest.fixture(scope="module")
def drifted():
    """True PDs known; the model's PDs are systematically 30% too low (like test drift)."""
    rng = np.random.default_rng(1)
    true_pd = rng.beta(2, 11, 40_000)
    y = (rng.uniform(size=true_pd.size) < true_pd).astype(int)
    return true_pd * 0.7, true_pd, y


def test_calibration_fixes_a_level_shift(drifted):
    model_pd, _, y = drifted
    before = calibration_metrics(y, model_pd)
    for method in METHODS:
        after = calibration_metrics(y, Calibrator(method).fit(model_pd, y).transform(model_pd))
        assert abs(after["citl"]) < 0.002 < abs(before["citl"])
        assert after["ece"] < before["ece"]


def test_metrics_of_true_probabilities_look_calibrated(drifted):
    _, true_pd, y = drifted
    m = calibration_metrics(y, true_pd)
    assert m["slope"] == pytest.approx(1.0, abs=0.1)
    assert m["intercept"] == pytest.approx(0.0, abs=0.25)
    assert m["ece"] < 0.01
    rel = reliability_table(y, true_pd)
    assert len(rel) == 10 and rel["mean_pd"].is_monotonic_increasing


def test_method_selection_uses_out_of_fold_scores(drifted):
    model_pd, _, y = drifted
    method, table = select_method(model_pd, y)
    assert method in METHODS
    assert set(table.index) == set(METHODS)
    assert method == table["brier"].idxmin()


def test_isotonic_is_monotone(drifted):
    model_pd, _, y = drifted
    grid = np.linspace(0.01, 0.6, 200)
    assert np.all(np.diff(Calibrator("isotonic").fit(model_pd, y).transform(grid)) >= 0)


@pytest.fixture
def book():
    """Four 36-month loans: two paid off, two charged off (one with recoveries)."""
    outcomes = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "default": [0, 0, 1, 1],
            "funded_amnt": [10_000.0, 20_000.0, 10_000.0, 10_000.0],
            "total_pymnt": [11_500.0, 22_000.0, 4_000.0, 3_000.0],
            "total_rec_prncp": [10_000.0, 20_000.0, 3_000.0, 2_000.0],
            "recoveries": [0.0, 0.0, 500.0, 0.0],
            "collection_recovery_fee": [0.0, 0.0, 100.0, 0.0],
        }
    )
    econ = loan_economics(outcomes)
    return econ.assign(
        default=outcomes["default"],
        grade=pd.Categorical(["A", "B", "C", "D"]),
        sub_grade=pd.Categorical(["A1", "B1", "C1", "D1"]),
        term_months=36,
    )


def test_loan_economics(book):
    assert book["profit"].tolist() == [1_500.0, 2_000.0, -6_100.0, -7_000.0]
    # Loan 3: 10k lent, 3k principal repaid, 400 net recovered -> 6.6k principal lost.
    assert book.loc[2, "principal_loss"] == pytest.approx(6_600.0)
    assert book.loc[2, "lgd"] == pytest.approx(0.66)
    assert book.loc[0, "principal_loss"] == 0 and np.isnan(book.loc[0, "lgd"])


def test_lgd_assignment_and_expected_loss(book):
    table = lgd_table(book)
    lgd = assign_lgd(book.assign(grade=pd.Categorical(["C", "A", "D", "D"])), table, 0.5)
    assert lgd.tolist() == pytest.approx([0.66, 0.5, 0.8, 0.8])  # A unseen -> fallback
    assert expected_loss([0.1], [0.5], [10_000])[0] == pytest.approx(500.0)


def test_profit_curve_and_cutoffs(book):
    pd_hat = np.array([0.05, 0.10, 0.30, 0.20])  # model ranks the two bads last
    curve = sweep(pd_hat, book, rates=[0.25, 0.5, 0.75, 1.0])
    assert curve["profit"].tolist() == [1_500.0, 3_500.0, -3_500.0, -9_600.0]
    assert curve["bad_rate"].tolist() == [0.0, 0.0, 1 / 3, 0.5]
    assert best_cutoff(curve, "profit") == pytest.approx(0.10)
    assert best_cutoff(curve, "return") == pytest.approx(0.05)  # 15% vs 11.7%
    r = evaluate(pd_hat <= 0.10, book)
    assert r["loans"] == 2 and r["return_on_lent"] == pytest.approx(3_500 / 30_000)


def test_grade_policy_and_expected_profit_rule(book):
    grades = grade_policy(book).set_index("worst_grade")
    assert grades.loc["B", "profit"] == 3_500.0 and grades.loc["G", "approval_rate"] == 1.0
    margins = pd.DataFrame(
        {"good_return": [0.15, 0.10, 0.12, 0.20], "bad_loss": [0.40, 0.40, 0.60, 0.70]},
        index=["A1", "B1", "C1", "D1"],
    )
    # A1: 0.9*0.15 - 0.1*0.40 > 0; C1 at PD 0.3: 0.7*0.12 - 0.3*0.6 < 0.
    approve = expected_profit_rule([0.1, 0.1, 0.3, 0.1], book, margins)
    assert approve.tolist() == [True, True, False, True]
