import numpy as np
import pandas as pd
import pytest

from crisk.data.build import FEATURES
from crisk.data.columns import KNOWN_LEAKS
from crisk.models.compare import auc_lift_ci
from crisk.models.features import DERIVED
from crisk.models.gbm import MONOTONE, SUB_GRADES, GBMModel, prepare, sample_configs


def test_constraints_only_on_audited_features():
    assert set(MONOTONE) <= set(FEATURES) | set(DERIVED)
    assert not set(MONOTONE) & set(KNOWN_LEAKS)
    assert set(MONOTONE.values()) == {-1, 1}


def test_grades_become_ordinals():
    df = pd.DataFrame(
        {"grade": pd.Categorical(["A", "G"]), "sub_grade": pd.Categorical(["A1", "G5"])}
    )
    X = prepare(df, ["grade", "sub_grade"])
    assert X["grade"].tolist() == [0, 6]
    assert X["sub_grade"].tolist() == [0, len(SUB_GRADES) - 1]


def test_search_is_seeded_and_unique():
    assert sample_configs(8) == sample_configs(8)
    assert len({tuple(c.values()) for c in sample_configs(8)}) == 8


@pytest.fixture(scope="module")
def loans():
    """FICO risk is monotone but noisy enough that an unconstrained tree would wiggle."""
    rng = np.random.default_rng(3)
    n = 8000
    fico = rng.normal(700, 35, n)
    dti = rng.uniform(0, 40, n)
    purpose = rng.choice(["car", "credit_card", "small_business"], n)
    logit = -1.7 - 0.025 * (fico - 700) + 0.03 * (dti - 20) + (purpose == "small_business")
    df = pd.DataFrame(
        {
            "fico_range_low": fico,
            "dti": dti,
            "purpose": pd.Categorical(purpose),
            "default": (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int),
        }
    )
    return df.iloc[:6000], df.iloc[6000:]


def test_monotone_constraints_hold(loans):
    train, valid = loans
    model = GBMModel(["fico_range_low", "dti", "purpose"], {"num_leaves": 31}).fit(train, valid)
    grid = pd.DataFrame(
        {
            "fico_range_low": np.linspace(600, 820, 60),
            "dti": 20.0,
            "purpose": pd.Categorical(
                ["car"] * 60, categories=["car", "credit_card", "small_business"]
            ),
        }
    )
    pd_fico = model.predict_pd(grid)
    assert np.all(np.diff(pd_fico) <= 1e-12)  # higher FICO never raises PD
    grid = grid.assign(fico_range_low=700.0, dti=np.linspace(0, 40, 60))
    assert np.all(np.diff(model.predict_pd(grid)) >= -1e-12)  # higher DTI never lowers it
    imp = model.importance()
    assert set(imp["feature"]) == {"fico_range_low", "dti", "purpose"}
    assert imp["gain_share"].sum() == pytest.approx(1.0)


def test_auc_lift_ci():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    good = y + rng.normal(0, 0.8, 2000)
    weak = y + rng.normal(0, 2.0, 2000)
    same = auc_lift_ci(y, weak, weak, n_boot=50)
    assert same["lift"] == 0 and same["ci_low"] == 0 and same["ci_high"] == 0
    better = auc_lift_ci(y, good, weak, n_boot=100)
    assert better["lift"] > 0 and better["ci_low"] > 0
    assert better["ci_low"] <= better["lift"] <= better["ci_high"]


def test_auc_by_vintage_chart(tmp_path):
    from crisk.models.plots import SERIES, auc_by_vintage

    quarters = ["2014Q3", "2014Q4", "2015Q1", "2015Q2"]
    frame = pd.DataFrame(
        [
            {"issue_quarter": q, "variant": m, "auc": 0.66 + 0.01 * i + 0.002 * j}
            for i, m in enumerate(SERIES)
            for j, q in enumerate(quarters)
        ]
    )
    path = auc_by_vintage(frame, tmp_path / "auc.png")
    assert path.read_bytes()[:4] == b"\x89PNG"
