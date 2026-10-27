import numpy as np
import pandas as pd
import pytest

from crisk.data.build import FEATURES
from crisk.explain import fairness, reasons
from crisk.explain.contributions import contributions, dependence, global_importance
from crisk.explain.run import decide_state
from crisk.models.calibration import logit
from crisk.models.features import DERIVED, EXCLUDED, feature_sets
from crisk.models.gbm import GBMModel


def test_every_feature_has_exactly_one_reason():
    mapped = [f for r in reasons.REASONS for f in r.features]
    assert len(mapped) == len(set(mapped)), "a feature is in two reasons"
    assert set(FEATURES) | set(DERIVED) <= set(mapped)
    assert set(mapped) <= set(FEATURES) | set(DERIVED)
    assert len({r.code for r in reasons.REASONS}) == len(reasons.REASONS)


@pytest.fixture(scope="module")
def fitted():
    """A small monotonic model where DTI and FICO drive risk and purpose adds a bump."""
    rng = np.random.default_rng(5)
    n = 8000
    fico = rng.normal(700, 35, n)
    dti = rng.uniform(0, 40, n)
    purpose = rng.choice(["car", "credit_card", "small_business"], n)
    z = -1.8 - 0.025 * (fico - 700) + 0.05 * (dti - 20) + (purpose == "small_business")
    df = pd.DataFrame(
        {
            "fico_range_low": fico,
            "dti": dti,
            "purpose": pd.Categorical(purpose),
            "default": (rng.uniform(size=n) < 1 / (1 + np.exp(-z))).astype(int),
        }
    )
    train, valid = df.iloc[:6000], df.iloc[6000:].reset_index(drop=True)
    model = GBMModel(["fico_range_low", "dti", "purpose"], {"num_leaves": 15}).fit(train, valid)
    return model, valid


def test_shap_values_add_up_to_the_model_logit(fitted):
    model, valid = fitted
    phi, base = contributions(model, valid)
    assert list(phi.columns) == model.features
    np.testing.assert_allclose(base + phi.sum(axis=1), logit(model.predict_pd(valid)), atol=1e-6)
    imp = global_importance(phi, model.importance().set_index("feature")["gain_share"])
    assert imp["shap_share"].sum() == pytest.approx(1.0)
    assert imp.loc[0, "feature"] in ("dti", "fico_range_low")
    dep = dependence(phi, valid, "dti", bins=10)
    assert len(dep) == 10 and dep["mean_shap"].is_monotonic_increasing  # monotone constraint
    assert set(dependence(phi, valid, "purpose")["level"]) == {
        "car",
        "credit_card",
        "small_business",
    }


def test_every_decline_gets_reasons(fitted):
    model, valid = fitted
    p = model.predict_pd(valid)
    declined = p > np.quantile(p, 0.7)
    phi, _ = contributions(model, valid)
    codes = reasons.reason_codes(phi[declined])
    assert len(codes) == declined.sum()
    assert (codes["n_reasons"] >= 1).all() and (codes["n_reasons"] <= 3).all()  # 3 reasons exist
    assert codes["reason_1"].notna().all()
    assert codes["reason_1"].isin(["R01", "R04", "R16"]).all()
    w = codes[["weight_1", "weight_2"]].dropna()
    assert (w["weight_1"] >= w["weight_2"]).all() and (w > 0).all().all()
    text = reasons.describe(codes)
    assert text["reason_1"].isin([reasons.BY_CODE[c].text for c in ("R01", "R04", "R16")]).all()
    # High-DTI declines cite DTI.
    high_dti = valid.loc[declined, "dti"].to_numpy() > 38
    assert (codes.loc[high_dti, ["reason_1", "reason_2", "reason_3"]] == "R04").any(axis=1).all()
    freq = reasons.frequency(codes)
    assert freq["first_reason_share"].sum() == pytest.approx(1.0)


def test_reasons_sum_features_and_refuse_rows_without_a_reason():
    phi = pd.DataFrame(
        {"dti": [0.3, -0.2], "dti_joint": [0.1, -0.1], "fico_range_low": [-0.5, -0.3]}
    )
    weights = reasons.grouped(phi)
    assert weights.loc[0, "R04"] == pytest.approx(0.4)
    codes = reasons.reason_codes(phi.iloc[[0]])
    assert codes.loc[0, "reason_1"] == "R04" and codes.loc[0, "n_reasons"] == 1
    assert codes.loc[0, "reason_2"] is None
    with pytest.raises(ValueError, match="no adverse reason"):
        reasons.reason_codes(phi)
    with pytest.raises(KeyError, match="without a reason code"):
        reasons.grouped(phi.assign(mystery=0.0))


def test_group_table_metrics():
    # Group a: 4 loans (1 bad), 3 approved incl. the bad one. Group b: 2 loans, 1 approved.
    group = pd.Series(["a", "a", "a", "a", "b", "b"])
    approve = [True, True, False, True, True, False]
    y = [0, 0, 0, 1, 0, 0]
    p = [0.1, 0.1, 0.3, 0.2, 0.1, 0.4]
    t = fairness.group_table(group, approve, y, p, min_group=1).set_index("group")
    assert t.loc["a", "approval_rate"] == 0.75 and t.loc["b", "approval_rate"] == 0.5
    assert t.loc["b", "air"] == pytest.approx(0.5 / 0.75)
    assert t.loc["a", "tpr"] == pytest.approx(2 / 3) and t.loc["b", "tpr"] == 0.5
    assert t.loc["b", "tpr_gap"] == pytest.approx(0.5 - 2 / 3)
    assert t.loc["a", "citl"] == pytest.approx(0.175 - 0.25)
    assert t.loc["a", "reference"] and not t.loc["b", "reference"]
    s = fairness.summary(t.reset_index(), "x", min_group=1)
    assert s["min_air_group"] == "b" and s["groups_below_0_8"] == 1


def test_bands():
    inc = fairness.income_band(pd.Series([39_999.0, 40_000.0, 150_000.0, np.nan]))
    assert inc.tolist() == ["< 40k", "40-60k", ">= 150k", "missing"]
    emp = fairness.employment_band(pd.Series([0.0, 1.0, 9.0, 10.0, np.nan]))
    assert emp.tolist() == ["< 1", "1-4", "5-9", "10+", "missing"]


def test_state_rule():
    def ablation(auc_with, ret_with):
        return pd.DataFrame(
            {
                "with_addr_state": [True, False],
                "split": "valid",
                "auc": [auc_with, 0.690],
                "return_on_lent": [ret_with, 0.070],
            }
        )

    assert not decide_state(ablation(0.692, 0.0705))["keep_addr_state"]
    assert decide_state(ablation(0.696, 0.0705))["keep_addr_state"]
    assert decide_state(ablation(0.691, 0.0712))["keep_addr_state"]


def test_excluded_features_never_reach_a_model():
    train = pd.DataFrame({c: [1.0, 2.0] for c in [*FEATURES, *DERIVED]})
    assert set(EXCLUDED) <= set(FEATURES)
    for features in feature_sets(train).values():
        assert not set(features) & set(EXCLUDED)
    assert "addr_state" in fairness.attributes(
        train.assign(addr_state="CA", home_ownership="RENT", zip_code="123xx")
    )
