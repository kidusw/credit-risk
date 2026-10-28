"""Phase 8: export -> predictor parity, the API contract, decision logging, dashboard client."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from crisk.dashboard.client import ApiClient
from crisk.db.models import Base
from crisk.mobile_money.features import FEATURES as MM_FEATURES
from crisk.mobile_money.features import MONOTONE as MM_MONOTONE
from crisk.mobile_money.run import REASONS as MM_REASONS
from crisk.models.calibration import Calibrator
from crisk.models.gbm import GBMModel
from crisk.serve import export as ex
from crisk.serve.app import create_app
from crisk.serve.demo import FEATURES as LC_FEATURES
from crisk.serve.demo import synthetic_loans
from crisk.serve.predictor import load_models


def _mm_frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f: rng.gamma(2, 1, n) for f in MM_FEATURES})
    z = (
        -2
        + 0.8 * (df["inflow_cv"] - 2)
        + 0.6 * (df["loan_to_inflow"] - 2)
        - 0.5 * df["equb_member"]
    )
    df["default"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return df


@pytest.fixture(scope="module")
def serving(tmp_path_factory):
    """Both tracks exported from small models, plus what they were fitted on."""
    out = tmp_path_factory.mktemp("serving")
    lc = synthetic_loans(6000)
    lc_train, lc_valid = (
        lc[lc["split"] == "train"],
        lc[lc["split"] != "train"].reset_index(drop=True),
    )
    lc_gbm = GBMModel(LC_FEATURES, {"num_leaves": 7}).fit(lc_train, lc_valid)
    lc_cal = Calibrator("platt").fit(lc_gbm.predict_pd(lc_valid), lc_valid["default"])
    from crisk.explain.reasons import REASONS

    ex.export_track(
        out,
        "lending_club",
        lc_gbm,
        lc_cal,
        0.15,
        REASONS,
        ex.examples_records(lc_valid.head(3), LC_FEATURES, lc_valid.head(3).index.to_series(), "t"),
        {
            "model_name": "test",
            "trained_on": "test",
            "synthetic": True,
            "demo": True,
            "cutoff_rule": "fixed",
            "currency": "USD",
            "metrics": {},
        },
    )
    mm_train, mm_valid = _mm_frame(4000, 1), _mm_frame(1000, 2)
    mm_gbm = GBMModel(MM_FEATURES, {"num_leaves": 7}, monotone=MM_MONOTONE).fit(mm_train, mm_valid)
    mm_cal = Calibrator("isotonic").fit(mm_gbm.predict_pd(mm_valid), mm_valid["default"])
    ex.export_track(
        out,
        "mobile_money",
        mm_gbm,
        mm_cal,
        0.2,
        MM_REASONS,
        [],
        {
            "model_name": "test",
            "trained_on": "test",
            "synthetic": True,
            "demo": False,
            "cutoff_rule": "fixed",
            "currency": "ETB",
            "metrics": {},
        },
    )
    return out, (lc_gbm, lc_cal, lc_valid), (mm_gbm, mm_cal, mm_valid)


@pytest.fixture
def engine():
    eng = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture
def client(serving, engine):
    app = create_app(models=load_models(serving[0]), engine=engine)
    with TestClient(app) as c:
        yield c


def _records(df: pd.DataFrame, features: list[str]) -> list[dict]:
    return df[features].astype(object).where(df[features].notna(), None).to_dict("records")


def test_served_pd_matches_the_training_pipeline(serving):
    out, (lc_gbm, lc_cal, lc_valid), (mm_gbm, mm_cal, mm_valid) = serving
    models = load_models(out)
    lc = np.array([s.pd for s in models["lending_club"].score(_records(lc_valid, LC_FEATURES))])
    np.testing.assert_allclose(lc, lc_cal.transform(lc_gbm.predict_pd(lc_valid)), atol=1e-9)
    mm = np.array([s.pd for s in models["mobile_money"].score(_records(mm_valid, MM_FEATURES))])
    np.testing.assert_allclose(mm, mm_cal.transform(mm_gbm.predict_pd(mm_valid)), atol=1e-9)


def test_export_writes_no_pickles(serving):
    files = {p.name for p in serving[0].iterdir()}
    assert files == {
        f"{t}{suffix}" for t in ex.TRACKS for suffix in (".json", ".lgb.txt", "_examples.json")
    }


def test_score_contract_and_reasons(client, serving):
    lc_valid = serving[1][2]
    rows = _records(lc_valid, LC_FEATURES)
    seen = set()
    for row in rows[:60]:
        r = client.post("/score", json={"track": "lending_club", "applicant": row}).json()
        seen.add(r["decision"])
        assert 0 <= r["pd"] <= 1 and r["risk_band"] in "ABCDEF"
        assert r["decision"] == ("decline" if r["pd"] > r["cutoff"] else "approve")
        if r["decision"] == "decline":
            assert 1 <= len(r["reasons"]) <= 4
            assert all(x["weight"] > 0 for x in r["reasons"])
        else:
            assert r["reasons"] == []
        assert r["missing_features"] == []
    assert seen == {"approve", "decline"}


def test_score_rejects_unknown_features_and_tracks(client):
    r = client.post("/score", json={"track": "lending_club", "applicant": {"zip_code": "123xx"}})
    assert r.status_code == 422
    r = client.post("/score", json={"track": "other", "applicant": {}})
    assert r.status_code == 422


def test_missing_features_are_reported_and_loan_to_income_is_derived(client):
    r = client.post(
        "/score",
        json={"track": "lending_club", "applicant": {"loan_amnt": 10000, "annual_inc": 50000}},
    ).json()
    assert "loan_to_income" not in r["missing_features"]
    assert "fico_range_low" in r["missing_features"]
    assert r["synthetic"] is True  # demo model


def test_points_fall_as_pd_rises(serving):
    m = load_models(serving[0])["lending_club"]
    pts = [m.points(p) for p in (0.01, 0.05, 0.1, 0.3, 0.6)]
    assert pts == sorted(pts, reverse=True)
    assert m.points(1 / 51) == 600  # 50:1 odds is the base score
    assert [m.band(p) for p in (0.01, 0.07, 0.12, 0.17, 0.25, 0.5)] == list("ABCDEF")


def test_decisions_are_logged(client):
    body = {"track": "mobile_money", "applicant": {"inflow_cv": 3.0}, "applicant_ref": "abc"}
    first = client.post("/score", json=body).json()
    assert first["decision_id"] == 1
    logged = client.get("/decisions").json()
    assert logged[0]["applicant_ref"] == "abc" and logged[0]["track"] == "mobile_money"


def test_scoring_survives_a_dead_database(serving):
    dead = create_engine(
        "postgresql+psycopg://x:x@127.0.0.1:1/x", connect_args={"connect_timeout": 1}
    )
    app = create_app(models=load_models(serving[0]), engine=dead)
    with TestClient(app) as c:
        r = c.post("/score", json={"track": "mobile_money", "applicant": {}})
        assert r.status_code == 200 and r.json()["decision_id"] is None
        assert c.get("/health").json()["database"] is False


def test_model_and_examples_endpoints(client):
    info = client.get("/model").json()
    assert set(info) == {"lending_club", "mobile_money"}
    assert info["mobile_money"]["synthetic"] is True
    assert len(client.get("/examples/lending_club").json()) == 3
    assert client.get("/health").json() == {
        "status": "ok",
        "tracks": ["lending_club", "mobile_money"],
        "database": True,
    }


def test_dashboard_client_talks_to_the_api(client):
    api = ApiClient(base_url="http://testserver", http=client)
    assert api.health()["database"] is True
    ex_row = api.examples("lending_club")[0]
    res = api.score("lending_club", ex_row["applicant"], ex_row["ref"])
    assert res["track"] == "lending_club"
