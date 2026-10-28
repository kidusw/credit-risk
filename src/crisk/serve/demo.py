"""A DEMO Lending Club model for fresh clones without a Kaggle token.

The real loan-level data can't ship with the repository (licence: research use, never
redistributed), so `docker compose up` on a fresh clone would have no Lending Club model.
This trains one on **synthetic, Lending-Club-shaped** loans instead: the same feature
names, the same LightGBM + Platt pipeline, a seeded generator with plausible
relationships (FICO drives sub-grade, sub-grade drives the rate, risk rises with DTI,
loan-to-income and recent credit hunger). Every response from it says `synthetic: true`,
and /model says `demo: true`.

It is there so the API and dashboard work end to end. Its numbers mean nothing; the real
model's are in the README. With KAGGLE_API_TOKEN set, bootstrap trains the real one.
"""

import numpy as np
import pandas as pd

from crisk.decision.policy import best_cutoff, sweep
from crisk.models.calibration import Calibrator
from crisk.models.gbm import SUB_GRADES, GBMModel
from crisk.models.metrics import binary_metrics

SEED = 7
FEATURES = [
    "loan_amnt",
    "term_months",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "annual_inc",
    "loan_to_income",
    "dti",
    "fico_range_low",
    "emp_length_years",
    "home_ownership",
    "purpose",
    "verification_status",
    "revol_util",
    "inq_last_6mths",
    "acc_open_past_24mths",
    "delinq_2yrs",
    "pub_rec",
    "mort_acc",
    "credit_history_months",
]
PARAMS = {"num_leaves": 15, "min_child_samples": 200, "subsample": 0.85, "colsample_bytree": 0.8}
PURPOSES = [
    "debt_consolidation",
    "credit_card",
    "home_improvement",
    "other",
    "major_purchase",
    "small_business",
    "car",
    "medical",
]


def synthetic_loans(n: int = 40_000, seed: int = SEED) -> pd.DataFrame:
    """SYNTHETIC Lending-Club-shaped loans with a default label. No real data involved."""
    rng = np.random.default_rng(seed)
    fico = np.clip(np.round(rng.normal(695, 30, n) / 5) * 5, 660, 845)
    idx = np.clip(np.round((765 - fico) / 4.5 + rng.normal(0, 3, n)), 0, 34).astype(int)
    sub_grade = np.array(SUB_GRADES)[idx]
    term = np.where(rng.random(n) < 0.25 + idx / 100, 60, 36)
    rate = np.round(5.5 + 0.62 * idx + rng.normal(0, 0.6, n), 2)
    income = np.round(rng.lognormal(np.log(68_000), 0.5, n), -2)
    amount = np.clip(np.round(income * rng.uniform(0.05, 0.45, n), -2), 1_000, 35_000)
    r = rate / 1200
    installment = np.round(amount * r / (1 - (1 + r) ** -term), 2)
    dti = np.clip(rng.gamma(4, 4.5, n), 0, 39.9).round(2)
    acc24 = rng.poisson(4, n)
    inq = rng.poisson(0.7, n)
    revol = np.clip(rng.normal(55, 22, n), 0, 120).round(1)
    history = np.clip(rng.normal(190, 80, n), 36, 600).round()
    home = rng.choice(["MORTGAGE", "RENT", "OWN"], n, p=[0.48, 0.41, 0.11])
    z = (
        -2.15
        + 0.075 * (idx - 17)
        + 0.45 * (term == 60)
        + 0.025 * (dti - 18)
        + 0.9 * (amount / income - 0.2)
        + 0.07 * (acc24 - 4)
        + 0.12 * inq
        - 0.25 * np.log(income / 68_000)
        - 0.002 * (history - 190)
        + 0.2 * (home == "RENT")
        + rng.normal(0, 0.6, n)
    )
    df = pd.DataFrame(
        {
            "loan_amnt": amount,
            "term_months": term,
            "int_rate": rate,
            "installment": installment,
            "grade": pd.Series(sub_grade).str[0],
            "sub_grade": sub_grade,
            "annual_inc": income,
            "loan_to_income": amount / income,
            "dti": dti,
            "fico_range_low": fico,
            "emp_length_years": rng.choice(np.arange(11), n).astype(float),
            "home_ownership": home,
            "purpose": rng.choice(PURPOSES, n),
            "verification_status": rng.choice(["Verified", "Source Verified", "Not Verified"], n),
            "revol_util": revol,
            "inq_last_6mths": inq,
            "acc_open_past_24mths": acc24,
            "delinq_2yrs": rng.poisson(0.3, n),
            "pub_rec": rng.poisson(0.15, n),
            "mort_acc": np.where(home == "MORTGAGE", rng.poisson(2, n), rng.poisson(0.3, n)),
            "credit_history_months": history,
            "default": (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int),
        }
    )
    for c in ("grade", "sub_grade", "home_ownership", "purpose", "verification_status"):
        df[c] = df[c].astype("category")
    df["split"] = np.select([df.index < 0.6 * n, df.index < 0.8 * n], ["train", "valid"], "test")
    # Simplified economics for the cutoff: interest on good loans, 60% lost on bad ones.
    good = df["int_rate"] / 100 * df["term_months"] / 12 * 0.55
    df["funded_amnt"] = df["loan_amnt"]
    df["profit"] = np.where(df["default"] == 1, -0.6, good) * df["loan_amnt"]
    df["source"] = "synthetic"
    return df


def train_demo(n: int = 40_000):
    """Fit the demo LightGBM, calibrate on its validation split, choose a return cutoff."""
    df = synthetic_loans(n)
    train, valid, test = (
        df[df["split"] == s].reset_index(drop=True) for s in ("train", "valid", "test")
    )
    gbm = GBMModel(FEATURES, PARAMS).fit(train, valid)
    cal = Calibrator("platt").fit(gbm.predict_pd(valid), valid["default"])
    cutoff = best_cutoff(sweep(cal.transform(gbm.predict_pd(valid)), valid), "return")
    metrics = {
        s: {
            k: round(v, 4)
            for k, v in binary_metrics(part["default"], gbm.predict_pd(part)).items()
            if k in ("auc", "ks", "brier")
        }
        for s, part in (("train", train), ("valid", valid), ("test", test))
    }
    return gbm, cal, cutoff, test, metrics
