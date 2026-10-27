"""Model inputs from the processed table: cleaning, derived ratios and feature sets."""

from pathlib import Path

import numpy as np
import pandas as pd

from crisk.data.build import CATEGORICAL, FEATURES
from crisk.data.columns import Role, names

PRICING = names(Role.PRICING)  # grade, sub_grade, int_rate, installment
# Features empty (or nearly) in train can't be binned or learned: joint applications and the
# bureau fields Lending Club added in late 2015.
MAX_TRAIN_MISSING = 0.95
DERIVED = ["loan_to_income"]
# Kept out of every model by a decision made on validation (README, Phase 5). The column stays
# in the modelling table, so the fairness audit can still measure approval rates by state.
EXCLUDED = {
    "addr_state": "geography is a redlining proxy, not actionable for an applicant, and added "
    "only +0.0012 validation AUC and +0.05 pp validation return (rule: +0.005 or +0.10 pp)",
}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix source sentinels and add application-time ratios. Returns a new frame."""
    df = df.copy()
    # dti uses -1 and 999 for "not computable"; treat as missing.
    df["dti"] = df["dti"].where((df["dti"] >= 0) & (df["dti"] < 999))
    income = df["annual_inc"].where(df["annual_inc"] > 0)
    df["loan_to_income"] = (df["loan_amnt"] / income).astype("float32")
    return df


def available(train: pd.DataFrame, candidates: list[str]) -> list[str]:
    missing = train[candidates].isna().mean()
    return [c for c in candidates if missing[c] <= MAX_TRAIN_MISSING]


def feature_sets(train: pd.DataFrame) -> dict[str, list[str]]:
    """`with_pricing` uses Lending Club's grade and rate; `no_pricing` only borrower data."""
    candidates = [c for c in [*FEATURES, *DERIVED] if c not in EXCLUDED]
    all_features = available(train, candidates)
    return {
        "with_pricing": all_features,
        "no_pricing": [c for c in all_features if c not in PRICING],
    }


def categorical(features: list[str]) -> list[str]:
    return [c for c in features if c in CATEGORICAL]


def as_binning_input(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """optbinning wants object strings (with NaN) for categoricals and float64 numerics."""
    out = {}
    for c in features:
        if c in CATEGORICAL:
            out[c] = df[c].astype(object).where(df[c].notna(), np.nan)
        else:
            out[c] = df[c].astype("float64")
    return pd.DataFrame(out, index=df.index)


def load_model_table(path: Path) -> pd.DataFrame:
    return clean(pd.read_parquet(path))
