"""Challenger: LightGBM with monotonic constraints where the risk direction is known.

A constraint only says the PD can't move the wrong way as the feature grows (all else
equal); the model still learns the shape. Features with no agreed direction are left free.
Lending Club's grade and sub-grade are turned into ordinals (A1 = 0 ... G5 = 34) so they
can be constrained too.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from crisk.data.build import CATEGORICAL

# +1: higher value -> PD can only go up. -1: higher value -> PD can only go down.
MONOTONE = {
    # loan and pricing
    "term_months": +1,
    "int_rate": +1,
    "grade": +1,
    "sub_grade": +1,
    "loan_to_income": +1,
    # capacity
    "annual_inc": -1,
    "dti": +1,
    # credit quality and history
    "fico_range_low": -1,
    "credit_history_months": -1,
    "mo_sin_old_rev_tl_op": -1,
    "mo_sin_old_il_acct": -1,
    "pct_tl_nvr_dlq": -1,
    "mort_acc": -1,
    # utilisation
    "revol_util": +1,
    "bc_util": +1,
    "percent_bc_gt_75": +1,
    "bc_open_to_buy": -1,
    # credit hunger: recent inquiries and new accounts
    "inq_last_6mths": +1,
    "acc_open_past_24mths": +1,
    "num_tl_op_past_12m": +1,
    "mths_since_recent_inq": -1,
    "mo_sin_rcnt_tl": -1,
    "mo_sin_rcnt_rev_tl_op": -1,
    # derogatory marks
    "delinq_2yrs": +1,
    "pub_rec": +1,
    "pub_rec_bankruptcies": +1,
    "tax_liens": +1,
    "collections_12_mths_ex_med": +1,
    "chargeoff_within_12_mths": +1,
    "num_accts_ever_120_pd": +1,
    "num_tl_90g_dpd_24m": +1,
}

ORDINAL = ("grade", "sub_grade")
SUB_GRADES = [f"{g}{n}" for g in "ABCDEFG" for n in range(1, 6)]

FIXED_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "n_estimators": 3000,  # upper bound; early stopping on validation picks the count
    "subsample_freq": 1,
    "verbose": -1,
    "n_jobs": 6,
    "random_state": 0,
}
# Random-search space (seeded). Kept small: ~2 minutes per configuration on 315k loans.
SEARCH_SPACE = {
    "num_leaves": [15, 31, 63],
    "min_child_samples": [200, 500, 1000],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.5, 0.7, 0.9],
    "reg_lambda": [0.0, 1.0, 10.0],
}


def prepare(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Model matrix: grades as ordinals, other categoricals as pandas categories."""
    X = df[features].copy()
    for col in features:
        if col == "grade":
            X[col] = X[col].astype(str).map({g: i for i, g in enumerate("ABCDEFG")})
        elif col == "sub_grade":
            X[col] = X[col].astype(str).map({g: i for i, g in enumerate(SUB_GRADES)})
        elif col in CATEGORICAL:
            X[col] = X[col].astype("category")
        else:
            X[col] = X[col].astype("float32")
    return X


def constraints(features: list[str], monotone: dict[str, int] | None = None) -> list[int]:
    monotone = MONOTONE if monotone is None else monotone
    return [monotone.get(f, 0) for f in features]


def sample_configs(n: int, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    configs, seen = [], set()
    while len(configs) < n:
        cfg = {k: v[rng.integers(len(v))] for k, v in SEARCH_SPACE.items()}
        key = tuple(cfg.values())
        if key not in seen:
            seen.add(key)
            configs.append(cfg)
    return configs


@dataclass
class GBMModel:
    features: list[str]
    params: dict = field(default_factory=dict)
    model: object = None
    best_iteration: int = 0
    monotone: dict[str, int] | None = None  # None: the Lending Club directions (MONOTONE)

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame, early_stopping: int = 100):
        import lightgbm as lgb  # lazy: keeps the API image free of libgomp

        self.model = lgb.LGBMClassifier(
            **FIXED_PARAMS,
            **self.params,
            monotone_constraints=constraints(self.features, self.monotone),
            monotone_constraints_method="advanced",
        )
        self.model.fit(
            prepare(train, self.features),
            train["default"],
            eval_set=[(prepare(valid, self.features), valid["default"])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(early_stopping, verbose=False)],
        )
        self.best_iteration = int(self.model.best_iteration_)
        return self

    def predict_pd(self, df: pd.DataFrame) -> np.ndarray:
        X = prepare(df, self.features)
        return self.model.predict_proba(X, num_iteration=self.best_iteration)[:, 1]

    def importance(self) -> pd.DataFrame:
        booster = self.model.booster_
        return (
            pd.DataFrame(
                {
                    "feature": self.features,
                    "gain": booster.feature_importance("gain"),
                    "splits": booster.feature_importance("split"),
                    "monotone": constraints(self.features, self.monotone),
                }
            )
            .assign(gain_share=lambda d: d["gain"] / d["gain"].sum())
            .sort_values("gain", ascending=False, ignore_index=True)
        )
