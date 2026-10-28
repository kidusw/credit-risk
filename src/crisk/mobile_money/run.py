"""Phase 6: the thin-file track on SYNTHETIC mobile-money borrowers, end to end.

Same pipeline and protocol as the Lending Club model: WoE scorecard and monotonic LightGBM
fitted on the 2024 cohorts, every choice (hyperparameters, champion, calibration method,
approval cutoff) made on the Q1 2025 cohort, and the Q2 2025 cohort scored once at the end.
On top: reason codes for every decline, a fairness audit by gender, region and urban/rural
(attributes the generator created for that purpose), and a check of how well the model
recovered the hidden rule that produced the labels.

**Caveat:** the labels come from a rule written in `generate.py`. This shows the method on a
realistic local use case. It says nothing about real-world accuracy.
"""

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from crisk.decision.policy import best_cutoff, evaluate, sweep
from crisk.explain import fairness, reasons
from crisk.explain.contributions import contributions, global_importance
from crisk.explain.reasons import Reason
from crisk.mobile_money.features import FEATURES, MONOTONE
from crisk.mobile_money.generate import NOISE_SD, RULE
from crisk.models.calibration import Calibrator, calibration_metrics, select_method
from crisk.models.compare import auc_lift_ci
from crisk.models.gbm import GBMModel
from crisk.models.metrics import binary_metrics
from crisk.models.scorecard import Scorecard

TRAIN_MONTHS = range(0, 12)  # Jan-Dec 2024
VALID_MONTHS = range(12, 15)  # Q1 2025
MODEL_FILE = "mm_model.joblib"

# Assumed microloan economics (synthetic, illustrative): a flat 9% fee on a 90-day loan,
# and 75% of the amount lost when a borrower defaults (some is recovered from later inflows).
FEE = 0.09
LGD = 0.75

ATTRIBUTES = ("gender", "region", "urban_rural", "occupation")
MIN_GROUP = 200  # the test cohort is ~4,700 borrowers, so groups are smaller than for LC

SEARCH = [  # small seeded grid, as for the LC model
    {"num_leaves": nl, "min_child_samples": mc, "colsample_bytree": cs, "reg_lambda": rl}
    for nl in (7, 15, 31)
    for mc in (50, 200)
    for cs in (0.7, 1.0)
    for rl in (0.0, 5.0)
]

REASONS = [
    Reason(
        "M01",
        "Income is irregular or seasonal",
        ("income_regularity", "months_without_inflow", "airtime_gap_cv"),
    ),
    Reason("M02", "Income varies a lot from month to month", ("inflow_cv", "balance_volatility")),
    Reason(
        "M03",
        "Spending close to or above income",
        ("inflow_outflow_ratio", "balance_trend", "cash_out_share"),
    ),
    Reason(
        "M04",
        "Wallet balance often low",
        ("balance_min_to_inflow", "balance_mean_to_inflow", "low_balance_days"),
    ),
    Reason(
        "M05", "Loan amount high compared with wallet inflows", ("loan_to_inflow", "loan_amount")
    ),
    Reason("M06", "Recent fall in income", ("recent_inflow_change", "recent_income_dip")),
    Reason("M07", "No regular equb or iddir contributions", ("equb_member", "iddir_member")),
    Reason(
        "M08", "Few people sending or receiving transfers", ("p2p_out_contacts", "p2p_in_contacts")
    ),
    Reason("M09", "Low wallet inflows", ("inflow_monthly",)),
    Reason(
        "M10",
        "Airtime, bill and transaction pattern",
        ("airtime_per_month", "airtime_share", "regular_bills", "txn_per_month"),
    ),
]

# Which true driver of the hidden rule each feature measures (None: not a driver).
FEATURE_DRIVER = {
    "income_regularity": "irregular_income",
    "months_without_inflow": "irregular_income",
    "airtime_gap_cv": "irregular_income",
    "inflow_cv": "volatility",
    "balance_volatility": "volatility",
    "inflow_outflow_ratio": "overspending",
    "balance_trend": "overspending",
    "cash_out_share": "overspending",
    "balance_min_to_inflow": "thin_buffer",
    "balance_mean_to_inflow": "thin_buffer",
    "low_balance_days": "thin_buffer",
    "loan_to_inflow": "loan_size",
    "loan_amount": "loan_size",
    "recent_inflow_change": "income_shock",
    "recent_income_dip": "income_shock",
    "equb_member": "social_ties",
    "iddir_member": "social_ties",
    "p2p_out_contacts": "social_ties",
    "p2p_in_contacts": "social_ties",
    "inflow_monthly": "low_income",
}


@dataclass
class MMResult:
    metrics: pd.DataFrame = None
    lift: pd.DataFrame = None
    tuning: pd.DataFrame = None
    champion: str = ""
    calibration: pd.DataFrame = None
    policies: pd.DataFrame = None
    cutoffs: dict = field(default_factory=dict)
    importance: pd.DataFrame = None
    recovery: pd.DataFrame = None
    reason_frequency: pd.DataFrame = None
    reason_counts: pd.DataFrame = None
    reason_sample: pd.DataFrame = None
    fairness: pd.DataFrame = None
    fairness_summary: pd.DataFrame = None
    scorecard_points: pd.DataFrame = None
    bundle: dict = field(default_factory=dict)


def split_of(cohort_month: pd.Series) -> np.ndarray:
    return np.select(
        [cohort_month.isin(TRAIN_MONTHS), cohort_month.isin(VALID_MONTHS)],
        ["train", "valid"],
        "test",
    )


def model_table(features: pd.DataFrame, borrowers: pd.DataFrame) -> pd.DataFrame:
    """Features joined to the label, the split and the audit attributes (never model inputs)."""
    keep = ["borrower_id", "cohort_month", "default", "true_pd", "loan_amount", *ATTRIBUTES]
    keep += [f"rule_{d}" for d in RULE]
    df = features.drop(columns="loan_amount").merge(
        borrowers[keep], on="borrower_id", validate="one_to_one"
    )
    df["split"] = split_of(df["cohort_month"])
    # The economics the policies are scored on: realised profit per loan.
    df["funded_amnt"] = df["loan_amount"]
    df["profit"] = np.where(df["default"] == 1, -LGD, FEE) * df["loan_amount"]
    return df


def tune(
    train: pd.DataFrame, valid: pd.DataFrame, search: list[dict] = SEARCH
) -> tuple[GBMModel, pd.DataFrame]:
    rows, best, best_auc = [], None, -1.0
    for i, cfg in enumerate(search):
        model = GBMModel(FEATURES, cfg, monotone=MONOTONE).fit(train, valid)
        auc = roc_auc_score(valid["default"], model.predict_pd(valid))
        rows.append({"config": i, **cfg, "best_iteration": model.best_iteration, "valid_auc": auc})
        if auc > best_auc:
            best, best_auc = model, auc
    return best, pd.DataFrame(rows).sort_values("valid_auc", ascending=False, ignore_index=True)


class LoanToInflowBaseline:
    """What a lender without a model might use: the loan's size relative to wallet inflows,
    turned into a PD by a one-feature logistic regression on train."""

    def _x(self, df: pd.DataFrame) -> np.ndarray:
        x = df["loan_to_inflow"].astype(float).fillna(df["loan_to_inflow"].max())
        return np.log(x.clip(lower=0.01)).to_numpy()[:, None]

    def fit(self, df: pd.DataFrame, y: pd.Series) -> "LoanToInflowBaseline":
        self.model = LogisticRegression().fit(self._x(df), y)
        return self

    def predict_pd(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._x(df))[:, 1]


def recovery(phi: pd.DataFrame, rule: pd.DataFrame) -> pd.DataFrame:
    """Share of the model's |SHAP| per true driver vs that driver's share of the hidden rule.

    A driver's weight in the rule is the mean absolute deviation of its term from the
    average borrower's, which is what SHAP measures on the model side.
    """
    shap = phi.abs().mean()
    by_driver = shap.groupby(shap.index.map(lambda f: FEATURE_DRIVER.get(f, "none"))).sum()
    true = (rule - rule.mean()).abs().mean()
    true.index = true.index.str.removeprefix("rule_")
    noise = NOISE_SD * np.sqrt(2 / np.pi)  # mean |N(0, sd)|
    out = pd.DataFrame(
        {
            "true_share": true / true.sum(),
            "model_share": by_driver.reindex(true.index).fillna(0) / by_driver.sum(),
            "formula": pd.Series(RULE),
        }
    )
    extra = pd.DataFrame(
        {
            "true_share": [0.0],
            "model_share": [by_driver.get("none", 0) / by_driver.sum()],
            "formula": ["features that measure no driver"],
        },
        index=["none"],
    )
    out = pd.concat([out, extra]).rename_axis("driver").reset_index()
    out["noise_vs_signal"] = noise / true.sum()
    return out.sort_values("true_share", ascending=False, ignore_index=True)


def run(
    features: pd.DataFrame,
    borrowers: pd.DataFrame,
    search: list[dict] = SEARCH,
    min_group: int = MIN_GROUP,
) -> MMResult:
    res = MMResult()
    df = model_table(features, borrowers)
    train, valid, test = (df[df["split"] == s] for s in ("train", "valid", "test"))
    held = df[df["split"] != "train"].reset_index(drop=True)
    is_valid = (held["split"] == "valid").to_numpy()

    card = Scorecard(FEATURES).fit(train, train["default"])
    gbm, res.tuning = tune(train, valid, search)
    baseline = LoanToInflowBaseline().fit(train, train["default"])
    models = {"baseline_loan_to_inflow": baseline, "scorecard": card, "gbm": gbm}

    rows = []
    for name, m in models.items():
        for split, part in (("train", train), ("valid", valid), ("test", test)):
            rows.append(
                {
                    "model": name,
                    "split": split,
                    **binary_metrics(part["default"].to_numpy(), m.predict_pd(part)),
                }
            )
    res.metrics = pd.DataFrame(rows)
    v = res.metrics[res.metrics["split"] == "valid"].set_index("model")["auc"]
    res.champion = str(v.drop("baseline_loan_to_inflow").idxmax())
    champion = models[res.champion]

    scored = {name: m.predict_pd(held) for name, m in models.items()}
    lifts = []
    for a, b in (
        ("gbm", "scorecard"),
        ("gbm", "baseline_loan_to_inflow"),
        ("scorecard", "baseline_loan_to_inflow"),
    ):
        for split in ("valid", "test"):
            m = (held["split"] == split).to_numpy()
            lifts.append(
                {
                    "challenger": a,
                    "incumbent": b,
                    "split": split,
                    **auc_lift_ci(held.loc[m, "default"].to_numpy(), scored[a][m], scored[b][m]),
                }
            )
    res.lift = pd.DataFrame(lifts)

    # Calibration: method by cross-validation within validation, then fitted on all of it.
    y = held["default"].to_numpy()
    method, _ = select_method(scored[res.champion][is_valid], y[is_valid])
    calibrator = Calibrator(method).fit(scored[res.champion][is_valid], y[is_valid])
    pd_cal = calibrator.transform(scored[res.champion])
    res.calibration = pd.DataFrame(
        [
            {"version": ver, "split": s, "method": method, **calibration_metrics(y[m], p[m])}
            for ver, p in (("raw", scored[res.champion]), ("calibrated", pd_cal))
            for s, m in (("valid", is_valid), ("test", ~is_valid))
        ]
    )

    # Approval cutoffs chosen on validation (total profit), applied to test.
    sc_cal = Calibrator(method).fit(scored["scorecard"][is_valid], y[is_valid])
    scores = {
        "rule of thumb: loan / inflow": scored["baseline_loan_to_inflow"],
        "scorecard": sc_cal.transform(scored["scorecard"]),
        f"champion ({res.champion})": pd_cal,
    }
    vh, th = held[is_valid], held[~is_valid]
    policies = [
        {"policy": "approve all", "split": s, **evaluate(np.ones(len(p), bool), p)}
        for s, p in (("valid", vh), ("test", th))
    ]
    for name, s in scores.items():
        cut = best_cutoff(sweep(s[is_valid], vh), "profit")
        res.cutoffs[name] = cut
        for split, part, m in (("valid", vh, is_valid), ("test", th, ~is_valid)):
            policies.append(
                {"policy": name, "split": split, "cutoff": cut, **evaluate(s[m] <= cut, part)}
            )
    res.policies = pd.DataFrame(policies)
    cutoff = res.cutoffs[f"champion ({res.champion})"]
    approve = pd_cal <= cutoff

    # Explanations (the GBM's exact TreeSHAP) and reason codes for every decline.
    phi, _ = contributions(gbm, held)
    test_mask = ~is_valid
    res.importance = global_importance(
        phi[test_mask], gbm.importance().set_index("feature")["gain_share"]
    )
    res.recovery = recovery(phi[test_mask], held.loc[test_mask, [f"rule_{d}" for d in RULE]])
    declined = ~approve
    codes = reasons.reason_codes(phi[declined], reasons=REASONS)
    codes.insert(0, "split", held.loc[declined, "split"].to_numpy())
    res.reason_counts = (
        codes.groupby("split")["n_reasons"].value_counts().rename("loans").reset_index()
    )
    res.reason_frequency = reasons.frequency(codes[codes["split"] == "test"], REASONS)
    sample = codes[codes["split"] == "test"].sample(15, random_state=0).index
    res.reason_sample = pd.concat(
        [
            held.loc[sample, ["borrower_id"]].assign(pd=pd_cal[sample]),
            reasons.describe(codes.loc[sample], REASONS),
        ],
        axis=1,
    )

    # Fairness on test, for the champion and for the rule of thumb.
    rot = scores["rule of thumb: loan / inflow"] <= res.cutoffs["rule of thumb: loan / inflow"]
    tables, summaries = [], []
    for policy, mask in ((f"champion PD <= {cutoff:.3f}", approve), ("rule of thumb", rot)):
        for attr in ATTRIBUTES:
            t = fairness.group_table(
                th[attr], mask[test_mask], th["default"], pd_cal[test_mask], min_group=min_group
            )
            tables.append(t.assign(policy=policy, attribute=attr))
            summaries.append({"policy": policy, **fairness.summary(t, attr, min_group)})
    res.fairness = pd.concat(tables, ignore_index=True)
    res.fairness_summary = pd.DataFrame(summaries)
    res.scorecard_points = card.points_table()

    res.bundle = {
        "track": "mobile_money (synthetic)",
        "model_name": res.champion,
        "model": champion,
        "explainer": gbm,
        "calibrator": calibrator,
        "cutoff": cutoff,
        "features": FEATURES,
        "reasons": REASONS,
    }
    return res


def score(bundle: dict, features: pd.DataFrame) -> pd.DataFrame:
    """Score feature rows with a saved bundle: calibrated PD, decision and reason codes."""
    pd_cal = bundle["calibrator"].transform(bundle["model"].predict_pd(features))
    out = pd.DataFrame(
        {"pd": pd_cal, "decision": np.where(pd_cal <= bundle["cutoff"], "approve", "decline")},
        index=features.index,
    )
    phi, _ = contributions(bundle["explainer"], features)
    declined = out["decision"] == "decline"
    if declined.any():
        codes = reasons.reason_codes(phi[declined], reasons=bundle["reasons"])
        out = out.join(reasons.describe(codes, bundle["reasons"]))
    if "borrower_id" in features:
        out.insert(0, "borrower_id", features["borrower_id"].to_numpy())
    return out


def save(res: MMResult, reports: Path, models_dir: Path) -> None:
    from crisk.models import plots

    reports.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "mm_metrics": res.metrics,
        "mm_lift": res.lift,
        "mm_gbm_tuning": res.tuning,
        "mm_calibration": res.calibration,
        "mm_policies": res.policies,
        "mm_shap_importance": res.importance,
        "mm_rule_recovery": res.recovery,
        "mm_reason_frequency": res.reason_frequency,
        "mm_reason_counts": res.reason_counts,
        "mm_reason_codes_sample": res.reason_sample,
        "mm_fairness_by_group": res.fairness,
        "mm_fairness_summary": res.fairness_summary,
        "mm_scorecard_points": res.scorecard_points,
        "mm_reason_code_map": reasons.code_map(REASONS),
    }
    for name, t in tables.items():
        t.assign(source="synthetic").to_csv(
            reports / f"{name}.csv", index=False, float_format="%.4f"
        )
    (reports / "mm_champion.txt").write_text(res.champion + "\n", encoding="utf-8")
    plots.shap_importance(
        res.importance,
        reports / "mm_shap_importance.png",
        title="SYNTHETIC thin-file track: what moves the PD (test cohort, mean |SHAP|)",
    )
    plots.rule_recovery(res.recovery, reports / "mm_rule_recovery.png")
    joblib.dump(res.bundle, models_dir / MODEL_FILE)
