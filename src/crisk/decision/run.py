"""Phase 4: calibrate PDs on validation, expected vs realised loss, profit-based threshold.

Uses the Phase 3 champion (LightGBM + pricing) and the Phase 2 scorecard + pricing, the
model a bank would more likely ship. Every choice (calibration method, PD cutoff, grade
cutoff) is made on validation; test only reports what those choices would have earned.
"""

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from crisk.decision.economics import assign_lgd, expected_loss, lgd_table, loan_economics
from crisk.decision.policy import (
    OBJECTIVES,
    best_cutoff,
    evaluate,
    expected_profit_rule,
    grade_policy,
    margins,
    sweep,
)
from crisk.models import plots
from crisk.models.calibration import (
    METHODS,
    Calibrator,
    calibration_metrics,
    reliability_table,
    select_method,
)
from crisk.models.features import load_model_table

MODELS = ("gbm_with_pricing", "scorecard_with_pricing")


@dataclass
class DecisionResult:
    calibration: pd.DataFrame = None  # model x version x split
    method_cv: pd.DataFrame = None  # out-of-fold comparison within validation
    chosen: dict = field(default_factory=dict)  # model -> calibration method
    by_vintage: pd.DataFrame = None  # calibration + EL vs realised per quarter
    reliability: pd.DataFrame = None
    curves: pd.DataFrame = None  # profit curves, valid and test
    policies: pd.DataFrame = None
    lgd: pd.DataFrame = None
    grade_returns: pd.DataFrame = None  # LC's own pricing, as it played out
    cutoffs: dict = field(default_factory=dict)


def _load(model_table: Path, outcomes: Path) -> pd.DataFrame:
    df = load_model_table(model_table)
    econ = loan_economics(pd.read_parquet(outcomes)).rename(columns={"lgd": "realised_lgd"})
    return df.merge(econ, on="id", validate="one_to_one")


def run(model_table: Path, outcomes: Path, models_dir: Path) -> DecisionResult:
    res = DecisionResult()
    df = _load(model_table, outcomes)
    train = df[df["split"] == "train"]
    held = df[df["split"].isin(["valid", "test"])].reset_index(drop=True)
    is_valid = (held["split"] == "valid").to_numpy()
    y = held["default"].to_numpy()

    # LGD by grade x term from charged-off train loans.
    table = lgd_table(train.rename(columns={"realised_lgd": "lgd"}))
    fallback = float(train.loc[train["default"] == 1, "realised_lgd"].mean())
    res.lgd = table.rename("lgd").reset_index()
    held["lgd_hat"] = assign_lgd(held, table, fallback)

    cal_rows, cv_tables, vintage_rows, rel_rows = [], [], [], []
    for name in MODELS:
        raw = joblib.load(models_dir / f"{name}.joblib").predict_pd(held)
        method, cv = select_method(raw[is_valid], y[is_valid])
        res.chosen[name] = method
        cv_tables.append(cv.reset_index().assign(model=name))
        versions = {"raw": raw}
        for m in METHODS:  # both fitted on validation, reported for transparency
            versions[m] = Calibrator(m).fit(raw[is_valid], y[is_valid]).transform(raw)
        held[f"pd_{name}"] = versions[method]
        for version, p in versions.items():
            for split in ("valid", "test"):
                mask = (held["split"] == split).to_numpy()
                cal_rows.append(
                    {
                        "model": name,
                        "version": version,
                        "split": split,
                        "chosen": version == method,
                        **calibration_metrics(y[mask], p[mask]),
                    }
                )
        for version in ("raw", method):
            p = versions[version]
            test = (held["split"] == "test").to_numpy()
            rel_rows.append(reliability_table(y[test], p[test]).assign(model=name, version=version))
            el = expected_loss(p, held["lgd_hat"], held["funded_amnt"])
            frame = held.assign(pd=p, el=el)
            for (split, q), g in frame.groupby(["split", "issue_quarter"], observed=True):
                vintage_rows.append(
                    {
                        "model": name,
                        "version": "calibrated" if version != "raw" else "raw",
                        "split": split,
                        "issue_quarter": q,
                        "loans": len(g),
                        "default_rate": g["default"].mean(),
                        "mean_pd": g["pd"].mean(),
                        "expected_loss": g["el"].sum(),
                        "realised_loss": g["principal_loss"].sum(),
                        "el_over_realised": g["el"].sum() / g["principal_loss"].sum(),
                    }
                )
    res.calibration = pd.DataFrame(cal_rows)
    res.method_cv = pd.concat(cv_tables, ignore_index=True)
    res.by_vintage = pd.DataFrame(vintage_rows)
    res.reliability = pd.concat(rel_rows, ignore_index=True)

    # Policies: thresholds from validation, applied to test.
    valid, test = held[is_valid], held[~is_valid]
    curves, policies = [], []

    def add(policy: str, objective: str, masks: dict[str, np.ndarray]) -> None:
        for split, part in (("valid", valid), ("test", test)):
            policies.append(
                {
                    "policy": policy,
                    "objective": objective,
                    "split": split,
                    **evaluate(masks[split], part),
                }
            )

    add(
        "approve all (LC's book)",
        "-",
        {s: np.ones(len(p), bool) for s, p in (("valid", valid), ("test", test))},
    )
    grades_valid = grade_policy(valid)
    for objective, col in OBJECTIVES.items():
        worst = grades_valid.loc[grades_valid[col].idxmax(), "worst_grade"]
        res.cutoffs[f"lc_grade_{objective}"] = worst
        allowed = list("ABCDEFG")[: "ABCDEFG".index(worst) + 1]
        add(
            f"LC grades A-{worst}",
            objective,
            {
                s: p["grade"].astype(str).isin(allowed).to_numpy()
                for s, p in (("valid", valid), ("test", test))
            },
        )
    for name in MODELS:
        col = f"pd_{name}"
        curve_valid = sweep(valid[col], valid)
        curves.append(curve_valid.assign(model=name, split="valid"))
        curves.append(sweep(test[col], test).assign(model=name, split="test"))
        for objective in OBJECTIVES:
            cutoff = best_cutoff(curve_valid, objective)
            res.cutoffs[f"{name}_{objective}"] = cutoff
            add(
                f"{name}: PD <= {cutoff:.3f}",
                objective,
                {s: (p[col] <= cutoff).to_numpy() for s, p in (("valid", valid), ("test", test))},
            )
    champion = MODELS[0]
    m = margins(train)
    add(
        f"{champion}: expected profit > 0",
        "untuned",
        {
            s: expected_profit_rule(p[f"pd_{champion}"], p, m)
            for s, p in (("valid", valid), ("test", test))
        },
    )
    res.grade_returns = (
        held.groupby(["split", held["grade"].astype(str)])
        .agg(
            loans=("id", "size"),
            lent=("funded_amnt", "sum"),
            profit=("profit", "sum"),
            bad_rate=("default", "mean"),
        )
        .assign(return_on_lent=lambda g: g["profit"] / g["lent"])
        .reset_index()
    )
    res.curves = pd.concat(curves, ignore_index=True)
    res.policies = pd.DataFrame(policies)
    return res


def save(res: DecisionResult, reports: Path) -> None:
    reports.mkdir(parents=True, exist_ok=True)
    res.calibration.round(5).to_csv(reports / "calibration_metrics.csv", index=False)
    res.method_cv.round(5).to_csv(reports / "calibration_method_cv.csv", index=False)
    res.by_vintage.round(4).to_csv(reports / "expected_vs_realised_loss.csv", index=False)
    res.reliability.round(4).to_csv(reports / "reliability_test.csv", index=False)
    res.curves.round(5).to_csv(reports / "profit_curves.csv", index=False)
    res.policies.round(4).to_csv(reports / "policies.csv", index=False)
    res.lgd.round(4).to_csv(reports / "lgd_by_grade_term.csv", index=False)
    res.grade_returns.round(4).to_csv(reports / "return_by_grade.csv", index=False)
    pd.Series(res.cutoffs | {f"calibration_{k}": v for k, v in res.chosen.items()}).to_csv(
        reports / "decision_choices.csv", header=["value"], index_label="choice"
    )
    plots.reliability(res.reliability, reports / "reliability_test.png")
    test = res.policies[res.policies["split"] == "test"]
    champion = MODELS[0]
    points = [
        (
            test["policy"].str.startswith(champion) & (test["objective"] == "profit"),
            "Champion cutoff: max profit",
            "#2a78d6",
            "P",
        ),
        (
            test["policy"].str.startswith(champion) & (test["objective"] == "return"),
            "Champion cutoff: max return",
            "#2a78d6",
            "D",
        ),
        (
            test["policy"].str.startswith("LC grades") & (test["objective"] == "return"),
            "LC grades A-B",
            "#e87ba4",
            "v",
        ),
        (test["policy"].str.startswith("approve all"), "Approve all (LC's book)", plots.INK_2, "X"),
    ]
    markers = pd.concat(
        [
            test[mask].iloc[[0]].assign(label=label, colour=colour, marker=marker)
            for mask, label, colour, marker in points
        ]
    )
    plots.profit_curves(res.curves, markers, reports / "profit_curves_test.png")
