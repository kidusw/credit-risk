"""Phase 7 runs: stability of the Lending Club champion by vintage, and reject inference."""

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from crisk.data.build import CATEGORICAL
from crisk.models.features import load_model_table
from crisk.monitor import stability as st

CHAMPION = "gbm_with_pricing"
SCORECARD = "scorecard_with_pricing"
# The champion's largest SHAP features (Phase 5), plus FICO and purpose.
KEY_FEATURES = [
    "sub_grade",
    "int_rate",
    "annual_inc",
    "loan_to_income",
    "acc_open_past_24mths",
    "dti",
    "term_months",
    "fico_range_low",
    "home_ownership",
    "purpose",
]
SCORES = ["champion_pd", "scorecard_points"]
RECENT_QUARTERS = 4  # second reference: the last year of train, after every field existed


@dataclass
class StabilityResult:
    psi: pd.DataFrame  # one row per issue quarter, one column per score/feature
    psi_recent: pd.DataFrame  # the same against the last year of train
    csr: pd.DataFrame  # characteristic x quarter: mean points shift vs train
    latest_csr: pd.DataFrame  # the most recent test quarter, in detail
    summary: pd.DataFrame  # per score/feature: worst test PSI, its quarter and band


def stability_run(model_table: Path, models_dir: Path) -> StabilityResult:
    df = load_model_table(model_table)
    champion = joblib.load(models_dir / f"{CHAMPION}.joblib")
    card = joblib.load(models_dir / f"{SCORECARD}.joblib")
    df["champion_pd"] = champion.predict_pd(df)
    df["scorecard_points"] = card.score(df)
    is_train = df["split"] == "train"

    split_of = df.groupby("issue_quarter", observed=True)["split"].first().astype(str)
    psi = st.psi_by_period(
        df, [*SCORES, *KEY_FEATURES], reference=is_train, categorical=tuple(CATEGORICAL)
    )
    psi.insert(1, "split", psi["issue_quarter"].map(split_of))
    train_quarters = sorted(df.loc[is_train, "issue_quarter"].astype(str).unique())
    recent = df["issue_quarter"].astype(str).isin(train_quarters[-RECENT_QUARTERS:])
    held_rows = df["split"] != "train"
    psi_recent = st.psi_by_period(
        df[held_rows | recent],
        [*SCORES, *KEY_FEATURES],
        reference=recent[held_rows | recent],
        categorical=tuple(CATEGORICAL),
    )
    psi_recent.insert(1, "split", psi_recent["issue_quarter"].map(split_of))

    points = st.characteristic_points(card, df)
    rows = []
    for q in psi["issue_quarter"]:
        mask = df["issue_quarter"].astype(str) == q
        rep = st.characteristic_report(points, is_train, mask).set_index("characteristic")
        rows.append(rep["shift"].rename(q))
    csr = pd.concat(rows, axis=1)
    csr.loc["total score"] = csr.sum()
    csr = csr.rename_axis("characteristic").reset_index()

    test_quarters = psi.loc[psi["split"] == "test", "issue_quarter"]
    latest = test_quarters.max()
    latest_csr = st.characteristic_report(
        points, is_train, df["issue_quarter"].astype(str) == latest
    )
    latest_csr.insert(0, "quarter", latest)

    held = psi[psi["split"] != "train"]
    held_recent = psi_recent[psi_recent["split"] != "train"]
    summary = []
    for c in [*SCORES, *KEY_FEATURES]:
        worst = held.loc[held[c].idxmax()]
        summary.append(
            {
                "variable": c,
                "valid_max_psi": float(held.loc[held["split"] == "valid", c].max()),
                "test_max_psi": float(held.loc[held["split"] == "test", c].max()),
                "worst_quarter": worst["issue_quarter"],
                "band": st.band(float(worst[c])),
                "test_max_psi_vs_recent": float(
                    held_recent.loc[held_recent["split"] == "test", c].max()
                ),
                "band_vs_recent": st.band(
                    float(held_recent.loc[held_recent["split"] == "test", c].max())
                ),
            }
        )
    return StabilityResult(
        psi=psi,
        psi_recent=psi_recent,
        csr=csr,
        latest_csr=latest_csr,
        summary=pd.DataFrame(summary),
    )


def save_stability(res: StabilityResult, reports: Path) -> None:
    from crisk.models import plots

    res.psi.to_csv(reports / "psi_by_quarter.csv", index=False, float_format="%.4f")
    res.psi_recent.to_csv(reports / "psi_vs_recent_train.csv", index=False, float_format="%.4f")
    res.csr.to_csv(reports / "csr_by_quarter.csv", index=False, float_format="%.2f")
    res.latest_csr.to_csv(reports / "csr_latest_quarter.csv", index=False, float_format="%.2f")
    res.summary.to_csv(reports / "stability_summary.csv", index=False, float_format="%.4f")
    # Against the last year of train: the full-train reference is dominated by fields that
    # did not exist before 2012. Term is left out: holdout windows are 36-month by design.
    plots.psi_by_quarter(res.psi_recent, reports / "psi_by_quarter.png", exclude=("term_months",))
