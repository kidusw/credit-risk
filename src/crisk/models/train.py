"""Fit the scorecard variants on train, evaluate on valid and test vintages, log to MLflow.

The champion is chosen on validation AUC only; test is scored once and reported as is.
"""

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from crisk.config import get_settings
from crisk.models.features import feature_sets, load_model_table
from crisk.models.metrics import binary_metrics, grouped_metrics
from crisk.models.scorecard import BASE_ODDS, BASE_POINTS, MAX_CORR, MIN_IV, PDO, Scorecard

EXPERIMENT = "crisk-scorecard"
EVAL_SPLITS = ("train", "valid", "test")


@dataclass
class VariantResult:
    name: str
    scorecard: "Scorecard | SubGradeBaseline"
    metrics: pd.DataFrame  # one row per split
    by_vintage: pd.DataFrame
    by_grade: pd.DataFrame
    by_term: pd.DataFrame


def evaluate(card: Scorecard, df: pd.DataFrame, name: str) -> VariantResult:
    scored = df[["split", "issue_quarter", "grade", "term_months", "default"]].copy()
    scored["pd"] = card.predict_pd(df)
    evals = scored[scored["split"].isin(EVAL_SPLITS)]
    metrics = (
        pd.DataFrame(
            [
                {"split": s, **binary_metrics(g["default"], g["pd"])}
                for s, g in evals.groupby("split", observed=True)
            ]
        )
        .set_index("split")
        .loc[list(EVAL_SPLITS)]
    )
    held_out = evals[evals["split"] != "train"]
    by_vintage = grouped_metrics(held_out, "issue_quarter").assign(variant=name)
    by_grade = grouped_metrics(held_out, "grade").assign(variant=name)
    # Per term on every split: train is the only one with 60-month loans.
    by_term = pd.concat(
        [
            grouped_metrics(evals[evals["split"] == s], "term_months").assign(split=s, variant=name)
            for s in EVAL_SPLITS
        ]
    )
    return VariantResult(name, card, metrics.assign(variant=name), by_vintage, by_grade, by_term)


class SubGradeBaseline:
    """Baseline PD: the train default rate of the loan's Lending Club sub-grade."""

    def fit(self, df: pd.DataFrame, y: pd.Series) -> "SubGradeBaseline":
        self.rates = y.groupby(df["sub_grade"].astype(str)).mean()
        self.overall = float(y.mean())
        return self

    def predict_pd(self, df: pd.DataFrame):
        return df["sub_grade"].astype(str).map(self.rates).fillna(self.overall).to_numpy()


BASELINE = "baseline_lc_sub_grade"


def train_variants(model_table: Path) -> tuple[list[VariantResult], VariantResult, str]:
    df = load_model_table(model_table)
    train = df[df["split"] == "train"]
    results = []
    for name, features in feature_sets(train).items():
        card = Scorecard(features).fit(train, train["default"])
        results.append(evaluate(card, df, name))
    baseline = evaluate(SubGradeBaseline().fit(train, train["default"]), df, BASELINE)
    champion = max(results, key=lambda r: r.metrics.loc["valid", "auc"]).name
    return results, baseline, champion


def save(
    results: list[VariantResult],
    baseline: VariantResult,
    champion: str,
    models_dir: Path,
    reports_dir: Path,
):
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        joblib.dump(r.scorecard, models_dir / f"scorecard_{r.name}.joblib")
        points = r.scorecard.points_table()
        points.to_csv(reports_dir / f"scorecard_{r.name}.csv", index=False)
        (reports_dir / f"scorecard_{r.name}.md").write_text(
            points_markdown(points, r.name), encoding="utf-8"
        )
        r.scorecard.iv.round(4).to_csv(reports_dir / f"iv_{r.name}.csv", index_label="feature")
    everything = [*results, baseline]
    metrics = pd.concat([r.metrics for r in everything]).round(4)
    metrics.to_csv(reports_dir / "scorecard_metrics.csv")
    for part in ("by_vintage", "by_grade", "by_term"):
        frame = pd.concat([getattr(r, part) for r in everything]).round(4)
        frame.to_csv(reports_dir / f"scorecard_{part}.csv", index=False)
    (reports_dir / "scorecard_champion.txt").write_text(champion + "\n", encoding="utf-8")


def points_markdown(points: pd.DataFrame, name: str) -> str:
    """The scorecard as a bank would print it: one section per feature, bin -> points."""
    lines = [
        f"# Scorecard: {name}",
        "",
        f"Score = sum of points. {BASE_POINTS} points = {BASE_ODDS}:1 good:bad odds; "
        f"every {PDO} points doubles the odds. Empty bins (count 0) score as neutral WoE.",
    ]
    for feature, rows in points.groupby("feature", sort=False):
        lines += [
            "",
            f"## {feature}",
            "",
            "| Bin | Share | Default rate | Points |",
            "|---|---:|---:|---:|",
        ]
        for r in rows.itertuples():
            lines.append(f"| {r.bin} | {r.share:.1%} | {r.default_rate:.1%} | {r.points} |")
    return "\n".join(lines) + "\n"


def _log_metrics(r: VariantResult) -> None:
    import mlflow

    for split, row in r.metrics.iterrows():
        for metric in ("auc", "gini", "ks", "pr_auc", "brier"):
            mlflow.log_metric(f"{split}_{metric}", float(row[metric]))


def log_mlflow(
    results: list[VariantResult], baseline: VariantResult, champion: str, reports_dir: Path
) -> None:
    import mlflow

    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=BASELINE):
        mlflow.set_tags({"model": "sub_grade_default_rate", "variant": BASELINE})
        _log_metrics(baseline)
    for r in results:
        with mlflow.start_run(run_name=f"scorecard-{r.name}"):
            mlflow.set_tags({"model": "woe_logistic_scorecard", "variant": r.name})
            mlflow.log_params(
                {
                    "variant": r.name,
                    "candidate_features": len(r.scorecard.features),
                    "selected_features": len(r.scorecard.selected),
                    "min_iv": MIN_IV,
                    "max_corr": MAX_CORR,
                    "min_bin_size": r.scorecard.min_bin_size,
                    "max_n_bins": r.scorecard.max_n_bins,
                    "points": f"{BASE_POINTS} at {BASE_ODDS}:1, PDO {PDO}",
                    "champion": r.name == champion,
                }
            )
            _log_metrics(r)
            for f in (f"scorecard_{r.name}.csv", f"iv_{r.name}.csv"):
                mlflow.log_artifact(str(reports_dir / f))
