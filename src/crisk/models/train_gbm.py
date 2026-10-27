"""Phase 3: tune the monotonic LightGBM on validation, compare with the scorecards.

Protocol: every configuration is fitted on train with early stopping on validation; the
configuration with the best validation AUC wins; test is scored once, at the end, for every
model. The overall champion is picked on validation AUC only, whatever test says.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score

from crisk.config import get_settings
from crisk.models.compare import compare
from crisk.models.features import feature_sets, load_model_table
from crisk.models.gbm import GBMModel, sample_configs
from crisk.models.plots import auc_by_vintage
from crisk.models.train import BASELINE, SubGradeBaseline, VariantResult, evaluate

EXPERIMENT = "crisk-gbm"


def tune(df: pd.DataFrame, features: list[str], n_configs: int) -> tuple[GBMModel, pd.DataFrame]:
    train = df[df["split"] == "train"]
    valid = df[df["split"] == "valid"]
    rows, best, best_auc = [], None, -1.0
    for i, cfg in enumerate(sample_configs(n_configs)):
        model = GBMModel(features, cfg).fit(train, valid)
        auc = roc_auc_score(valid["default"], model.predict_pd(valid))
        rows.append({"config": i, **cfg, "best_iteration": model.best_iteration, "valid_auc": auc})
        if auc > best_auc:
            best, best_auc = model, auc
    return best, pd.DataFrame(rows).sort_values("valid_auc", ascending=False, ignore_index=True)


def run(model_table: Path, models_dir: Path, n_configs: int = 12, n_boot: int = 200):
    df = load_model_table(model_table)
    train = df[df["split"] == "train"]
    sets = feature_sets(train)

    gbms, tuning = {}, {}
    for variant, features in sets.items():
        gbms[variant], tuning[variant] = tune(df, features, n_configs)

    models = {f"gbm_{v}": m for v, m in gbms.items()}
    for v in sets:
        path = models_dir / f"scorecard_{v}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run `crisk train scorecard` first")
        models[f"scorecard_{v}"] = joblib.load(path)
    models[BASELINE] = SubGradeBaseline().fit(train, train["default"])

    results = {name: evaluate(m, df, name) for name, m in models.items()}
    held_out = df[df["split"].isin(["valid", "test"])]
    scored = held_out[["split", "default"]].copy()
    for name, m in models.items():
        scored[name] = m.predict_pd(held_out)
    pairs = [
        ("gbm_with_pricing", "scorecard_with_pricing"),
        ("gbm_no_pricing", "scorecard_no_pricing"),
        ("gbm_with_pricing", BASELINE),
        ("gbm_no_pricing", BASELINE),
    ]
    lifts = compare(scored, pairs, n_boot=n_boot)
    champion = max(results.values(), key=lambda r: r.metrics.loc["valid", "auc"]).name
    return gbms, tuning, results, lifts, champion


def save(gbms, tuning, results: dict[str, VariantResult], lifts, champion, models_dir, reports):
    models_dir.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    for variant, model in gbms.items():
        joblib.dump(model, models_dir / f"gbm_{variant}.joblib")
        tuning[variant].round(5).to_csv(reports / f"gbm_tuning_{variant}.csv", index=False)
        model.importance().round(5).to_csv(reports / f"gbm_importance_{variant}.csv", index=False)
    all_results = list(results.values())
    pd.concat([r.metrics for r in all_results]).round(4).to_csv(reports / "model_metrics.csv")
    for part in ("by_vintage", "by_grade", "by_term"):
        frame = pd.concat([getattr(r, part) for r in all_results]).round(4)
        frame.to_csv(reports / f"model_{part}.csv", index=False)
        if part == "by_vintage":
            auc_by_vintage(frame, reports / "auc_by_vintage.png")
    lifts.round(4).to_csv(reports / "model_lift.csv", index=False)
    (reports / "champion.txt").write_text(champion + "\n", encoding="utf-8")


def log_mlflow(gbms, tuning, results, lifts, champion, reports: Path) -> None:
    import mlflow

    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    for variant, model in gbms.items():
        name = f"gbm_{variant}"
        with mlflow.start_run(run_name=name):
            mlflow.set_tags({"model": "lightgbm_monotone", "variant": variant})
            mlflow.log_params(
                {
                    **model.params,
                    "best_iteration": model.best_iteration,
                    "features": len(model.features),
                    "monotone_features": sum(c != 0 for c in model.model.monotone_constraints),
                    "configs_tried": len(tuning[variant]),
                    "champion": name == champion,
                }
            )
            for split, row in results[name].metrics.iterrows():
                for metric in ("auc", "gini", "ks", "pr_auc", "brier"):
                    mlflow.log_metric(f"{split}_{metric}", float(row[metric]))
            for r in lifts[lifts["challenger"] == name].itertuples():
                mlflow.log_metric(f"{r.split}_auc_lift_vs_{r.incumbent}", r.lift)
            for f in (f"gbm_tuning_{variant}.csv", f"gbm_importance_{variant}.csv"):
                mlflow.log_artifact(str(reports / f))
