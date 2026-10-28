"""Export the two champions to safe serving artefacts (no pickle at serving time).

Per track, `models/serving/` gets:

- `<track>.lgb.txt`: the LightGBM booster in LightGBM's own text format, cut at the best
  iteration. It stores the training category lists, so categoricals map the same way.
- `<track>.json`: everything else the API needs, as plain data: the feature list and how
  each is encoded, the calibrator's parameters, the approval cutoff, the score scale,
  risk bands, the reason-code map and the headline metrics.
- `<track>_examples.json`: a few applicants for the dashboard's what-if. For Lending Club
  these are real test-set loans, so the file stays local (models/ is git-ignored).

The API (`crisk.serve.predictor`) reads only these files, with lightgbm, numpy and pandas.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from crisk.data.build import CATEGORICAL
from crisk.explain.reasons import Reason
from crisk.models.calibration import Calibrator
from crisk.models.gbm import ORDINAL, SUB_GRADES, GBMModel
from crisk.models.scorecard import BASE_ODDS, BASE_POINTS, PDO

SERVING_DIR = Path("models/serving")
TRACKS = ("lending_club", "mobile_money")
# Risk bands on the calibrated PD (upper bounds). Shared by both tracks.
RISK_BANDS = [(0.05, "A"), (0.10, "B"), (0.15, "C"), (0.20, "D"), (0.30, "E"), (1.01, "F")]
ORDINAL_LEVELS = {"grade": list("ABCDEFG"), "sub_grade": SUB_GRADES}


def encoding(features: list[str]) -> dict[str, dict]:
    """How the API must encode each feature: as gbm.prepare does at training time."""
    out = {}
    for f in features:
        if f in ORDINAL:
            out[f] = {"type": "ordinal", "levels": ORDINAL_LEVELS[f]}
        elif f in CATEGORICAL:
            out[f] = {"type": "category"}
        else:
            out[f] = {"type": "numeric"}
    return out


def calibration_params(cal: Calibrator) -> dict:
    if cal.method == "platt":
        return {
            "method": "platt",
            "coef": float(cal.model.coef_[0][0]),
            "intercept": float(cal.model.intercept_[0]),
        }
    return {
        "method": "isotonic",
        "x": cal.model.X_thresholds_.tolist(),
        "y": cal.model.y_thresholds_.tolist(),
    }


def _jsonable(v):
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if v is pd.NA or v is None:
        return None
    return v if isinstance(v, (int, str, bool)) else str(v)


def examples_records(df: pd.DataFrame, features: list[str], ref: pd.Series, label: str):
    rows = []
    for r, (_, row) in zip(ref, df[features].iterrows(), strict=True):
        rows.append({"ref": f"{label} {r}", "applicant": {f: _jsonable(row[f]) for f in features}})
    return rows


def export_track(
    out_dir: Path,
    track: str,
    gbm: GBMModel,
    calibrator: Calibrator,
    cutoff: float,
    reasons: list[Reason],
    examples: list[dict],
    info: dict,
) -> Path:
    """Write one track's booster, metadata and examples. Returns the metadata path."""
    if track not in TRACKS:
        raise ValueError(f"unknown track {track!r}")
    out_dir.mkdir(parents=True, exist_ok=True)
    booster = gbm.model.booster_
    model_path = out_dir / f"{track}.lgb.txt"
    booster.save_model(str(model_path), num_iteration=gbm.best_iteration)
    text = model_path.read_text(encoding="utf-8")
    covered = {f for r in reasons for f in r.features}
    missing = sorted(set(gbm.features) - covered)
    if missing:
        raise ValueError(f"features without a reason code: {missing}")
    meta = {
        "track": track,
        "version": f"{track}-{hashlib.sha256(text.encode()).hexdigest()[:10]}",
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "features": list(gbm.features),
        "encoding": encoding(gbm.features),
        "calibration": calibration_params(calibrator),
        "cutoff": float(cutoff),
        "score_scale": {"base_points": BASE_POINTS, "base_odds": BASE_ODDS, "pdo": PDO},
        "risk_bands": RISK_BANDS,
        "reasons": [
            {
                "code": r.code,
                "text": r.text,
                "actionable": r.actionable,
                "features": [f for f in r.features if f in gbm.features],
            }
            for r in reasons
            if any(f in gbm.features for f in r.features)
        ],
        **info,
    }
    path = out_dir / f"{track}.json"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / f"{track}_examples.json").write_text(
        json.dumps(examples, indent=1), encoding="utf-8"
    )
    return path


def _metric_rows(path: Path, **match) -> dict:
    if not path.exists():
        return {}
    t = pd.read_csv(path)
    for k, v in match.items():
        t = t[t[k] == v]
    return {
        row["split"]: {m: round(float(row[m]), 4) for m in ("auc", "ks", "brier") if m in row}
        for _, row in t.iterrows()
    }


def export_lending_club(models_dir: Path, model_table: Path, reports: Path, out_dir: Path) -> Path:
    """Export the real LC champion (Phases 3-5) with its validation-chosen calibration and
    return-maximising cutoff."""
    import joblib

    from crisk.explain.reasons import REASONS
    from crisk.models.features import load_model_table

    gbm: GBMModel = joblib.load(models_dir / "gbm_with_pricing.joblib")
    decision = joblib.load(models_dir / "decision.joblib")
    cal = decision["calibrators"]["gbm_with_pricing"]
    cutoff = decision["cutoffs"]["gbm_with_pricing_return"]
    test = load_model_table(model_table)
    test = test[test["split"] == "test"].reset_index(drop=True)
    p = cal.transform(gbm.predict_pd(test))
    order = np.argsort(p)
    pick = np.r_[order[:: max(1, len(order) // 10)][:10], order[-5:]]  # spread over the PD range
    ex = test.iloc[pick]
    examples = examples_records(ex, gbm.features, ex["id"], "LC test loan")
    info = {
        "model_name": "LightGBM (monotonic) + LC pricing",
        "trained_on": "Lending Club 36-month loans issued 2007-2014Q2 (matured)",
        "synthetic": False,
        "demo": False,
        "cutoff_rule": "return-maximising calibrated-PD cutoff chosen on validation (Phase 4)",
        "currency": "USD",
        "metrics": _metric_rows(reports / "model_metrics.csv", variant="gbm_with_pricing"),
    }
    return export_track(out_dir, "lending_club", gbm, cal, cutoff, REASONS, examples, info)


def export_mobile_money(models_dir: Path, features: Path, reports: Path, out_dir: Path) -> Path:
    """Export the SYNTHETIC thin-file champion (Phase 6)."""
    import joblib

    bundle = joblib.load(models_dir / "mm_model.joblib")
    gbm: GBMModel = bundle["explainer"]
    if bundle["model_name"] != "gbm":
        raise ValueError("the served thin-file model must be the LightGBM champion")
    f = pd.read_parquet(features)
    sample = f.sample(15, random_state=1)
    examples = examples_records(sample, gbm.features, sample["borrower_id"], "synthetic borrower")
    info = {
        "model_name": "LightGBM (monotonic), thin-file wallet features",
        "trained_on": "SYNTHETIC telebirr-style wallet histories, applications Jan-Dec 2024",
        "synthetic": True,
        "demo": False,
        "cutoff_rule": "profit-maximising calibrated-PD cutoff chosen on the Q1 2025 cohort",
        "currency": "ETB",
        "metrics": _metric_rows(reports / "mm_metrics.csv", model="gbm"),
    }
    return export_track(
        out_dir,
        "mobile_money",
        gbm,
        bundle["calibrator"],
        bundle["cutoff"],
        bundle["reasons"],
        examples,
        info,
    )


def export_demo_lending_club(out_dir: Path, n: int = 40_000) -> Path:
    """Train and export the DEMO LC model on synthetic LC-shaped loans (no Kaggle token)."""
    from crisk.explain.reasons import REASONS
    from crisk.serve.demo import FEATURES, train_demo

    gbm, cal, cutoff, test, metrics = train_demo(n)
    sample = test.sample(15, random_state=0)
    examples = examples_records(sample, FEATURES, sample.index.to_series(), "synthetic demo loan")
    info = {
        "model_name": "DEMO: LightGBM (monotonic) on synthetic Lending-Club-shaped loans",
        "trained_on": "SYNTHETIC demo data (set KAGGLE_API_TOKEN for the real Lending Club model)",
        "synthetic": True,
        "demo": True,
        "cutoff_rule": "return-maximising cutoff on the demo validation split",
        "currency": "USD",
        "metrics": metrics,
    }
    return export_track(out_dir, "lending_club", gbm, cal, cutoff, REASONS, examples, info)
