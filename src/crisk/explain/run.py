"""Phase 5: explain the champion, give reasons for every decline, audit fairness, and settle
whether `addr_state` stays in the model.

The policy audited is Phase 4's capital-constrained lender: approve when the calibrated
champion PD is at or below the return-maximising cutoff chosen on validation. Lending Club's
own grade cutoff (A-B, a similar approval rate) is audited alongside as the reference.

**The addr_state rule, fixed before the ablation was run:** geography is the textbook
redlining proxy, state is not something an applicant can act on, and LightGBM's gain on a
50-level categorical is inflated. So state stays only if it earns its place on validation:
at least +0.005 AUC, or at least +0.10 percentage points of return on the amount lent at
the cutoff re-chosen for each model. Test numbers are reported for both models, after the
decision, and play no part in it.
"""

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from crisk.decision.economics import loan_economics
from crisk.decision.policy import best_cutoff, evaluate, sweep
from crisk.decision.run import DECISION_FILE
from crisk.decision.run import MODELS as DECISION_MODELS
from crisk.explain import fairness, reasons
from crisk.explain.contributions import contributions, dependence, global_importance
from crisk.models import plots
from crisk.models.calibration import Calibrator
from crisk.models.features import load_model_table
from crisk.models.gbm import GBMModel

CHAMPION = DECISION_MODELS[0]
STATE = "addr_state"
STATE_MIN_AUC_GAIN = 0.005
STATE_MIN_RETURN_GAIN = 0.001
DEPENDENCE_TOP = 6
SHAP_SAMPLE = 50_000


@dataclass
class ExplainResult:
    importance: pd.DataFrame = None
    dependence: pd.DataFrame = None
    examples: pd.DataFrame = None  # per-applicant explanations
    reason_frequency: pd.DataFrame = None
    reason_counts: pd.DataFrame = None
    reason_sample: pd.DataFrame = None
    fairness: pd.DataFrame = None  # policy x attribute x group
    fairness_summary: pd.DataFrame = None
    state_ablation: pd.DataFrame = None
    state_decision: dict = field(default_factory=dict)
    waterfall: pd.DataFrame = None


def _load(model_table: Path, outcomes: Path) -> pd.DataFrame:
    df = load_model_table(model_table)
    out = pd.read_parquet(outcomes)
    econ = loan_economics(out)[["id", "funded_amnt", "profit"]]
    econ["zip_code"] = out["zip_code"].to_numpy()
    train = df[df["split"] == "train"]
    held = df[df["split"].isin(["valid", "test"])]
    held = held.merge(econ, on="id", validate="one_to_one").reset_index(drop=True)
    return train, held


def _calibrated(model: GBMModel, held: pd.DataFrame, is_valid: np.ndarray, method: str):
    """Raw and calibrated PDs, the calibrator fitted on validation only."""
    raw = model.predict_pd(held)
    cal = Calibrator(method).fit(raw[is_valid], held.loc[is_valid, "default"])
    return raw, cal.transform(raw)


def state_ablation(train, held, champion: GBMModel, champion_pd, method: str) -> pd.DataFrame:
    """Champion vs the same hyperparameters with addr_state toggled, on valid and test."""
    is_valid = (held["split"] == "valid").to_numpy()
    if STATE in champion.features:
        features = [f for f in champion.features if f != STATE]
    else:
        features = [*champion.features, STATE]
    other = GBMModel(features, champion.params).fit(train, held[is_valid])
    _, other_pd = _calibrated(other, held, is_valid, method)
    attrs = fairness.attributes(held)
    rows = []
    for label, has_state, p in (
        ("champion", STATE in champion.features, champion_pd),
        ("toggled", STATE in features, other_pd),
    ):
        cutoff = best_cutoff(sweep(p[is_valid], held[is_valid]), "return")
        for split in ("valid", "test"):
            m = (held["split"] == split).to_numpy()
            part, approve = held[m], p[m] <= cutoff
            row = {
                "model": label,
                "with_addr_state": has_state,
                "split": split,
                "pd_cutoff": cutoff,
                "auc": roc_auc_score(part["default"], p[m]),
                **evaluate(approve, part),
            }
            for attr in ("addr_state", "zip3"):
                t = fairness.group_table(attrs.loc[m, attr], approve, part["default"], p[m])
                s = fairness.summary(t, attr)
                row[f"{attr}_min_air"] = s["min_air"]
                row[f"{attr}_below_0_8"] = s["groups_below_0_8"]
                row[f"{attr}_approval_spread"] = s["approval_spread"]
            rows.append(row)
    return pd.DataFrame(rows)


def decide_state(ablation: pd.DataFrame) -> dict:
    """Apply the pre-registered rule to the validation rows of the ablation."""
    v = ablation[ablation["split"] == "valid"].set_index("with_addr_state")
    auc_gain = float(v.loc[True, "auc"] - v.loc[False, "auc"])
    return_gain = float(v.loc[True, "return_on_lent"] - v.loc[False, "return_on_lent"])
    keep = auc_gain >= STATE_MIN_AUC_GAIN or return_gain >= STATE_MIN_RETURN_GAIN
    return {
        "valid_auc_gain_from_state": auc_gain,
        "valid_return_gain_from_state": return_gain,
        "min_auc_gain": STATE_MIN_AUC_GAIN,
        "min_return_gain": STATE_MIN_RETURN_GAIN,
        "keep_addr_state": keep,
    }


def _examples(phi, base, held, pd_cal, cutoff, test) -> pd.DataFrame:
    """Three test applicants: clearly approved, declined just over the cutoff, clearly declined."""
    t = np.flatnonzero(test)
    p = pd_cal[t]
    picks = {
        "approved (median approved PD)": t[np.argsort(np.abs(p - np.median(p[p <= cutoff])))[0]],
        "declined, just over the cutoff": t[np.argmin(np.where(p > cutoff, p, np.inf))],
        "declined (90th percentile PD)": t[np.argsort(np.abs(p - np.quantile(p, 0.9)))[0]],
    }
    rows = []
    for label, i in picks.items():
        contrib = phi.iloc[i]
        for f in contrib.abs().sort_values(ascending=False).index[:8]:
            rows.append(
                {
                    "applicant": label,
                    "id": held.loc[i, "id"],
                    "pd": pd_cal[i],
                    "decision": "approve" if pd_cal[i] <= cutoff else "decline",
                    "feature": f,
                    "value": str(held.loc[i, f]),
                    "shap": contrib[f],
                    "reason_group": reasons.BY_CODE[reasons.FEATURE_REASON[f]].text,
                    "base_value": base[i],
                }
            )
    return pd.DataFrame(rows)


def run(model_table: Path, outcomes: Path, models_dir: Path, ablate: bool = True):
    res = ExplainResult()
    train, held = _load(model_table, outcomes)
    is_valid = (held["split"] == "valid").to_numpy()
    test = ~is_valid
    champion: GBMModel = joblib.load(models_dir / f"{CHAMPION}.joblib")
    decision = joblib.load(models_dir / DECISION_FILE)
    calibrator: Calibrator = decision["calibrators"][CHAMPION]
    cutoff = decision["cutoffs"][f"{CHAMPION}_return"]
    pd_cal = calibrator.transform(champion.predict_pd(held))
    approve = pd_cal <= cutoff

    # SHAP on every held-out loan: reasons are needed for every decline, not a sample.
    phi, base = contributions(champion, held)
    gain = champion.importance().set_index("feature")["gain_share"]
    res.importance = global_importance(phi[test], gain)
    sample = held[test].sample(min(SHAP_SAMPLE, int(test.sum())), random_state=0).index
    top = [
        f
        for f in res.importance["feature"]
        if f not in ("grade", "sub_grade")  # ordinal copies of the same signal
    ][:DEPENDENCE_TOP]
    res.dependence = pd.concat(
        [dependence(phi.loc[sample], held.loc[sample], f) for f in top], ignore_index=True
    )
    res.examples = _examples(phi, base, held, pd_cal, cutoff, test)

    # Reason codes for every declined applicant, valid and test.
    declined = ~approve
    codes = reasons.reason_codes(phi[declined])
    codes.insert(0, "split", held.loc[declined, "split"].to_numpy())
    res.reason_counts = (
        codes.groupby("split")["n_reasons"].value_counts().rename("loans").reset_index()
    )
    res.reason_frequency = reasons.frequency(codes[codes["split"] == "test"])
    sample_ids = codes[codes["split"] == "test"].sample(20, random_state=0).index
    res.reason_sample = pd.concat(
        [
            held.loc[sample_ids, ["id"]].assign(pd=pd_cal[sample_ids]),
            reasons.describe(codes.loc[sample_ids]),
        ],
        axis=1,
    )
    decline_id = res.examples.loc[res.examples["decision"] == "decline", "id"].iloc[0]
    i = int(np.flatnonzero(held["id"].to_numpy() == decline_id)[0])
    res.waterfall = (
        reasons.grouped(phi.iloc[[i]])
        .T.set_axis(["shap"], axis=1)
        .assign(text=lambda d: d.index.map(lambda c: reasons.BY_CODE[c].text))
        .assign(base_value=base[i], logit=base[i] + phi.iloc[i].sum(), pd=pd_cal[i])
        .rename_axis("code")
        .reset_index()
    )

    # Fairness on test: the champion's policy and LC's grade cutoff at a similar approval rate.
    attrs = fairness.attributes(held)
    lc_worst = decision["cutoffs"]["lc_grade_return"]
    lc_allowed = list("ABCDEFG")[: "ABCDEFG".index(lc_worst) + 1]
    policies = {
        f"champion PD <= {cutoff:.3f}": approve,
        f"LC grades A-{lc_worst}": held["grade"].astype(str).isin(lc_allowed).to_numpy(),
    }
    tables, summaries = [], []
    for policy, mask in policies.items():
        for attr in attrs.columns:
            t = fairness.group_table(
                attrs.loc[test, attr], mask[test], held.loc[test, "default"], pd_cal[test]
            )
            tables.append(t.assign(policy=policy, attribute=attr))
            summaries.append({"policy": policy, **fairness.summary(t, attr)})
    res.fairness = pd.concat(tables, ignore_index=True)
    res.fairness_summary = pd.DataFrame(summaries)

    if ablate:
        method = calibrator.method
        res.state_ablation = state_ablation(train, held, champion, pd_cal, method)
        res.state_decision = decide_state(res.state_ablation)
    return res


def save(res: ExplainResult, reports: Path) -> None:
    reports.mkdir(parents=True, exist_ok=True)
    res.importance.round(5).to_csv(reports / "shap_importance.csv", index=False)
    res.dependence.round(5).to_csv(reports / "shap_dependence.csv", index=False)
    res.examples.round(4).to_csv(reports / "shap_examples.csv", index=False)
    res.reason_frequency.round(4).to_csv(reports / "reason_frequency.csv", index=False)
    res.reason_counts.to_csv(reports / "reason_counts.csv", index=False)
    res.reason_sample.round(4).to_csv(reports / "reason_codes_sample.csv", index=False)
    reasons.code_map().to_csv(reports / "reason_code_map.csv", index=False)
    res.fairness.round(4).to_csv(reports / "fairness_by_group.csv", index=False)
    res.fairness_summary.round(4).to_csv(reports / "fairness_summary.csv", index=False)
    plots.shap_importance(res.importance, reports / "shap_importance.png")
    plots.shap_dependence(res.dependence, reports / "shap_dependence.png")
    plots.reason_waterfall(res.waterfall, reports / "reason_waterfall.png")
    if res.state_ablation is not None:
        res.state_ablation.round(5).to_csv(reports / "addr_state_ablation.csv", index=False)
        pd.Series(res.state_decision).to_csv(
            reports / "addr_state_decision.csv", header=["value"], index_label="item"
        )
