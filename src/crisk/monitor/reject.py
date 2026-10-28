"""Reject inference on Lending Club's rejected applications, on the columns both files share.

Outcomes exist only for approved loans, so any model learns the risk of *LC's approved*
applicants. Reject inference guesses what the rejected ones would have done and refits on
everyone. Two standard methods, both starting from a known-good-bad (KGB) model fitted on
accepted loans only:

- **Fuzzy augmentation:** each reject enters twice, once as bad with weight PD and once as
  good with weight 1 - PD (the KGB's PD).
- **Parcelling:** rejects are banded by KGB PD; each band's bad rate on accepted loans is
  inflated by `PARCEL_FACTOR` (rejects in a band are assumed riskier than accepts in it,
  because LC saw more than these columns) and rejects are labelled bad at that rate.

**What can't be checked:** whether either guess is right. That needs the outcomes of
loans that were never made. Accuracy on accepted loans can be measured; the effect on
rejected applicants can only be described.

**Scope.** The shared columns are loan amount, risk score and DTI (employment length is
coded "< 1 year" for most rejects, a field default rather than tenure, so it is left out;
state and zip stay out as in Phase 5). Rejects outside the range LC ever approved are
*policy rejects* (risk score below 660 or missing, DTI over 40, amount outside
1,000-35,000): no model trained on accepted loans has data there, so they are not
inferred and stay declined. Rejects are weighted so that, within each window,
rejects : accepts matches LC's real ratio.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from crisk.models.features import load_model_table
from crisk.monitor import stability as st

SHARED = ["fico_range_low", "dti", "loan_amnt"]
MONOTONE = {"fico_range_low": -1, "dti": +1, "loan_amnt": 0}
FICO_FLOOR = 660
DTI_CAP = 40
AMOUNT_RANGE = (1_000, 35_000)
PARCEL_FACTOR = 2.0
PARCEL_BANDS = 10
SEED = 0
PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "n_estimators": 2000,
    "num_leaves": 15,
    "min_child_samples": 500,
    "verbose": -1,
    "n_jobs": 6,
    "random_state": SEED,
}


def in_range(r: pd.DataFrame) -> pd.Series:
    """Rejects inside the range LC ever approved (the rest are policy rejects)."""
    return (
        r["fico_range_low"].between(FICO_FLOOR, 850)
        & r["dti"].le(DTI_CAP)
        & r["loan_amnt"].between(*AMOUNT_RANGE)
    )


@dataclass
class WeightedGBM:
    """LightGBM on the shared columns with sample weights and monotone constraints."""

    model: object = None
    best_iteration: int = 0

    def fit(self, X, y, w, X_valid, y_valid) -> "WeightedGBM":
        import lightgbm as lgb

        self.model = lgb.LGBMClassifier(
            **PARAMS,
            monotone_constraints=[MONOTONE[c] for c in SHARED],
            monotone_constraints_method="advanced",
        )
        self.model.fit(
            X[SHARED],
            y,
            sample_weight=w,
            eval_set=[(X_valid[SHARED], y_valid)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        self.best_iteration = int(self.model.best_iteration_)
        return self

    def predict_pd(self, X) -> np.ndarray:
        return self.model.predict_proba(X[SHARED], num_iteration=self.best_iteration)[:, 1]


def fuzzy_rows(rejects: pd.DataFrame, p: np.ndarray, weight: float) -> pd.DataFrame:
    bad = rejects[SHARED].assign(default=1, w=weight * p)
    good = rejects[SHARED].assign(default=0, w=weight * (1 - p))
    return pd.concat([bad, good], ignore_index=True)


def parcel_rows(
    rejects, p, weight, accepted_p, accepted_y, rng
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Label rejects bad at their KGB band's accepted bad rate x PARCEL_FACTOR."""
    edges = np.unique(np.quantile(accepted_p, np.linspace(0, 1, PARCEL_BANDS + 1))[1:-1])
    acc_band = np.digitize(accepted_p, edges)
    band_rate = pd.Series(accepted_y).groupby(acc_band).mean()
    rej_band = np.digitize(p, edges)
    rate = np.minimum(0.95, PARCEL_FACTOR * band_rate.reindex(rej_band).to_numpy())
    labelled = rejects[SHARED].assign(default=(rng.random(len(p)) < rate).astype(int), w=weight)
    bands = pd.DataFrame(
        {
            "band": band_rate.index,
            "accepted_bad_rate": band_rate.to_numpy(),
            "parcel_bad_rate": np.minimum(0.95, PARCEL_FACTOR * band_rate.to_numpy()),
            "rejects": pd.Series(rej_band).value_counts().reindex(band_rate.index, fill_value=0),
        }
    )
    return labelled, bands


@dataclass
class RejectResult:
    population: pd.DataFrame = None  # counts and weights per window
    models: pd.DataFrame = None  # AUC on accepted loans, mean PDs
    decisions: pd.DataFrame = None  # through-the-door approval at LC's own rate
    flips: pd.DataFrame = None  # decisions that change vs the KGB model
    parcel_bands: pd.DataFrame = None
    extras: dict = field(default_factory=dict)


def _weights(accepted_model: pd.DataFrame, accepted_all: pd.Series, rejects: pd.DataFrame):
    """Per window: reject weight so rejects : accepts matches LC's real ratio."""
    rows = []
    for split in ("train", "valid", "test"):
        n_model = int((accepted_model["split"] == split).sum())
        n_all = int(accepted_all.get(split, 0))
        r = rejects[rejects["split"] == split]
        n_in = int(r["in_range"].sum())
        rows.append(
            {
                "split": split,
                "accepted_all": n_all,
                "accepted_modelled": n_model,
                "rejected": len(r),
                "policy_rejects": len(r) - n_in,
                "in_range_rejects": n_in,
                "lc_approval_rate": n_all / (n_all + len(r)),
                "reject_weight": n_model / n_all,
            }
        )
    return pd.DataFrame(rows).set_index("split")


def _approve_share(p: np.ndarray, w: np.ndarray, share: float) -> np.ndarray:
    """Approve the lowest-PD applicants until `share` of the total weight is approved."""
    order = np.argsort(p, kind="stable")
    cum = np.cumsum(w[order]) / w.sum()
    approve = np.zeros(len(p), dtype=bool)
    approve[order[cum <= share]] = True
    return approve


def run(model_table: Path, rejected: Path, accepted_issue_split: pd.Series) -> RejectResult:
    """`accepted_issue_split`: the split of every accepted application (any term or status)."""
    res = RejectResult()
    acc = load_model_table(model_table)[["split", "default", *SHARED]]
    rej = pd.read_parquet(rejected, columns=["split", *SHARED])
    rej["in_range"] = in_range(rej)
    pop = _weights(acc, accepted_issue_split.value_counts(), rej)
    res.population = pop.reset_index()

    a = {s: acc[acc["split"] == s].reset_index(drop=True) for s in ("train", "valid", "test")}
    r = {s: rej[(rej["split"] == s) & rej["in_range"]].reset_index(drop=True) for s in a}
    w_train = pop.loc["train", "reject_weight"]
    ones = np.ones(len(a["train"]))

    kgb = WeightedGBM().fit(
        a["train"], a["train"]["default"], ones, a["valid"], a["valid"]["default"]
    )
    p_rej = kgb.predict_pd(r["train"])
    p_acc = kgb.predict_pd(a["train"])

    fuzzy = pd.concat(
        [
            a["train"][SHARED].assign(default=a["train"]["default"], w=1.0),
            fuzzy_rows(r["train"], p_rej, w_train),
        ],
        ignore_index=True,
    )
    parcel, res.parcel_bands = parcel_rows(
        r["train"],
        p_rej,
        w_train,
        p_acc,
        a["train"]["default"].to_numpy(),
        np.random.default_rng(SEED),
    )
    parcel = pd.concat(
        [a["train"][SHARED].assign(default=a["train"]["default"], w=1.0), parcel], ignore_index=True
    )
    models = {"kgb (accepted only)": kgb}
    for name, data in (("fuzzy augmentation", fuzzy), ("parcelling", parcel)):
        models[name] = WeightedGBM().fit(
            data, data["default"], data["w"], a["valid"], a["valid"]["default"]
        )

    rows, decisions, flips = [], [], []
    for split in ("valid", "test"):
        acc_s, rej_s = a[split], r[split]
        w_r = pop.loc[split, "reject_weight"]
        # Through the door, in range: accepted loans (weight 1) and in-range rejects. Policy
        # rejects are declined by every model, so each model approves as many applicants as
        # LC did, all from the in-range population.
        n_policy = pop.loc[split, "policy_rejects"] * w_r
        in_share = len(acc_s) / (len(acc_s) + len(rej_s) * w_r)
        weights = np.r_[np.ones(len(acc_s)), np.full(len(rej_s), w_r)]
        is_rej = np.r_[np.zeros(len(acc_s), bool), np.ones(len(rej_s), bool)]
        y_acc = acc_s["default"].to_numpy()
        approved = {}
        for name, m in models.items():
            pa_, pr_ = m.predict_pd(acc_s), m.predict_pd(rej_s)
            rows.append(
                {
                    "model": name,
                    "split": split,
                    "accepted_auc": roc_auc_score(y_acc, pa_),
                    "accepted_mean_pd": float(pa_.mean()),
                    "accepted_default_rate": float(y_acc.mean()),
                    "reject_mean_pd": float(pr_.mean()),
                    "best_iteration": m.best_iteration,
                }
            )
            p_all = np.r_[pa_, pr_]
            appr = _approve_share(p_all, weights, min(in_share, 1.0))
            approved[name] = appr
            a_acc, a_rej = appr[~is_rej], appr[is_rej]
            decisions.append(
                {
                    "model": name,
                    "split": split,
                    "in_range_approval_share": float(np.average(appr, weights=weights)),
                    "accepted_approved": float(a_acc.mean()),
                    "rejects_approved": float(a_rej.mean()),
                    "approved_slots_to_rejects": float(
                        (weights * appr * is_rej).sum() / (weights * appr).sum()
                    ),
                    "bad_rate_approved_accepted": float(y_acc[a_acc].mean()),
                    "expected_bad_rate_approved_rejects": float(pr_[a_rej].mean())
                    if a_rej.any()
                    else np.nan,
                    "policy_reject_weight": float(n_policy),
                }
            )
        base = approved["kgb (accepted only)"]
        for name, appr in approved.items():
            flips.append(
                {
                    "model": name,
                    "split": split,
                    "flipped_share": float(np.average(appr != base, weights=weights)),
                    "newly_approved_rejects": float((weights * (appr & ~base) * is_rej).sum()),
                    "newly_declined_accepted": float(((~appr & base) & ~is_rej).sum()),
                    "newly_approved_accepted": float(((appr & ~base) & ~is_rej).sum()),
                    "newly_declined_rejects": float((weights * (~appr & base) * is_rej).sum()),
                }
            )
        # Approval-rate change at the KGB's own PD cutoff (the plan's metric).
        kgb_p = np.r_[
            models["kgb (accepted only)"].predict_pd(acc_s),
            models["kgb (accepted only)"].predict_pd(rej_s),
        ]
        kgb_cut = float(kgb_p[base].max())
        for name, m in models.items():
            p_all = np.r_[m.predict_pd(acc_s), m.predict_pd(rej_s)]
            res.extras.setdefault("fixed_cutoff", []).append(
                {
                    "model": name,
                    "split": split,
                    "kgb_cutoff": kgb_cut,
                    "approval_share_at_kgb_cutoff": float(
                        np.average(p_all <= kgb_cut, weights=weights)
                    ),
                }
            )
    # How different are rejects from accepts on what both files record? PSI of each shared
    # column and of the KGB score, rejects (in range, and all) against accepted loans.
    shift = []
    for split in ("valid", "test"):
        all_rej = rej[rej["split"] == split]
        for name, part in (("in-range rejects", r[split]), ("all rejects", all_rej)):
            row = {"split": split, "population": name, "n": len(part)}
            for c in SHARED:
                row[c] = st.psi(a[split][c], part[c])
            if name == "in-range rejects":
                row["kgb_score"] = st.psi(
                    pd.Series(kgb.predict_pd(a[split])), pd.Series(kgb.predict_pd(part))
                )
            shift.append(row)
    res.extras["reject_psi"] = pd.DataFrame(shift)
    res.models = pd.DataFrame(rows)
    res.decisions = pd.DataFrame(decisions)
    res.flips = pd.DataFrame(flips)
    res.extras["fixed_cutoff"] = pd.DataFrame(res.extras["fixed_cutoff"])
    return res


def accepted_issue_split(accepted_parquet: Path) -> pd.Series:
    """The time-split window of every accepted application, whatever its term or status."""
    from crisk.data.build import assign_split

    issue = pd.read_parquet(accepted_parquet, columns=["issue_d"])["issue_d"].astype(str)
    split = assign_split(pd.to_datetime(issue, format="%b-%Y", errors="coerce"))
    return split.dropna().astype(str)


def save(res: RejectResult, reports: Path) -> None:
    res.population.to_csv(reports / "reject_population.csv", index=False, float_format="%.4f")
    res.models.to_csv(reports / "reject_models.csv", index=False, float_format="%.4f")
    res.decisions.to_csv(reports / "reject_decisions.csv", index=False, float_format="%.4f")
    res.flips.to_csv(reports / "reject_flips.csv", index=False, float_format="%.1f")
    res.parcel_bands.to_csv(reports / "reject_parcel_bands.csv", index=False, float_format="%.4f")
    res.extras["fixed_cutoff"].to_csv(
        reports / "reject_fixed_cutoff.csv", index=False, float_format="%.4f"
    )
    res.extras["reject_psi"].to_csv(reports / "reject_psi.csv", index=False, float_format="%.4f")
