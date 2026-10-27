"""Exact TreeSHAP contributions for the LightGBM models.

LightGBM implements TreeSHAP itself (`pred_contrib=True`, the algorithm of Lundberg et al.
that the `shap` package uses for trees), so no extra dependency is needed and the API image
can explain a decision with the same library that scores it. Contributions are additive in
log-odds: base value + sum over features = the model's raw logit, for every loan.

Calibration doesn't change the explanation. Platt is `logit(PD_cal) = a + b * logit(PD)`
with b > 0, so every contribution is scaled by the same b and the order of reasons is the
same before and after calibration.
"""

import numpy as np
import pandas as pd

from crisk.models.gbm import GBMModel, prepare


def contributions(model: GBMModel, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Per-loan SHAP values (log-odds, one column per feature) and the base value."""
    raw = model.model.predict(
        prepare(df, model.features), num_iteration=model.best_iteration, pred_contrib=True
    )
    phi = pd.DataFrame(raw[:, :-1], columns=model.features, index=df.index)
    return phi, raw[:, -1]


def global_importance(phi: pd.DataFrame, gain_share: pd.Series | None = None) -> pd.DataFrame:
    """Mean |SHAP| per feature, its share of the total, and (optionally) LightGBM's gain share."""
    imp = phi.abs().mean().rename("mean_abs_shap").to_frame()
    imp["shap_share"] = imp["mean_abs_shap"] / imp["mean_abs_shap"].sum()
    imp["mean_shap"] = phi.mean()
    if gain_share is not None:
        imp["gain_share"] = gain_share.reindex(imp.index)
    return imp.sort_values("mean_abs_shap", ascending=False).rename_axis("feature").reset_index()


def dependence(phi: pd.DataFrame, df: pd.DataFrame, feature: str, bins: int = 20) -> pd.DataFrame:
    """Mean SHAP of `feature` per quantile bin of its value (missing values as their own bin).

    Categorical features are summarised per level instead.
    """
    value = df[feature]
    frame = pd.DataFrame({"value": value, "shap": phi[feature]})
    if isinstance(value.dtype, pd.CategoricalDtype) or value.dtype == object:
        g = frame.groupby(frame["value"].astype(str), observed=True)
        out = g.agg(n=("shap", "size"), mean_shap=("shap", "mean")).reset_index()
        return out.rename(columns={"value": "level"}).assign(feature=feature)
    numeric = frame["value"].astype("float64")
    present = frame[numeric.notna()].assign(value=numeric[numeric.notna()])
    present["bin"] = pd.qcut(present["value"], bins, duplicates="drop")
    out = (
        present.groupby("bin", observed=True)
        .agg(n=("shap", "size"), value=("value", "median"), mean_shap=("shap", "mean"))
        .reset_index(drop=True)
    )
    missing = frame[numeric.isna()]
    if len(missing):
        out = pd.concat(
            [
                out,
                pd.DataFrame(
                    {"n": [len(missing)], "value": [np.nan], "mean_shap": [missing["shap"].mean()]}
                ),
            ],
            ignore_index=True,
        )
    return out.assign(feature=feature)
