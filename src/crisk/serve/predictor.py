"""Score applicants from the exported artefacts: calibrated PD, points, band, decision, reasons.

Needs only lightgbm, numpy and pandas, and reads only text files written by
`crisk.serve.export`, so the API image carries no training code and unpickles nothing.

The encoding reproduces training exactly: Lending Club grades become ordinals, other
categoricals become pandas categories (the booster maps them with its stored category
lists) and the rest float32. For Lending Club the API also derives what `features.clean`
derives at training time: `loan_to_income`, and DTI sentinels (-1, 999) as missing.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from crisk.explain import reasons as rc


@dataclass
class Scored:
    pd: float
    score: int
    risk_band: str
    decision: str
    reasons: list[dict]
    missing_features: list[str]


class TrackModel:
    def __init__(self, meta: dict, booster, examples: list[dict] | None = None):
        self.meta = meta
        self.booster = booster
        self.examples = examples or []
        self.features: list[str] = meta["features"]
        self.reasons = [
            rc.Reason(r["code"], r["text"], tuple(r["features"]), r["actionable"])
            for r in meta["reasons"]
        ]
        self.by_code = {r.code: r for r in self.reasons}

    @classmethod
    def load(cls, directory: Path, track: str) -> "TrackModel":
        import lightgbm as lgb

        meta = json.loads((directory / f"{track}.json").read_text(encoding="utf-8"))
        booster = lgb.Booster(model_file=str(directory / f"{track}.lgb.txt"))
        ex_path = directory / f"{track}_examples.json"
        examples = json.loads(ex_path.read_text(encoding="utf-8")) if ex_path.exists() else []
        return cls(meta, booster, examples)

    @property
    def track(self) -> str:
        return self.meta["track"]

    # --- encoding ----------------------------------------------------------------------

    def derive(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.track != "lending_club":
            return df
        df = df.copy()
        if "dti" in df:
            dti = pd.to_numeric(df["dti"], errors="coerce")
            df["dti"] = dti.where((dti >= 0) & (dti < 999))
        if "loan_to_income" in self.features and df.get("loan_to_income") is None:
            income = pd.to_numeric(df.get("annual_inc"), errors="coerce")
            amount = pd.to_numeric(df.get("loan_amnt"), errors="coerce")
            df["loan_to_income"] = amount / income.where(income > 0)
        if "grade" in self.features and "sub_grade" in df and df.get("grade") is None:
            df["grade"] = df["sub_grade"].astype(str).str[0]
        return df

    def prepare(self, applicants: list[dict]) -> tuple[pd.DataFrame, list[list[str]]]:
        raw = self.derive(pd.DataFrame(applicants))
        records = raw.to_dict("records")
        missing = [[f for f in self.features if f not in a or pd.isna(a[f])] for a in records]
        X = pd.DataFrame(index=raw.index)
        for f in self.features:
            enc = self.meta["encoding"][f]
            col = raw[f] if f in raw else pd.Series(np.nan, index=raw.index)
            if enc["type"] == "ordinal":
                X[f] = (
                    col.astype(str).map({v: i for i, v in enumerate(enc["levels"])}).astype(float)
                )
            elif enc["type"] == "category":
                X[f] = col.astype("object").where(col.notna(), None).astype("category")
            else:
                X[f] = pd.to_numeric(col, errors="coerce").astype("float32")
        return X, missing

    # --- scoring -----------------------------------------------------------------------

    def calibrate(self, raw: np.ndarray) -> np.ndarray:
        c = self.meta["calibration"]
        if c["method"] == "platt":
            eps = 1e-6
            p = np.clip(raw, eps, 1 - eps)
            z = c["intercept"] + c["coef"] * np.log(p / (1 - p))
            return 1 / (1 + np.exp(-z))
        return np.interp(raw, c["x"], c["y"])

    def points(self, p: float) -> int:
        s = self.meta["score_scale"]
        factor = s["pdo"] / math.log(2)
        offset = s["base_points"] - factor * math.log(s["base_odds"])
        p = min(max(p, 1e-6), 1 - 1e-6)
        return round(offset + factor * math.log((1 - p) / p))

    def band(self, p: float) -> str:
        return next(name for upper, name in self.meta["risk_bands"] if p < upper)

    def score(self, applicants: list[dict]) -> list[Scored]:
        X, missing = self.prepare(applicants)
        raw = self.booster.predict(X)
        pd_cal = self.calibrate(raw)
        phi = self.booster.predict(X, pred_contrib=True)[:, :-1]
        phi = pd.DataFrame(phi, columns=self.features)
        out = []
        for i, p in enumerate(pd_cal):
            decline = p > self.meta["cutoff"]
            reasons = self._reasons(phi.iloc[[i]]) if decline else []
            out.append(
                Scored(
                    pd=float(p),
                    score=self.points(float(p)),
                    risk_band=self.band(float(p)),
                    decision="decline" if decline else "approve",
                    reasons=reasons,
                    missing_features=missing[i],
                )
            )
        return out

    def _reasons(self, phi_row: pd.DataFrame) -> list[dict]:
        try:
            codes = rc.reason_codes(phi_row, reasons=self.reasons).iloc[0]
        except ValueError:  # nothing pushed this PD up (possible only right at the cutoff)
            return []
        out = []
        for k in range(1, rc.MAX_REASONS + 1):
            code = codes[f"reason_{k}"]
            if code is None or (isinstance(code, float) and np.isnan(code)):
                continue
            r = self.by_code[code]
            out.append(
                {
                    "code": r.code,
                    "text": r.text,
                    "actionable": r.actionable,
                    "weight": round(float(codes[f"weight_{k}"]), 4),
                }
            )
        return out


def load_models(directory: Path) -> dict[str, TrackModel]:
    """Every track exported to `directory` (a missing track is simply not served)."""
    return {
        t: TrackModel.load(directory, t)
        for t in ("lending_club", "mobile_money")
        if (directory / f"{t}.json").exists()
    }
