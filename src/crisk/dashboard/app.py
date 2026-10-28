"""Streamlit dashboard: applicant what-if on both tracks, portfolio, fairness and models.

Talks to the API for every score (`CRISK_API_URL`), and reads the committed aggregate
reports (`CRISK_REPORTS_DIR`, default `reports/`) for the portfolio, fairness and model
pages. Run: `uv run streamlit run src/crisk/dashboard/app.py`.
"""

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from crisk.dashboard.client import ApiClient, ApiError

REPORTS = Path(os.environ.get("CRISK_REPORTS_DIR", "reports"))
LC_SLIDERS = {  # feature -> (label, min, max, step)
    "loan_amnt": ("Loan amount ($)", 1_000.0, 40_000.0, 500.0),
    "annual_inc": ("Annual income ($)", 10_000.0, 300_000.0, 1_000.0),
    "int_rate": ("Interest rate (%)", 5.0, 31.0, 0.1),
    "dti": ("Debt-to-income (%)", 0.0, 40.0, 0.5),
    "fico_range_low": ("FICO (low end)", 660.0, 845.0, 5.0),
    "acc_open_past_24mths": ("Accounts opened, last 24 months", 0.0, 30.0, 1.0),
    "inq_last_6mths": ("Credit inquiries, last 6 months", 0.0, 8.0, 1.0),
    "revol_util": ("Revolving utilisation (%)", 0.0, 120.0, 1.0),
    "emp_length_years": ("Employment length (years)", 0.0, 10.0, 1.0),
}
MM_SLIDERS = {
    "inflow_monthly": ("Monthly wallet inflow (ETB)", 500.0, 80_000.0, 100.0),
    "inflow_cv": ("Month-to-month inflow variation (CV)", 0.0, 2.0, 0.01),
    "income_regularity": ("Share of income from regular payers", 0.0, 1.0, 0.01),
    "balance_min_to_inflow": ("Lowest balance / monthly inflow", 0.0, 5.0, 0.01),
    "inflow_outflow_ratio": ("Inflow / outflow", 0.7, 1.5, 0.01),
    "loan_amount": ("Loan amount (ETB)", 500.0, 50_000.0, 100.0),
}
SUB_GRADES = [f"{g}{n}" for g in "ABCDEFG" for n in range(1, 6)]


def report(name: str) -> pd.DataFrame | None:
    path = REPORTS / name
    return pd.read_csv(path) if path.exists() else None


def show_result(res: dict, currency: str) -> None:
    if res["synthetic"]:
        st.warning("Scored by a model trained on SYNTHETIC data: illustrative only.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Probability of default", f"{res['pd']:.1%}")
    c2.metric("Score", res["score"])
    c3.metric("Risk band", res["risk_band"])
    c4.metric("Decision", res["decision"].upper(), f"cutoff {res['cutoff']:.1%}", delta_color="off")
    if res["reasons"]:
        st.markdown("**Principal reasons for the decision** (adverse-action notice)")
        st.table(pd.DataFrame(res["reasons"])[["code", "text", "weight", "actionable"]])
    if res["missing_features"]:
        st.caption(
            f"{len(res['missing_features'])} features missing (treated as missing by the "
            f"model): {', '.join(res['missing_features'][:12])}"
        )
    if res.get("decision_id"):
        st.caption(f"Logged as decision #{res['decision_id']} ({currency}).")


def what_if(api: ApiClient, track: str, sliders: dict, info: dict) -> None:
    examples = api.examples(track)
    if not examples:
        st.info("No example applicants exported for this track.")
        return
    ref = st.selectbox("Start from", [e["ref"] for e in examples], key=f"{track}-ex")
    base = next(e for e in examples if e["ref"] == ref)["applicant"].copy()
    features = set(info["features"])
    cols = st.columns(2)
    applicant = dict(base)
    shown = [f for f in sliders if f in features or (track == "lending_club" and f == "annual_inc")]
    for i, f in enumerate(shown):
        label, lo, hi, step = sliders[f]
        value = base.get(f)
        value = lo if value is None else min(max(float(value), lo), hi)
        applicant[f] = cols[i % 2].slider(label, lo, hi, value, step, key=f"{track}-{ref}-{f}")
    if track == "lending_club" and "sub_grade" in features:
        current = base.get("sub_grade") if base.get("sub_grade") in SUB_GRADES else "C1"
        applicant["sub_grade"] = cols[0].select_slider(
            "Lending Club sub-grade", SUB_GRADES, current, key=f"lc-{ref}-sg"
        )
        applicant["grade"] = applicant["sub_grade"][0]
        applicant.pop("loan_to_income", None)  # the API derives it from amount and income
    if track == "mobile_money" and applicant.get("inflow_monthly"):
        applicant["loan_to_inflow"] = applicant["loan_amount"] / applicant["inflow_monthly"]
    try:
        show_result(api.score(track, applicant, ref), info.get("currency", ""))
    except ApiError as exc:
        st.error(str(exc))


def portfolio() -> None:
    st.subheader("Expected vs realised loss by vintage (Lending Club champion)")
    el = report("expected_vs_realised_loss.csv")
    if el is not None:
        t = el[(el["model"] == "gbm_with_pricing") & (el["version"] == "calibrated")]
        t = t.set_index("issue_quarter").sort_index()
        st.bar_chart(t[["expected_loss", "realised_loss"]] / 1e6, y_label="$M", stack=False)
        st.caption(
            "Calibrated on validation (2014Q3-2015Q1). EL covers 86% of realised loss on "
            "test, falling to 77% by 2016Q1."
        )
    st.subheader("Population stability (PSI vs the last year of train)")
    psi = report("psi_vs_recent_train.csv")
    if psi is not None:
        cols = [
            c
            for c in ("champion_pd", "scorecard_points", "int_rate", "sub_grade", "dti")
            if c in psi
        ]
        st.line_chart(psi.set_index("issue_quarter")[cols])
        st.caption(
            "< 0.1 stable, 0.1-0.25 monitor, > 0.25 act. LC's repricing (int_rate) "
            "drives the score into the monitor band from 2015Q3."
        )
    st.subheader("AUC by issue quarter")
    by_v = report("model_by_vintage.csv")
    if by_v is not None:
        st.line_chart(by_v.pivot_table(index="issue_quarter", columns="variant", values="auc"))


def fairness() -> None:
    st.subheader("Lending Club: approval-rate ratios (test, champion vs LC grades A-B)")
    t = report("fairness_summary.csv")
    if t is not None:
        st.dataframe(t, hide_index=True)
    st.subheader("Thin-file track (SYNTHETIC): gender, region, urban/rural, occupation")
    t = report("mm_fairness_summary.csv")
    if t is not None:
        st.dataframe(t.drop(columns="source", errors="ignore"), hide_index=True)
    g = report("mm_fairness_by_group.csv")
    if g is not None:
        attr = st.selectbox("Groups for attribute", sorted(g["attribute"].unique()))
        st.dataframe(
            g[g["attribute"] == attr].drop(columns="source", errors="ignore"), hide_index=True
        )
    st.caption(
        "AIR = group approval rate / most-approved group's. The four-fifths rule flags "
        "< 0.8. citl = mean PD - default rate (calibration within the group)."
    )


def models(api: ApiClient) -> None:
    st.subheader("Served models")
    try:
        info = api.models()
        rows = [
            {
                "track": t,
                "model": m["model_name"],
                "version": m["version"],
                "synthetic": m["synthetic"],
                "demo": m["demo"],
                "cutoff": m["cutoff"],
                "test AUC": m["metrics"].get("test", {}).get("auc"),
                "trained on": m["trained_on"],
            }
            for t, m in info.items()
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    except ApiError as exc:
        st.error(str(exc))
    st.subheader("Lending Club model comparison (test, future vintages)")
    m = report("model_metrics.csv")
    if m is not None:
        st.dataframe(m[m["split"] == "test"], hide_index=True)
    p = report("policies.csv")
    if p is not None:
        st.subheader("Approval policies (cutoffs chosen on validation)")
        st.dataframe(p[p["split"] == "test"], hide_index=True)
    mm = report("mm_metrics.csv")
    if mm is not None:
        st.subheader("Thin-file track (SYNTHETIC) model comparison")
        st.dataframe(mm.drop(columns="source", errors="ignore"), hide_index=True)
    st.subheader("Recent decisions (audit log)")
    try:
        st.dataframe(pd.DataFrame(api.decisions(25)), hide_index=True)
    except ApiError as exc:
        st.caption(str(exc))


def main() -> None:
    st.set_page_config(page_title="crisk", layout="wide")
    st.title("Credit risk: PD, decisions and reason codes")
    api = ApiClient()
    try:
        health = api.health()
        info = api.models()
    except ApiError as exc:
        st.error(str(exc))
        st.stop()
    if info.get("lending_club", {}).get("demo"):
        st.warning(
            "The Lending Club model is a DEMO trained on synthetic data (no Kaggle "
            "token at bootstrap). See the README for the real model's results."
        )
    tabs = st.tabs(
        [
            "Lending Club applicant",
            "Thin-file applicant (synthetic)",
            "Portfolio",
            "Fairness",
            "Models",
        ]
    )
    with tabs[0]:
        if "lending_club" in info:
            what_if(api, "lending_club", LC_SLIDERS, info["lending_club"])
    with tabs[1]:
        st.info("SYNTHETIC Ethiopian mobile-money borrowers: no real customer data.")
        if "mobile_money" in info:
            what_if(api, "mobile_money", MM_SLIDERS, info["mobile_money"])
    with tabs[2]:
        portfolio()
    with tabs[3]:
        fairness()
    with tabs[4]:
        models(api)
    st.caption(
        f"API: {api.base_url} - tracks {', '.join(health['tracks'])} - "
        f"database {'up' if health['database'] else 'down'}"
    )


if __name__ == "__main__":
    main()
