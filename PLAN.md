# Project 4: Credit Risk & Loan Default (flagship)

Predict the probability that a loan defaults (PD) from what is known at application time,
turn it into an approve/decline decision priced on expected profit and loss, explain every
decision with reason codes, check fairness, and monitor stability across loan vintages.
A second, synthetic track scores Ethiopian "thin-file" borrowers (no credit-bureau
history) from telebirr-style mobile-money behaviour, reusing Project 1's generator and
recurring-payment detector.

Package name: `crisk`. Rules from the portfolio `PLAN.md` apply (time-based splits, honest
metrics against a baseline, synthetic data labelled synthetic, secrets and data git-ignored).

---

## Data

| Dataset | Size | Role |
|---|---|---|
| Lending Club accepted loans 2007–2018 (Kaggle mirror) | ~2.2M loans, ~150 columns, ~400–600 MB compressed | Main dataset: application features, issue date, outcome (fully paid / charged off), amounts |
| Lending Club rejected applications 2007–2018 | ~27M rows, ~9 columns | Reject inference (Phase 7) |
| Synthetic Ethiopian mobile-money borrowers (**synthetic**) | ~20k borrowers × 12 months of wallet history | Thin-file track (Phase 6); always `source='synthetic'` |
| Home Credit Default Risk | 307k applications | Out of scope (no calendar dates for a time split; competition usage rules). Optional later |

**Target:** `default = 1` for Charged Off / Default (and "Does not meet the credit policy.
Status: Charged Off"); `0` for Fully Paid. Loans still Current or Late at the data cut are
excluded. Only **matured vintages** are used: loans whose full term (36 or 60 months)
ended before the data cut. Otherwise recent vintages look artificially safe, because
their defaults haven't happened yet.

## Stack

Same as Project 1: Python 3.12 / uv (dependency groups core / train / dashboard / dev),
PostgreSQL 16, SQLAlchemy/Alembic, Pandera, scikit-learn, LightGBM, MLflow, FastAPI,
Streamlit, Docker Compose, pytest/ruff, GitHub Actions. New: `optbinning` (WoE binning
and scorecards) and `shap`, both small. Data is stored as parquet, with column selection
and down-cast dtypes to fit in 8 GB of RAM.

---

## Phase 1 — Setup + data + leakage audit

- [x] Extract `_template/` from Project 1 (uv groups, src layout, Postgres + Alembic,
      Pandera, CI with pinned actions, Dockerfile with the CA-bundle secret, bootstrap
      pattern, `/data/` ignore rule) and start `credit-risk` from it
- [x] **Licence check:** confirm the Lending Club Kaggle mirror's licence and terms and
      record them in the README. If it's unclear, document that it is used for research
      and demo only and never redistributed (the repo holds only the download script)
- [x] Kaggle download script (token from `.env`); stream the CSV in chunks into parquet,
      keeping only the needed columns, with typed dtypes and memory measured
- [x] **Leakage audit:** classify every column as *application-time* (kept) or
      *post-origination* (dropped: `total_pymnt*`, `recoveries`, `last_pymnt_d`,
      `last_fico_*`, `out_prncp*`, `collection_recovery_fee`, `hardship_*`,
      `debt_settlement_*`, …). The list is committed as code with a test, and the reason
      for each dropped column is in the README
- [x] Target definition + matured-vintage filter; vintage table of default rate by issue
      quarter
- [x] Time-based split by issue date: train = older vintages, validation = next,
      test = most recent matured vintages (boundaries recorded)
- [x] Pandera schema for the modelling table; loans + vintages into Postgres
- **Metrics:** rows kept/dropped at each filter, default rate by vintage and grade,
  missingness per feature
- **Exit criteria:** `crisk data build` produces a processed parquet with no
  post-origination columns (enforced by a test); split boundaries and vintage default
  rates in the README; CI green.

## Phase 2 — Scorecard baseline (WoE + logistic regression)

- [x] Monotonic WoE binning per feature (optbinning), with information value (IV) for
      feature screening
- [x] Logistic regression on WoE features → points scale (e.g. 600 points at 1:50 odds,
      20 points to double the odds)
- [x] Scorecard table (bin → points) exported as CSV/markdown, which is human-readable
      like a bank's
- [x] All runs logged to MLflow
- **Metrics:** ROC-AUC, Gini, KS, PR-AUC, Brier on validation and test vintages; IV table
- **Exit criteria:** scorecard documented and reproducible; numbers reported per vintage,
  not just pooled.

## Phase 3 — Challenger: monotonic LightGBM

- [ ] LightGBM with monotonic constraints on features where the risk direction is known
      (e.g. higher DTI or more recent inquiries can't lower risk; higher income can't
      raise it)
- [ ] Hyperparameters tuned on the validation vintages only; test scored once
- [ ] Comparison vs the scorecard on future vintages, including per-vintage curves and
      per-grade results
- **Metrics:** same as Phase 2, plus AUC lift over the scorecard with a bootstrap
  confidence interval
- **Exit criteria:** comparison table in the README; the champion is chosen on validation,
  and the result is reported honestly even if the scorecard wins.

## Phase 4 — Calibration, expected loss, decision threshold

- [ ] Calibrate PDs (isotonic vs Platt, fitted on validation); reliability curves per
      vintage
- [ ] Expected loss = PD × LGD × EAD. LGD is estimated from charged-off loans' recoveries
      (target-side information, allowed here because it isn't a model input); EAD = loan
      amount
- [ ] Profit curve: interest earned on good loans vs losses on bad ones → profit-
      maximising approval threshold; approval rate vs bad rate trade-off
- **Metrics:** Brier, expected calibration error, calibration slope/intercept; expected
  vs realised loss per vintage; profit at the chosen threshold vs "approve all" and vs
  Lending Club's own grades
- **Exit criteria:** threshold chosen on validation and applied to test; profit and
  approval-rate numbers in the README.

## Phase 5 — Explainability + fairness

- [ ] SHAP for the champion: global importance, dependence plots, per-applicant
      explanations
- [ ] Adverse-action reason codes: top 4 reasons for each decline, mapped to plain-
      language text ("Debt-to-income ratio too high")
- [ ] Fairness on the attributes Lending Club has (state, home ownership, employment
      length, income band) plus a zip-code-as-proxy discussion. Protected attributes
      (sex, race, age) are **not available**; this is stated, not worked around
- **Metrics:** approval-rate ratio between groups (demographic parity), true-positive
  rate gaps (equal opportunity), calibration by group
- **Exit criteria:** reason codes are returned for every decline in tests; the fairness
  table and its limitations are in the README.

## Phase 6 — Ethiopian mobile-money track (synthetic, thin-file)

- [ ] Generator: ~20k synthetic borrowers with 12 months of telebirr-style wallet history,
      built on Project 1's `SyntheticGenerator` (salary, airtime, P2P transfers, bills,
      equb/iddir), plus synthetic demographics (gender, region, urban/rural) **by design**
      so fairness can be measured. Every row is `source='synthetic'`
- [ ] Hidden default rule (documented, seeded): latent risk from income regularity,
      cash-flow volatility, balance troughs and the like, with noise. The model never sees
      the rule
- [ ] Alternative-data features: income regularity via Project 1's recurring detector,
      inflow/outflow ratio, balance volatility, airtime top-up cadence, equb/iddir
      participation, P2P network breadth
- [ ] Same scorecard → LightGBM → calibration → reason codes pipeline; full fairness
      metrics by gender and region
- **Metrics:** AUC/KS/Brier on a time-held-out cohort; recovery of the hidden rule
  (feature importance vs true drivers); fairness gaps
- **Caveat (in the README):** the labels come from a rule we wrote, so this track shows
  the pipeline, explainability and fairness method on a realistic local use case, not
  real-world accuracy
- **Exit criteria:** thin-file model served alongside the Lending Club model; fairness
  table by gender and region.

## Phase 7 — Reject inference + stability

- [ ] Reject inference on the Lending Club rejected applications (the shared columns
      only): fuzzy augmentation / parcelling, and the resulting change in approval
      decisions
- [ ] Population stability: PSI of scores and key features by vintage; characteristic
      stability report (sets up Project 5, drift monitoring)
- **Metrics:** PSI (<0.1 stable, 0.1–0.25 monitor, >0.25 act), approval-rate change from
  reject inference
- **Exit criteria:** stability table in the README; it's clearly stated that reject
  inference can't be validated without the outcomes of rejected applications.

## Phase 8 — Ship

- [ ] FastAPI: `POST /score` (PD, score points, risk band, decision, reason codes; both
      tracks), `GET /model`, `GET /health`; decisions logged to Postgres
- [ ] Streamlit: applicant what-if (sliders → PD and reasons), portfolio/vintage view,
      fairness tables, model comparison
- [ ] MODEL_CARD.md + a short **model validation report** in the style banks use
      (conceptual soundness, performance, stability, limitations, monitoring plan)
- [ ] Slim images + bootstrap + one-command `docker compose up` (as in Project 1);
      README with architecture diagram, results and screenshots
- **Exit criteria:** fresh clone → `docker compose up` → app works end-to-end (on a
  bundled small sample when no Kaggle token is set); CI green.

---

## Risks

- **Download size / network:** ~400–600 MB compressed (accepted) + rejected file;
  downloads on this network are slow. Done once, cached in `/data/`.
- **RAM (8 GB machine):** 2.2M × 150 doesn't fit naively. Mitigation: chunked CSV
  reading, column selection, categorical/float32 dtypes, parquet; train on the full data
  for the models, but sample for SHAP.
- **Data licence:** to be confirmed in Phase 1. Never redistribute the raw data.
- **Fairness limits on Lending Club:** no protected attributes, so conclusions are
  limited and the full fairness analysis runs on the synthetic track only.
- **Synthetic-label caveat:** the thin-file track's labels come from our own rule; its
  accuracy numbers are not evidence of real-world performance.
- **Target and selection bias:** outcomes exist only for approved loans (Phase 7), and
  pre-2010 vintages are small and different (the 2008 crisis); both are reported per
  vintage.
