# Model card: `crisk`

Two probability-of-default (PD) models served by one API (`POST /score`). Both are
monotonic LightGBM models with Platt calibration, an approval cutoff and adverse-action
reason codes. The full evidence is in the [README](README.md); the model validation
report, in the format banks use, is [docs/validation_report.md](docs/validation_report.md).

| | Lending Club track | Thin-file track |
|---|---|---|
| Data | **Real**: Lending Club accepted loans, 36-month, matured | **SYNTHETIC**: generated telebirr-style wallet histories |
| Applicants | US personal-loan borrowers with a credit file | Ethiopian wallet users with no bureau file (simulated) |
| Model | LightGBM, 70 features, 31 monotone constraints, 205 trees | LightGBM, 24 wallet features, 17 monotone constraints, 115 trees |
| Calibration | Platt, fitted on validation (2014Q3–2015Q1) | Platt, fitted on the Q1 2025 cohort |
| Decision | approve if PD ≤ 0.161 (return-maximising, chosen on validation) | approve if PD ≤ 0.085 (profit-maximising, chosen on validation) |
| Test AUC / KS / Brier | 0.706 / 0.301 / 0.120 (2015Q2–2016Q1) | 0.723 / 0.333 / 0.109 (Q2 2025 cohort) |
| Baseline on test | LC sub-grade alone: AUC 0.684 | loan ÷ inflow rule of thumb: AUC 0.618 |
| Served artefact | `models/serving/lending_club.{lgb.txt,json}` | `models/serving/mobile_money.{lgb.txt,json}` |

## Model details

- **Lending Club:** LightGBM (`objective=binary`) on every application-time feature.
  `addr_state` is excluded by a pre-registered Phase 5 rule: it added only 0.0007
  validation AUC. Lending Club's grade, sub-grade, rate and instalment are inputs (the
  "pricing" variant), so the model refines LC's own risk assessment. A WoE scorecard
  (18 characteristics) is kept as the interpretable challenger: test AUC 0.693.
- **Thin-file:** LightGBM on alternative-data features: income regularity (Project 1's
  recurring-payment detector), inflow volatility, balance troughs, airtime cadence,
  equb/iddir participation, P2P breadth, loan ÷ inflow. It ties its WoE scorecard
  (0.723 vs 0.723). That's expected, because the hidden rule is additive; a real lender
  should ship the scorecard.
- **Outputs:** calibrated PD; points on a 600 = 50:1 odds, 20-points-to-double scale; a
  risk band (A < 5% … F ≥ 30%); approve/decline; up to four reasons per decline, from
  exact TreeSHAP grouped into plain-language codes.
- **Serving:** native LightGBM text files plus JSON metadata, so no pickle is loaded at
  serving time. Served PDs equal the training pipeline's exactly (a test asserts this).
  Every answer is logged to the Postgres `decisions` table.

## Intended use

Portfolio demonstration of a credit-decisioning pipeline: data and leakage audit, time
splits, scorecard vs gradient boosting, calibration, profit-based cutoffs, reason codes,
fairness and stability monitoring. **Not for lending decisions.** The Lending Club model
describes loans issued in 2007–2016 by one platform. The thin-file model has learned a
rule written for this project.

## Training data

- **Lending Club:** 315,600 train loans (2007–2014Q2), 147,183 validation (2014Q3–2015Q1),
  316,243 test (2015Q2–2016Q1). The time split is by issue date, and only matured loans
  are used. The data is research-use only and never redistributed; the repository holds
  aggregates only.
- **Thin-file:** 20,000 synthetic borrowers (seed 42): train is the 2024 cohorts (11,125),
  validation Q1 2025 (4,219), test Q2 2025 (4,656). The generating assumptions are in
  the README.
- **Demo model:** a fresh clone without a Kaggle token gets a DEMO Lending Club model
  trained on synthetic LC-shaped loans. The API says `synthetic: true` and `/model` says
  `demo: true`. Its numbers mean nothing.

## Evaluation highlights

- Champion vs scorecard on test: +0.013 AUC (95% CI 0.012–0.014), stable on every vintage.
- Calibration: test PDs under-predict by 1.5 pp, and expected loss covers 86% of realised
  loss on test, falling to 77% in 2016Q1 as the book deteriorates.
- At the return cutoff on test: 66% approved, 7.11% return on the amount lent, against
  6.67% for LC's best grade cutoff (A–B).
- Reason codes on every decline (161,156 held-out declines). For 60% of them the first
  reason is LC's own grade, which is not actionable for the applicant.

## Fairness

No protected attributes (sex, race, age) exist in Lending Club's data; none is inferred.

- **Lending Club:**
  - State AIR ≥ 0.81 once state is out of the model.
  - Income under $40k has AIR 0.40 against 0.55 under LC's grades. Those PDs are
    calibrated (−0.1 pp), so the gap is measured risk, but it is a policy choice.
  - 20 of 77 large zip3 areas fall below 0.8.
- **Thin-file (synthetic, demographics by design):**
  - No gender gap (AIR 0.97).
  - Rural AIR 0.51 and farmers 0.15. These groups default more and their PDs are
    slightly low, so the gap is risk-driven.

## Limitations

- **Selection bias.** Outcomes exist only for loans Lending Club approved (about 12% of
  applications). Reject inference (Phase 7) is a sensitivity analysis and can't be
  validated without the rejected applicants' outcomes.
- **Drift.** The champion's score PSI reaches 0.155 by 2015Q4, driven by LC's repricing
  (`int_rate` PSI 0.70 against the last year of train). The model needs recalibration on
  recent vintages before any further use.
- **Economics.** Profit is undiscounted, with no cost of funds or servicing fees. Cutoffs
  compare policies on the same book; they don't forecast returns.
- **Thin-file realism.** Every relationship in the synthetic book was written by hand.
  The track shows method, not predictive validity.
- **Missing inputs.** The API treats any feature left out as missing. A partial
  application gets a valid but less informed PD, and the response lists what was missing.

## Monitoring

PSI of the score and its key inputs by month, with the < 0.1 / 0.1–0.25 / > 0.25 bands
(`crisk monitor stability`); realised vs expected loss by vintage as loans season;
approval rate against the cutoff's validation rate; AIR by group on every decision batch;
and the `decisions` table as the audit trail.
