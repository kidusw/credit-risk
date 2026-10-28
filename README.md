# Credit risk & loan default

Predict the probability that a Lending Club loan defaults (PD) from what was known when it
was approved, then turn PDs into priced approve/decline decisions with reason codes,
fairness checks and stability monitoring. See [PLAN.md](PLAN.md) for the eight phases.

**Status:** Phases 1–5 done: data and leakage audit, WoE scorecard, monotonic LightGBM
challenger, calibration with expected loss and an approval policy, SHAP explanations with
adverse-action reason codes and a fairness audit.

### Results so far (36-month loans, future vintages)

| Model | Valid AUC | Test AUC | Test Gini | Test KS |
|---|---:|---:|---:|---:|
| Lending Club sub-grade alone (baseline) | 0.669 | 0.684 | 0.368 | 0.269 |
| WoE scorecard, borrower data only | 0.661 | 0.670 | 0.340 | 0.245 |
| WoE scorecard + LC pricing | 0.681 | 0.693 | 0.386 | 0.280 |
| LightGBM (monotonic), borrower data only | 0.687 | 0.694 | 0.388 | 0.283 |
| **LightGBM (monotonic) + LC pricing (champion)** | **0.695** | **0.706** | **0.412** | **0.301** |

The champion was chosen on validation AUC. Details:
[Scorecard (Phase 2)](#scorecard-phase-2), [LightGBM challenger (Phase 3)](#lightgbm-challenger-phase-3),
[Calibration, expected loss and approval policy (Phase 4)](#calibration-expected-loss-and-approval-policy-phase-4),
[Explainability and fairness (Phase 5)](#explainability-and-fairness-phase-5). Every number
here is from the models without `addr_state`, which Phase 5 dropped.

**What the champion would have earned** on test (2015Q2–2016Q1), with every cutoff
chosen on validation: a capital-constrained lender approving the 66% of Lending Club's
loans with calibrated PD ≤ 16.1% earns **7.11% on the amount lent** ($199.8M), against
5.71% for the whole book and 6.67% for Lending Club's own best grade cutoff (A–B). Every
decline comes with up to four reason codes.

## Data and licence

| | |
|---|---|
| Source | [Lending Club accepted loans 2007–2018Q4](https://www.kaggle.com/datasets/wordsforthewise/lending-club), Kaggle mirror of LendingClub's public loan downloads |
| File | `accepted_2007_to_2018Q4.csv.gz`, 393 MB compressed, 1.7 GB CSV, 2,260,668 loans × 151 columns |
| Licence | The Kaggle page lists **CC0-1.0**. The data itself was published by LendingClub, whose download page later added its own terms of use (the uploader notes this). So this project treats it as **research and demonstration use only**: the repository contains only the download script and aggregate statistics, never the raw or processed loan-level data. |

`data/` is git-ignored; run the pipeline below to recreate everything locally.

## Pipeline

```powershell
uv sync
copy .env.example .env            # add KAGGLE_API_TOKEN
docker compose up -d db           # PostgreSQL 16 on localhost:5434 (own compose project: crisk)
uv run crisk migrate
uv run crisk data download        # 393 MB, cached in data/raw/
uv run crisk data build           # ingest (if needed) -> filters -> split -> schema check
uv run crisk data load            # matured loans + vintage table into Postgres
uv run crisk train scorecard      # Phase 2: WoE scorecards + baseline, logged to MLflow
uv run crisk train gbm            # Phase 3: monotonic LightGBM, tuned on validation, vs scorecards
uv run crisk decide               # Phase 4: calibration, expected loss, approval cutoffs
uv run pytest
```

| Step | Output | Time / memory on the 8 GB work laptop |
|---|---|---|
| `data ingest`: stream the CSV in 200k-row chunks, read only the 110 non-dropped columns, float32 numerics, append to parquet | `data/interim/accepted.parquet` (192 MB) | 50 s, peak RSS 308 MB |
| `data build`: label, matured-vintage filter, derived features, time split, Pandera check | `data/processed/model_table.parquet`, `outcomes.parquet`, `reports/*.csv` | 32 s, peak RSS 1.8 GB; model table 284 MB in memory |
| `data load`: truncate + COPY in one transaction (idempotent) | Postgres `loans` (779,026 rows), `vintages` (82 rows) | 26 s |
| `train scorecard`: two scorecards + baseline, evaluated per split, vintage, grade, term | `models/*.joblib`, `reports/scorecard_*`, `reports/iv_*`, MLflow runs (`mlflow.db`) | 107 s, peak RSS 1.2 GB |
| `train gbm`: 12-config random search per variant (early stopping on validation), all models re-scored, paired bootstrap | `models/gbm_*.joblib`, `reports/gbm_*`, `reports/model_*`, `reports/auc_by_vintage.png` | 11 min, peak RSS 1.3 GB |
| `decide`: calibrate on validation, loss and profit per vintage, cutoffs chosen on validation | `reports/calibration_*`, `reports/policies.csv`, `reports/profit_curves*`, `reports/reliability_test.png` | 75 s |

## Leakage audit

Every one of the 151 source columns is classified in
[`src/crisk/data/columns.py`](src/crisk/data/columns.py). Only **feature** (application
data, bureau attributes at application) and **pricing** columns reach the modelling table.
`tests/test_columns.py` fails if any column named as post-origination in the plan is a model
input. `tests/test_build.py` checks that the built table has exactly the audited columns. The
Pandera schema is `strict`, so an undeclared column fails validation. Ingest also refuses a
source file whose header differs from the audit.

**Pricing columns** (`grade`, `sub_grade`, `int_rate`, `installment`) are known at
approval, but they are Lending Club's own risk assessment. They are kept, and flagged, so
Phases 2–3 can report results with and without them (a model that leans on them mostly
re-learns Lending Club's grading).

Columns that are **not** model inputs, and why:

| Column | Role | Why it is not a model input |
|---|---|---|
| `id` | key | loan identifier |
| `member_id` | drop | empty in the public file |
| `funded_amnt` | exposure | amount issued; EAD for expected loss (Phase 4) |
| `funded_amnt_inv` | drop | investors' funding, decided after the application; near-copy of loan_amnt |
| `emp_title` | drop | free text job title, ~500k distinct values |
| `issue_d` | date | month the loan was issued |
| `loan_status` | target | outcome at the data cut; becomes the default label |
| `pymnt_plan` | drop | post-origination: payment plan put in place later |
| `url` | drop | listing URL, an identifier |
| `desc` | drop | free text, empty for almost every loan after 2013 |
| `title` | drop | free text loan title, duplicates purpose |
| `zip_code` | audit | 3-digit zip is a known proxy for race; kept only for the fairness audit |
| `fico_range_high` | drop | fico_range_low + 4 (+5 for 441 loans): redundant |
| `out_prncp`, `out_prncp_inv` | drop | post-origination: payment history after the loan was issued |
| `total_pymnt_inv`, `total_rec_late_fee`, `last_pymnt_amnt`, `next_pymnt_d` | drop | post-origination: payment history after the loan was issued |
| `total_pymnt`, `total_rec_prncp`, `total_rec_int`, `last_pymnt_d` | outcome | post-origination: payment history; kept for LGD and profit (Phase 4) |
| `recoveries`, `collection_recovery_fee` | outcome | post-origination: recovered after charge-off; kept for LGD |
| `last_credit_pull_d` | drop | post-origination: date of the latest bureau pull |
| `last_fico_range_high`, `last_fico_range_low` | drop | post-origination: FICO refreshed after issuance, falls as borrowers default |
| `policy_code` | drop | constant (1) in this file |
| `sec_app_fico_range_high` | drop | sec_app_fico_range_low + 4: redundant |
| `hardship_*` (12), `deferral_term`, `payment_plan_start_date`, `orig_projected_additional_accrued_interest` | drop | post-origination: hardship plan agreed after the borrower got into trouble |
| `debt_settlement_flag`, `debt_settlement_flag_date`, `settlement_*` (5) | drop | post-origination: debt settlement after charge-off |

(`uv run crisk data audit` prints the full, ungrouped table.)

Derived at build time: `term_months`, `emp_length_years`, `credit_history_months`
(issue date − earliest credit line) and the co-borrower equivalent. That leaves 99 features (95 application + 4 pricing) plus 5 meta columns
(id, issue_d, issue_quarter, split, default).

## Target and matured vintages

`default = 1` for *Charged Off*, *Default* and *Does not meet the credit policy.
Status:Charged Off*; `0` for *Fully Paid* (both variants). *Current*, *In Grace Period*
and *Late* loans are unresolved and excluded. Any other status fails the build.

The data cut is **March 2019** (the latest `last_pymnt_d`). A loan counts as matured when
`issue month + term ≤ 2019-03`, which means 36-month loans issued up to **2016Q1** and
60-month loans up to **2014Q1**.

| Filter | Rows left | Removed |
|---|---:|---:|
| Raw loans (33 summary lines at the end of the file dropped) | 2,260,668 | |
| Resolved (paid off / charged off) | 1,348,099 | 912,569 |
| Matured vintage | 779,026 | 569,073 |

### Why the matured filter matters

Default rate by issue year among resolved loans (count in brackets):

| Issue year | 36-month | 60-month |
|---|---|---|
| 2007 | 26.2% (603) | – |
| 2008 | 20.7% (2,393) | – |
| 2009 | 13.7% (5,281) | – |
| 2010 | 10.9% (9,156) | 22.4% (3,381) |
| 2011 | 10.6% (14,101) | 23.6% (7,620) |
| 2012 | 13.6% (43,470) | 27.7% (9,897) |
| 2013 | 12.3% (100,422) | 25.1% (34,382) |
| 2014 | 13.7% (162,570) | 31.1% (60,533) partly immature |
| 2015 | 14.9% (283,026) | 36.4% (92,520) immature |
| 2016 | 19.8% (232,361) partly immature | 36.4% (60,744) immature |
| 2017 | 20.0% (128,551) immature | 33.0% (40,770) immature |
| 2018 | 13.2% (41,272) immature | 22.6% (15,046) immature |

For immature vintages, "resolved" means "already resolved". That set is biased in both
directions: early charge-offs and early prepayers. The 2016–17 rates are inflated by early
defaults. The 2018 rates are deflated because most of those loans haven't had time to
default: 2018Q4 shows 2.6%. Quarterly detail is in
[`reports/vintages.csv`](reports/vintages.csv), and in the Postgres `vintages` table.

## Time split

Split by issue month. Boundaries are in `crisk.data.build.SPLITS`:

| Split | Issued | Loans | Defaults | Default rate | 60-month share |
|---|---|---:|---:|---:|---:|
| train | 2007-06 – 2014-06 | 315,600 | 48,981 | 15.5% | 21.6% |
| valid | 2014-07 – 2015-03 | 147,183 | 21,191 | 14.4% | 0% |
| test | 2015-04 – 2016-03 | 316,243 | 48,200 | 15.2% | 0% |

**Caveat:** 60-month loans issued after 2014Q1 haven't matured, so validation and test are
36-month loans only. Train keeps 60-month loans, with `term_months` as a feature: 36-month
default rate 12.8% in train vs 25.4% for 60-month. Headline metrics in later phases are
therefore for 36-month loans, and the per-term results are reported separately.

Default rate by Lending Club grade, per split ([`reports/default_rate_by_grade.csv`](reports/default_rate_by_grade.csv)):

| Grade | train | valid | test |
|---|---:|---:|---:|
| A | 5.7% | 5.5% | 5.4% |
| B | 11.0% | 11.3% | 12.2% |
| C | 17.6% | 18.1% | 20.1% |
| D | 22.8% | 23.8% | 27.3% |
| E | 28.6% | 29.9% | 34.4% |
| F | 34.1% | 32.1% | 42.8% |
| G | 36.9% | 36.9% | 48.4% |

The same grade got riskier over time, especially C–G in the test period. This is the drift a
model trained on older vintages has to survive, and why results are reported per vintage.

## Missingness

Per feature and split: [`reports/missingness.csv`](reports/missingness.csv). Two groups are
structurally missing in the matured window:

- **Joint applications** (`annual_inc_joint`, `dti_joint`, `sec_app_*`, `revol_bal_joint`):
  Lending Club started joint loans in late 2015, so these are 100% missing in train/valid and
  ≥ 99.5% in test.
- **Bureau fields added in late 2015** (`open_acc_6m`, `open_il_*`, `open_rv_*`, `il_util`,
  `all_util`, `inq_fi`, `total_cu_tl`, `inq_last_12m`, `mths_since_rcnt_il`, `total_bal_il`,
  `max_bal_bc`): 100% missing in train/valid, about 67% in test.

They stay in the table (they are legitimate application-time data), and Phase 2's feature
screening drops anything that is empty in train.

`dti` contains sentinel values (−1 and 999). The schema allows them, and Phase 2 cleans them.

## Database

Own Compose project (`name: crisk`) and port **5434**, so it runs alongside Project 1's
database on 5433. `loans` holds the loan-level facts (amounts, grade, purpose, outcome,
split, `source='lending_club'`). The full feature matrix stays in parquet. `vintages` is the
table above at quarter level. A `crisk_test` database is created on first start for the
integration tests.

## Scorecard (Phase 2)

A classic credit scorecard, built the way a bank would build one
([`src/crisk/models/scorecard.py`](src/crisk/models/scorecard.py)):

1. **Monotonic optimal binning** per feature with
   [optbinning](https://github.com/guillermo-navas-palencia/optbinning). Numeric features
   are constrained to a monotonic default-rate trend, with at most 8 bins and each bin at
   least 5% of train. Missing values get their own bin.
2. **Screening on information value (IV):** features with IV < 0.02 are dropped. Of any pair
   of WoE features with |correlation| > 0.7, the one with lower IV is dropped.
3. **Logistic regression on WoE.** A feature whose coefficient has the wrong sign (riskier
   bins scored as safer) is removed, and the model is refit until every sign is right.
4. **Points:** 600 points at 50:1 good:bad odds, and every 20 points doubles the odds. Each
   bin's points are rounded, so a score is just a sum of whole numbers.

Both variants are fitted on the same 315,600 train loans only:

- **with_pricing**: all application-time features plus Lending Club's grade, sub-grade,
  rate and installment. 18 features selected. `int_rate` and `grade` are dropped as
  near-duplicates of `sub_grade` (|corr| 0.97).
- **no_pricing**: borrower and bureau data only. 17 features selected.

The champion is chosen on **validation AUC** (with_pricing). Test was scored once.

### Results

| Model | Split | AUC | Gini | KS | PR-AUC | Brier | Mean PD | Default rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LC sub-grade baseline | valid | 0.669 | 0.338 | 0.250 | 0.232 | 0.1179 | 14.0% | 14.4% |
| | test | 0.684 | 0.368 | 0.269 | 0.262 | 0.1227 | 13.3% | 15.2% |
| Scorecard, no pricing | valid | 0.661 | 0.323 | 0.234 | 0.235 | 0.1184 | 13.6% | 14.4% |
| | test | 0.670 | 0.340 | 0.245 | 0.256 | 0.1237 | 12.9% | 15.2% |
| **Scorecard + pricing** | valid | **0.681** | **0.362** | **0.264** | **0.251** | **0.1171** | 13.5% | 14.4% |
| | test | **0.693** | **0.386** | **0.280** | **0.277** | **0.1221** | 12.8% | 15.2% |

The baseline gives each loan the train default rate of its Lending Club sub-grade.

Reading these honestly:

- **Lending Club's own grade is a strong baseline.** The borrower-only scorecard does
  *not* beat it (−0.014 test AUC). Lending Club priced loans with data we don't have:
  its full bureau file and its own models.
- **Combining both helps**, by +0.009 test AUC over the baseline. The gain sits *within*
  grades, where sub-grade alone can't rank. Test AUC within each grade (A–G) is 0.57–0.63
  for the scorecard against 0.47–0.60 for the baseline
  ([`reports/scorecard_by_grade.csv`](reports/scorecard_by_grade.csv)).
- **Test AUC is higher than validation AUC.** Later vintages are more spread out (the grade
  default rates diverge, see Phase 1), which makes ranking easier. It isn't a sign of a
  better model.
- **PDs drift low.** Mean predicted PD falls from 13.6% in 2014Q3 to 12.6% in 2016Q1, while
  the realised default rate rises from 13.6% to 16.1%. Ranking holds up, but calibration
  doesn't. That is Phase 4's job (recalibration on validation).
- 60-month loans exist only in train: AUC 0.640 there, against 0.672 for 36-month loans
  ([`reports/scorecard_by_term.csv`](reports/scorecard_by_term.csv)).

### Per vintage (test and validation quarters, AUC)

| Issue quarter | LC sub-grade | No pricing | + Pricing | Default rate | Mean PD (+ pricing) |
|---|---:|---:|---:|---:|---:|
| 2014Q3 (valid) | 0.657 | 0.664 | 0.678 | 13.6% | 13.6% |
| 2014Q4 (valid) | 0.676 | 0.660 | 0.684 | 14.5% | 13.6% |
| 2015Q1 (valid) | 0.673 | 0.661 | 0.681 | 14.8% | 13.4% |
| 2015Q2 (test) | 0.677 | 0.666 | 0.686 | 15.4% | 13.4% |
| 2015Q3 (test) | 0.681 | 0.671 | 0.692 | 14.6% | 12.9% |
| 2015Q4 (test) | 0.682 | 0.668 | 0.691 | 14.8% | 12.5% |
| 2016Q1 (test) | 0.694 | 0.675 | 0.700 | 16.1% | 12.6% |

Validation is 2014Q3–2015Q1, test 2015Q2–2016Q1. Full table: [`reports/scorecard_by_vintage.csv`](reports/scorecard_by_vintage.csv).

### The scorecard

Human-readable, bin → points:
[`reports/scorecard_with_pricing.md`](reports/scorecard_with_pricing.md), plus the CSV
and the no-pricing version. An excerpt:

| Feature | Bin | Default rate (train) | Points |
|---|---|---:|---:|
| sub_grade | A1–A4 | 4.9% | 50 |
| sub_grade | C1, C2 | 15.4% | 30 |
| sub_grade | E1–G5 | 30.6% | 16 |
| term | 36 months | 12.8% | 33 |
| term | 60 months | 25.4% | 20 |
| fico_range_low | < 672.5 | 20.3% | 28 |
| fico_range_low | ≥ 757.5 | 6.3% | 36 |
| loan_to_income | < 0.13 | 11.2% | 35 |
| loan_to_income | ≥ 0.39 | 24.7% | 21 |

Test scores run from 489 to 617, with a median of 546. Default rate falls steadily by score
decile, from 33.4% in the lowest decile to 3.3% in the highest.

IV for every candidate feature, and why each one was dropped:
[`reports/iv_with_pricing.csv`](reports/iv_with_pricing.csv). Of the borrower features,
`term_months` (0.136), `fico_range_low` (0.110) and `loan_to_income` (0.101) are
medium-strength. Everything else is weak (IV < 0.1). No feature reaches the
"suspiciously strong" level (IV > 0.5) that would point to leakage.

### Tracking

Every run is logged to MLflow (`mlflow.db`, git-ignored) under the experiment
`crisk-scorecard`, with parameters, per-split metrics, the points table and the IV table.
Browse with `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## LightGBM challenger (Phase 3)

[`src/crisk/models/gbm.py`](src/crisk/models/gbm.py): LightGBM on every available
application-time feature, with **monotonic constraints** wherever the direction of risk is
settled credit practice:

- **PD can only go up** with term, rate, sub-grade, DTI, loan-to-income, utilisation,
  recent inquiries and new accounts, and derogatory marks.
- **PD can only go down** with FICO, income, credit history length, months since the
  latest inquiry or account, mortgages and the share of accounts never delinquent.

31 of the 70 features are constrained. The rest, such as purpose and balances, are left free.
Grade and sub-grade become ordinals (A1 = 0 … G5 = 34) so they can be constrained too. A
test fits the model and checks on predictions that a higher FICO never raises the PD and
a higher DTI never lowers it.

**Protocol:** a 12-configuration seeded random search per variant (leaves, minimum leaf
size, row and column sampling, L2). Each configuration is fitted on train, with early
stopping on validation AUC. The best validation AUC wins. Test was scored once, for every
model, after all choices were made. The search space is small and flat: validation AUC
ranges over 0.694–0.695 (with pricing) and 0.685–0.687 (without). The model isn't
sensitive to these settings, so a larger search isn't worth the laptop time. Results:
[`reports/gbm_tuning_*.csv`](reports/gbm_tuning_with_pricing.csv).

![ROC-AUC per issue quarter for the five models](reports/auc_by_vintage.png)

### Results

| Model | Split | AUC | Gini | KS | PR-AUC | Brier |
|---|---|---:|---:|---:|---:|---:|
| LC sub-grade baseline | valid | 0.669 | 0.338 | 0.250 | 0.232 | 0.1179 |
| | test | 0.684 | 0.368 | 0.269 | 0.262 | 0.1227 |
| Scorecard + pricing | valid | 0.681 | 0.362 | 0.264 | 0.251 | 0.1171 |
| | test | 0.693 | 0.386 | 0.280 | 0.277 | 0.1221 |
| LightGBM, borrower only | valid | 0.687 | 0.374 | 0.271 | 0.257 | 0.1166 |
| | test | 0.694 | 0.388 | 0.283 | 0.278 | 0.1216 |
| **LightGBM + pricing** | valid | **0.695** | **0.390** | **0.287** | **0.264** | **0.1161** |
| | test | **0.706** | **0.412** | **0.301** | **0.291** | **0.1210** |

**AUC lift with a 95% paired bootstrap interval** (200 resamples of the same loans for
both models; [`reports/model_lift.csv`](reports/model_lift.csv)):

| Challenger | vs | Valid lift | Test lift [95% CI] |
|---|---|---:|---|
| LightGBM + pricing | Scorecard + pricing | +0.014 | **+0.013** [0.012, 0.014] |
| LightGBM, borrower only | Scorecard, borrower only | +0.026 | **+0.024** [0.023, 0.026] |
| LightGBM + pricing | LC sub-grade | +0.026 | **+0.022** [0.021, 0.023] |
| LightGBM, borrower only | LC sub-grade | +0.018 | **+0.010** [0.008, 0.012] |

What this shows:

- **LightGBM beats the scorecard on every held-out vintage**, by a stable margin
  (chart above). This isn't one lucky quarter.
- **Borrower data alone now beats Lending Club's grade.** The scorecard couldn't (−0.014),
  but LightGBM can (+0.010, CI excludes zero). The extra signal is in interactions and
  non-linear shapes that a one-dimensional WoE transform throws away.
- **The gain is within grades**, where it matters for pricing. Test AUC within each grade
  A–G is 0.59–0.65 for LightGBM + pricing, 0.57–0.63 for the scorecard and 0.47–0.60 for
  sub-grade alone ([`reports/model_by_grade.csv`](reports/model_by_grade.csv)).
- **Calibration still drifts.** Mean predicted PD on test is 12.4% against 15.2% realised,
  slightly worse than the scorecard. Ranking is the model's job here; Phase 4 recalibrates
  on validation.

**What the trade-off costs:** the scorecard is 18 bins anyone can read and audit by hand.
LightGBM is 205 trees. Phase 5 (SHAP and reason codes) is what makes the champion
explainable to an applicant. A bank might still ship the scorecard and keep LightGBM as a
challenger. This project carries both forward.

### What drives it

Gain share for the champion ([`reports/gbm_importance_with_pricing.csv`](reports/gbm_importance_with_pricing.csv)):
`sub_grade` 27%, `int_rate` 9%, `grade` 8%, `loan_to_income` 6%, `annual_inc` 5%,
`term_months` 4%. Without pricing, `term_months`, `fico_range_low`, `loan_to_income` and
`annual_inc` lead. SHAP (Phase 5) gives a less gain-biased view.

**`addr_state` was dropped in Phase 5.** In the first Phase 3 model it was the
second-largest feature by gain (9%), although its IV was too low for the scorecard (0.011).
LightGBM's native categorical splits inflate gain on a 50-level feature, and a
validation-only ablation showed it was worth only 0.0012 AUC. Phase 5 applied a rule fixed
in advance and dropped it ([below](#should-state-be-in-the-model)). The tables above are
from the retrained models without it.

### Tracking

MLflow experiment `crisk-gbm`: one run per variant, with the chosen hyperparameters, the
best iteration, every metric, the AUC lifts, and the tuning and importance tables.

## Calibration, expected loss and approval policy (Phase 4)

[`src/crisk/models/calibration.py`](src/crisk/models/calibration.py),
[`src/crisk/decision/`](src/crisk/decision). This phase covers the champion (LightGBM +
pricing) and, for comparison, the scorecard + pricing, which is the model a bank would more
likely ship. **Every choice is made on validation:** the calibration method, the PD
cutoffs and the grade cutoffs. Test only reports what those choices would have earned.

### Calibration

Platt (a logistic fit on logit PD) and isotonic calibrators are both fitted on validation.
The method is picked by **5-fold cross-validation within validation** (out-of-fold Brier):
Platt for both models, by a hair (0.11589 vs 0.11592 for the champion). Both are then refitted on all of
validation.

| Champion | Split | Brier | ECE | Mean PD − default rate | Slope | Mean PD | Default rate |
|---|---|---:|---:|---:|---:|---:|---:|
| Raw | valid | 0.1161 | 0.0139 | −1.35 pp | 0.93 | 13.1% | 14.4% |
| Raw | test | 0.1210 | 0.0287 | −2.87 pp | 0.96 | 12.4% | 15.2% |
| **Platt** | valid | 0.1159 | 0.0047 | 0.00 pp | 1.00 | 14.4% | 14.4% |
| **Platt** | test | **0.1203** | **0.0161** | **−1.54 pp** | 1.04 | 13.7% | 15.2% |

ECE uses 20 equal-frequency bins. Slope is the coefficient of `y ~ logit(PD)`: 1 means
well spread. Full table, including isotonic and the scorecard:
[`reports/calibration_metrics.csv`](reports/calibration_metrics.csv).

![Reliability on test: raw vs calibrated PD by decile](reports/reliability_test.png)

Calibration roughly **halves the test calibration error** and keeps the slope near 1, so
the ranking is untouched and the level is fixed. It can't fix everything. Validation is
calibrated almost perfectly, but test still under-predicts by 1.5 points, because the
book kept getting riskier after the calibration window (Phase 1's grade drift).

### Expected vs realised loss

**EL = PD × LGD × EAD.** EAD is the funded amount. LGD is the mean share of the funded
amount never recovered on charged-off **train** loans, by grade and term: from 0.48 (A,
36-month) to 0.68 (G, 60-month), in
[`reports/lgd_by_grade_term.csv`](reports/lgd_by_grade_term.csv). Realised loss is the
principal actually lost, net of recoveries. The outcome columns used here (payments,
recoveries) measure results and estimate LGD only. They are never model inputs.

| Issue quarter | Default rate | Mean PD (calibrated) | Realised loss | EL ÷ realised (raw) | EL ÷ realised (calibrated) |
|---|---:|---:|---:|---:|---:|
| 2014Q3 (valid) | 13.6% | 14.4% | $34.4M | 0.97 | 1.07 |
| 2014Q4 (valid) | 14.5% | 14.5% | $44.6M | 0.92 | 1.02 |
| 2015Q1 (valid) | 14.8% | 14.3% | $53.6M | 0.88 | 0.97 |
| 2015Q2 (test) | 15.4% | 14.2% | $64.3M | 0.82 | 0.91 |
| 2015Q3 (test) | 14.6% | 13.8% | $70.0M | 0.85 | 0.94 |
| 2015Q4 (test) | 14.8% | 13.3% | $86.1M | 0.78 | 0.86 |
| 2016Q1 (test) | 16.1% | 13.7% | $108.1M | 0.70 | 0.77 |
| **Test total** | 15.2% | 13.7% | **$328.5M** | 0.78 | **0.86** |

Over the whole validation window, expected loss matches realised loss (1.01 calibrated).
On test it covers 86%, falling to 77% by 2016Q1: **a model calibrated on yesterday's
vintages under-reserves when credit quality deteriorates.** A lender would need to watch
this (PSI and realised-vs-expected monitoring are Phase 7), not assume the calibration
holds. Per quarter: [`reports/expected_vs_realised_loss.csv`](reports/expected_vs_realised_loss.csv).

### Approval policy

**Realised profit** per loan = everything received (principal, interest, late fees,
recoveries) − collection fees − amount lent. It's undiscounted, with no servicing fee and
no cost of funds. Policies approve the lowest-PD share of applicants. Every loan in the
data was already approved by Lending Club, so these policies **decline a subset of LC's
book**, and "approve all" is LC's actual result.

Two lenders, two objectives, both optimised on validation:

- **Total profit** (capital is free and unlimited).
- **Return on the amount lent** (capital is the constraint).

![Profit and return on amount lent vs approval rate, test vintages](reports/profit_curves_test.png)

| Policy (cutoff chosen on validation) | Objective | Test approval | Lent | Profit | Return on lent | Bad rate |
|---|---|---:|---:|---:|---:|---:|
| Approve all (LC's book) | – | 100% | $4,108M | $234.6M | 5.71% | 15.2% |
| LC grades A–G (best grade cutoff) | profit | 100% | $4,108M | $234.6M | 5.71% | 15.2% |
| Champion PD ≤ 0.392 | profit | 99.1% | $4,072M | **$237.5M** | 5.83% | 15.0% |
| Champion: expected profit > 0 (untuned) | profit | 99.3% | $4,082M | $237.1M | 5.81% | 15.0% |
| LC grades A–B (best grade cutoff) | return | 58.4% | $2,480M | $165.3M | 6.67% | 9.3% |
| Scorecard PD ≤ 0.131 | return | 54.1% | $2,390M | $168.0M | 7.03% | 8.6% |
| **Champion PD ≤ 0.161** | return | 66.3% | $2,812M | $199.8M | **7.11%** | 9.6% |

Validation results and the scorecard's profit cutoff are in
[`reports/policies.csv`](reports/policies.csv); every choice is in
[`reports/decision_choices.csv`](reports/decision_choices.csv).

What the numbers say:

- **On total profit there is little to gain, judged on validation.** There, Lending
  Club's rates priced the risk: every grade A–G returned 4.7–8.8% as a block, so the best
  cutoff declines only the riskiest 1% (+$2.9M on test, +0.12 pp return). An untuned rule, "approve if (1 − PD) × good-loan return > PD ×
  bad-loan loss" with sub-grade margins from train, lands in the same place. That's a good
  sign the calibrated PDs are economically sensible.
- **Then the riskiest grades stopped paying.** On test, grades E, F and G *lost* money: −0.1%, −4.5% and
  −11.6% of the amount lent, against +4.7% to +8.8% on validation
  ([`reports/return_by_grade.csv`](reports/return_by_grade.csv)). They're small, 4% of
  test lending and a $1.6M loss, but the direction matters. That's why the test
  profit curve peaks nearer 95% approval, and why a cutoff tuned on validation leaves
  money on the table. Any policy fitted to the past inherits this risk. It's the strongest
  argument in this project for monitoring realised loss by vintage (Phase 7).
- **For a capital-constrained lender the model earns its keep.** At a similar approval
  rate, the champion's cutoff returns **7.11% vs 6.67%** for LC's grades A–B. It lends
  $332M more, makes **$34.5M more profit**, and has a similar bad rate (9.6% vs 9.3%).
  The scorecard's return cutoff sits in between (7.03%).
- **Approval rates drift with the PDs.** The cutoff approving 63% of validation approves
  66% of test, because test PDs are still slightly low (above). A production policy would
  monitor the approval rate and recalibrate.
- **Not modelled:** cost of funds, servicing fees, discounting, prepayment timing, and how
  declined applicants would have behaved (reject inference is Phase 7). These are
  comparisons between policies on the same book, not a forecast of absolute returns.

## Explainability and fairness (Phase 5)

[`src/crisk/explain/`](src/crisk/explain), run with `crisk explain`. The policy explained
and audited is Phase 4's capital-constrained lender: approve when the calibrated champion
PD is at or below 0.161, the cutoff chosen on validation. Lending Club's own grade cutoff
(A–B, a similar approval rate) is audited alongside as the reference.

### SHAP

Exact TreeSHAP from LightGBM's own `pred_contrib`, on the log-odds scale, for **every**
held-out loan (463k), not a sample: a decline needs its reasons. A test checks that each
row's contributions plus the base value add up to the model's logit.

![Mean |SHAP| per feature, test loans](reports/shap_importance.png)

| Feature | Share of mean \|SHAP\| | Share of gain |
|---|---:|---:|
| `sub_grade` | 15.0% | 26.7% |
| `annual_inc` | 7.2% | 5.2% |
| `loan_to_income` | 6.2% | 6.1% |
| `int_rate` | 6.2% | 9.0% |
| `acc_open_past_24mths` | 5.7% | 3.2% |
| `dti` | 4.4% | 1.8% |
| `term_months` | 4.1% | 4.2% |
| `grade` | 3.7% | 8.3% |

Gain flatters the features the trees split on first. SHAP spreads the credit more
evenly: **borrower data (income, loan size relative to income, recently opened accounts,
DTI) carries more of the decision than gain suggests**, and Lending Club's grade less.
Full table: [`reports/shap_importance.csv`](reports/shap_importance.csv).

![SHAP dependence for the largest non-grade features](reports/shap_dependence.png)

The dependence plots ([`reports/shap_dependence.csv`](reports/shap_dependence.csv)) show
the monotonic constraints at work: contributions rise steadily with DTI, loan-to-income
and recent accounts, and fall with income, flattening at the extremes (DTI above 25,
income above $100k). There are no jagged reversals.

### Adverse-action reason codes

Each of the 70 features belongs to exactly one of 19 plain-language reasons, such as
"Debt-to-income ratio too high" or "Too many recent credit inquiries"
([`reports/reason_code_map.csv`](reports/reason_code_map.csv)). A reason's weight for an
applicant is the sum of its features' SHAP values. Only reasons that pushed the PD **up**
can be stated, the largest first, at most four.

![Reason-code waterfall for an applicant declined just over the cutoff](reports/reason_waterfall.png)

**Every one of the 161,156 held-out declines gets reasons:** four for all but four loans,
which had only three reasons pushing their PD up
([`reports/reason_counts.csv`](reports/reason_counts.csv)). A test asserts this. The most
common reasons on test declines:

| Reason | First reason | Among the top 4 |
|---|---:|---:|
| Lending Club's risk grade and interest rate (not actionable) | 60.1% | 84.2% |
| Too many accounts opened recently | 18.7% | 58.7% |
| Income too low or not verified | 6.2% | 50.6% |
| Loan amount too high for income | 6.1% | 39.3% |
| Debt-to-income ratio too high | 0.4% | 37.7% |
| Level of existing balances and credit limits | 1.3% | 20.5% |

**The honest weak spot:** for 60% of declines the top reason is Lending Club's own grade.
That's a true statement about this model, which leans on LC's pricing, but it tells the
applicant nothing they can act on. So the code is flagged `actionable=False`, and an
applicant still gets up to three borrower-level reasons behind it. A lender building its
own model, with no third-party grade as an input, would not have this problem. The
borrower-only LightGBM (test AUC 0.694) would be the model to explain to applicants.
Sample notices: [`reports/reason_codes_sample.csv`](reports/reason_codes_sample.csv).

### Fairness

Lending Club publishes **no sex, race or age**, so none of these is measured, and none is
inferred. What can be audited is state, home ownership, employment length and income band,
plus the 3-digit zip code. The zip code is a well-known proxy for race. It is kept out of
the model and used only here. Each attribute gets three views on test:

- **Adverse impact ratio (AIR):** a group's approval rate divided by the
  most-approved group's. US practice flags ratios below 0.8, the four-fifths rule.
- **True-positive-rate gap:** the approval rate among loans that were repaid, against the
  best group. This is equal opportunity.
- **Calibration by group:** mean PD minus default rate. If PDs are right within a group,
  a lower approval rate reflects higher risk, not a biased score.

| Attribute | Champion: lowest AIR (group) | Groups < 0.8 | LC A–B: lowest AIR | Groups < 0.8 | Max \|mean PD − default\| |
|---|---|---:|---|---:|---:|
| State (40 with ≥ 1,000 loans) | 0.81 (HI) | 0 | 0.79 (HI) | 1 | 5.8 pp |
| Home ownership | 0.75 (rent) | 1 | 0.83 (rent) | 0 | 2.3 pp |
| Employment length | 0.61 (missing) | 1 | 0.85 (missing) | 0 | 3.4 pp |
| Income band | **0.40 (< $40k)** | 2 | 0.55 (< $40k) | 2 | 2.5 pp |
| Zip3 (77 with ≥ 1,000 loans) | 0.65 (104xx) | 20 | 0.66 (104xx) | 12 | 5.1 pp |

Per group: [`reports/fairness_by_group.csv`](reports/fairness_by_group.csv). Summary:
[`reports/fairness_summary.csv`](reports/fairness_summary.csv).

What this shows:

- **The champion concentrates on income more than Lending Club's grades do.** It approves
  36% of applicants earning under $40k against 89% of those over $150k (AIR 0.40, vs 0.55
  for LC's grades). Among repaid loans the gap is 51 points. But the low-income group
  really does default more (20.5% vs 10.6%), and its PDs are calibrated almost exactly
  (−0.1 pp). The gap is risk the model measures, not a score biased against the group.
  It is still a policy choice, and a regulator would ask about it. Income is a
  legitimate credit factor, but a lender could cap the gap and pay for it in return.
- **Renters and applicants with no stated employment length are approved less** (AIR 0.75
  and 0.61). Both groups default more (18.2% and 22.1%), and their PDs are, if anything,
  slightly *low* (−2.3 and −3.4 pp). So the model isn't penalising them beyond their risk.
- **Almost every group's PDs are a little low** (mean PD below the default rate), because
  test is riskier than the calibration window (Phase 4). Home ownership, employment and
  income groups all err in that direction, within 3.4 pp. Calibration is loosest by state
  (−5.8 to +3.6 pp: some states are over-predicted, some under), one more sign that
  geography carries noise rather than signal.
- **Zip code is the proxy to watch.** Twenty of 77 large zip3 areas fall below 0.8 even
  though zip code isn't a model input. Geography reaches the model through income, DTI and
  credit history. That can't be fully removed. It can be monitored, which is what this
  table is for. Without race data, it can't be said whether those areas are
  disproportionately minority. This is stated, not worked around.

### Should state be in the model?

`addr_state` was the second-largest LightGBM feature by gain in Phase 3. The rule was fixed
**before** the ablation was run: geography is the textbook redlining proxy, state isn't
something an applicant can act on, and gain on a 50-level categorical is inflated. So state
stays only if it earns, **on validation**, at least +0.005 AUC or +0.10 pp return on the
amount lent at a re-chosen cutoff.

| Model | Valid AUC | Valid return | Test AUC | Test return | Test profit | State AIR (test) | States < 0.8 | Zip3 < 0.8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| With `addr_state` | 0.6958 | 8.58% | 0.7075 | 7.22% | $183.6M | 0.55 | 17 | 33 |
| **Without (champion)** | **0.6951** | **8.57%** | **0.7060** | **7.11%** | **$199.8M** | **0.81** | **0** | **20** |

Validation gain from state: **+0.0007 AUC and +0.01 pp return**, far short of the bar,
so **it is dropped** ([`reports/addr_state_ablation.csv`](reports/addr_state_ablation.csv)).
The first ablation, before the retrain, found +0.0012 and +0.05 pp and reached the same
decision. Test numbers were computed after the decision and played no part in it.

Dropping it costs 0.0015 test AUC. On test, the version with state earns a slightly
higher return on a smaller book (59% approved), while the champion makes $16M more
profit at 66%. The fairness gain is large: **state-level approval gaps shrink from
0.55 to 0.81 AIR, and no state now falls below the four-fifths line.** The number of
flagged zip3 areas drops from 33 to 20.
