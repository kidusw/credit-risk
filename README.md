# Credit risk & loan default

Predict the probability that a Lending Club loan defaults (PD) from what was known when it
was approved, then turn PDs into priced approve/decline decisions with reason codes,
fairness checks and stability monitoring. See [PLAN.md](PLAN.md) for the eight phases.

**Status:** Phase 1 (setup, data, leakage audit) done.

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
uv run pytest
```

| Step | Output | Time / memory on the 8 GB work laptop |
|---|---|---|
| `data ingest`: stream the CSV in 200k-row chunks, read only the 110 non-dropped columns, float32 numerics, append to parquet | `data/interim/accepted.parquet` (192 MB) | 50 s, peak RSS 308 MB |
| `data build`: label, matured-vintage filter, derived features, time split, Pandera check | `data/processed/model_table.parquet`, `outcomes.parquet`, `reports/*.csv` | 32 s, peak RSS 1.8 GB; model table 284 MB in memory |
| `data load`: truncate + COPY in one transaction (idempotent) | Postgres `loans` (779,026 rows), `vintages` (82 rows) | 26 s |

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
