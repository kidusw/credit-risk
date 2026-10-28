"""Seeded generator of SYNTHETIC Ethiopian mobile-money borrowers and their wallet history.

**Everything here is synthetic.** No real customer, transaction or telebirr data is used;
every row carries `source="synthetic"`. The structure follows Project 1's
`SyntheticGenerator` (txcat): a per-person profile first, then scheduled recurring flows
(salary, bills, equb, iddir) and Poisson-distributed discretionary ones on top. It is
rewritten here vectorised over borrowers, since 20k borrowers x 12 months is ~6M rows, and
it models a wallet (agent cash-in/out, airtime, P2P) rather than a bank account.

**Who applies.** Borrowers apply for a 90-day microloan between Jan 2024 and Jun 2025, more
of them in later months as the product grows. Each has 360 days of wallet history before
the application date. Demographics are generated **by design** so fairness can be measured:
gender, region and urban/rural, with an occupation drawn from urban/rural shares.

**Generating assumptions** (illustrative, not estimated from any real data):

- Median monthly income: salaried 14k ETB, trader 11k, farmer 7k, day labourer 5.5k;
  x1.25 urban, x0.85 rural, and **x0.85 for women** (an assumed pay gap, so the audit has
  a real difference to find). Log-normal spread 0.5.
- Income regularity (latent, 0-1) by occupation: salaried high, traders medium, day
  labourers and farmers low. Farm income is seasonal (Oct-Jan harvest). Irregular earners
  get more volatile months, less regular airtime top-ups and less punctual bills.
- Spending ratio (latent): outflows / inflows, ~N(0.93, 0.07). Above 1 the balance drains.
- Buffer (latent): the lowest balance the borrower keeps, in months of income.
- 12% suffer an income shock (one of the last three months at 35%).
- Equb membership 35% (+10 pts women, +10 traders), iddir 55% (+15 rural), P2P contacts
  rise with a latent social-ties trait.

**The hidden default rule** (`RULE`, below) is a logit on those latent traits plus noise
and a small drift by application month. The model never sees it, the latent traits or
the demographics: it sees only features computed from the transactions. Gender, region
and urban/rural are **not** in the rule; any approval gap between groups comes through
income, regularity and the other behaviours above.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

SEED = 42
N_BORROWERS = 20_000
FIRST_MONTH = pd.Timestamp("2024-01-01")
N_COHORT_MONTHS = 18  # applications Jan 2024 .. Jun 2025
HISTORY_DAYS = 360  # 12 months of 30 days
MONTHS = HISTORY_DAYS // 30
LOAN_TERM_DAYS = 90

REGIONS = {  # share of borrowers, share urban
    "Addis Ababa": (0.24, 1.00),
    "Oromia": (0.24, 0.25),
    "Amhara": (0.18, 0.22),
    "SNNP & Sidama": (0.11, 0.20),
    "Tigray": (0.06, 0.30),
    "Somali": (0.05, 0.25),
    "Dire Dawa & Harari": (0.07, 0.85),
    "Afar": (0.05, 0.20),
}
OCCUPATIONS = ("salaried", "trader", "farmer", "day_labourer")
OCCUPATION_SHARE = {"urban": (0.40, 0.35, 0.03, 0.22), "rural": (0.10, 0.25, 0.50, 0.15)}
INCOME_MEDIAN = {"salaried": 14_000, "trader": 11_000, "farmer": 7_000, "day_labourer": 5_500}
REGULARITY_BETA = {"salaried": (9, 1.5), "trader": (4, 3), "farmer": (2, 4), "day_labourer": (2, 3)}
# Expected income credits per month (salary: exactly one).
INCOME_EVENTS = {"salaried": 0, "trader": 14, "farmer": 1.5, "day_labourer": 4.5}
HARVEST_MONTHS = (10, 11, 12, 1)
WOMEN_INCOME_FACTOR = 0.85

# Hidden default rule: logit = INTERCEPT + sum of these terms + DRIFT * cohort month + noise.
INTERCEPT = -2.75
RULE = {
    "irregular_income": "1.4 * (0.6 - regularity)",
    "volatility": "2.0 * (volatility - 0.3)",
    "overspending": "4.0 * (spend_ratio - 0.93)",
    "thin_buffer": "-0.45 * ln(buffer_months / 0.3)",
    "loan_size": "0.9 * (loan_to_income - 0.9)",
    "income_shock": "0.8 * shock",
    "social_ties": "-0.35 * equb - 0.2 * iddir - 0.6 * (social - 0.5)",
    "low_income": "-0.3 * ln(income / 10000)",
}
DRIFT_PER_MONTH = 0.015  # later cohorts are riskier (unobserved squeeze): +0.25 by Jun 2025
NOISE_SD = 0.8


@dataclass
class SyntheticBook:
    borrowers: pd.DataFrame  # one row per applicant: demographics, loan, latent traits, label
    transactions: pd.DataFrame  # wallet history before each application


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def _borrowers(rng: np.random.Generator, n: int) -> pd.DataFrame:
    month_weight = 1 + np.arange(N_COHORT_MONTHS) / 10  # the product grows
    cohort = rng.choice(N_COHORT_MONTHS, n, p=month_weight / month_weight.sum())
    app_date = (
        FIRST_MONTH
        + pd.to_timedelta(cohort * 30.44, unit="D").round("D")
        + pd.to_timedelta(rng.integers(0, 28, n), unit="D")
    )
    names = list(REGIONS)
    share = np.array([REGIONS[r][0] for r in names])
    region = rng.choice(names, n, p=share / share.sum())
    urban = rng.random(n) < np.array([REGIONS[r][1] for r in region])
    gender = np.where(rng.random(n) < 0.45, "female", "male")
    female = gender == "female"

    occupation = np.empty(n, dtype=object)
    for setting, is_setting in (("urban", urban), ("rural", ~urban)):
        for women in (True, False):
            m = is_setting & (female == women)
            p = np.array(OCCUPATION_SHARE[setting])
            if women:  # more women trade, fewer are salaried
                p = p * np.array([0.85, 1.3, 1.0, 1.0])
            occupation[m] = rng.choice(OCCUPATIONS, m.sum(), p=p / p.sum())

    median = np.array([INCOME_MEDIAN[o] for o in occupation], dtype=float)
    median *= np.where(urban, 1.25, 0.85) * np.where(female, WOMEN_INCOME_FACTOR, 1.0)
    income = np.round(median * rng.lognormal(0, 0.5, n), -1)

    regularity = np.array([rng.beta(*REGULARITY_BETA[o]) for o in occupation])
    volatility = np.clip(0.08 + 0.5 * (1 - regularity) + rng.normal(0, 0.08, n), 0.03, 0.9)
    spend_ratio = np.clip(rng.normal(0.93, 0.07, n), 0.70, 1.15)
    buffer_months = (
        rng.lognormal(np.log(0.3), 0.9, n) * np.clip(1.15 - spend_ratio, 0.05, 0.45) / 0.22
    )
    shock = rng.random(n) < 0.12
    equb = rng.random(n) < 0.35 + 0.10 * female + 0.10 * (occupation == "trader")
    iddir = rng.random(n) < 0.55 + 0.15 * ~urban
    social = rng.beta(2, 2, n)
    loan_to_income = rng.uniform(0.3, 1.6, n) * np.where(spend_ratio > 1, 1.2, 1.0)
    loan_amount = np.clip(np.round(income * loan_to_income, -2), 500, 50_000)

    df = pd.DataFrame(
        {
            "borrower_id": np.arange(1, n + 1, dtype=np.int32),
            "application_date": app_date,
            "cohort_month": cohort.astype(np.int16),
            "gender": gender,
            "region": region,
            "urban_rural": np.where(urban, "urban", "rural"),
            "occupation": occupation.astype(str),
            "loan_amount": loan_amount,
            "latent_income": income,
            "latent_regularity": regularity,
            "latent_volatility": volatility,
            "latent_spend_ratio": spend_ratio,
            "latent_buffer_months": buffer_months,
            "latent_shock": shock,
            "latent_equb": equb,
            "latent_iddir": iddir,
            "latent_social": social,
        }
    )
    return df


def rule_terms(b: pd.DataFrame) -> pd.DataFrame:
    """The hidden rule's terms per borrower (log-odds). See `RULE` for the formulas."""
    lti = b["loan_amount"] / b["latent_income"]
    return pd.DataFrame(
        {
            "irregular_income": 1.4 * (0.6 - b["latent_regularity"]),
            "volatility": 2.0 * (b["latent_volatility"] - 0.3),
            "overspending": 4.0 * (b["latent_spend_ratio"] - 0.93),
            "thin_buffer": -0.45 * np.log(b["latent_buffer_months"] / 0.3),
            "loan_size": 0.9 * (lti - 0.9),
            "income_shock": 0.8 * b["latent_shock"].astype(float),
            "social_ties": -0.35 * b["latent_equb"]
            - 0.2 * b["latent_iddir"]
            - 0.6 * (b["latent_social"] - 0.5),
            "low_income": -0.3 * np.log(b["latent_income"] / 10_000),
        },
        index=b.index,
    )


def _label(rng: np.random.Generator, b: pd.DataFrame) -> pd.DataFrame:
    terms = rule_terms(b)
    logit = (
        INTERCEPT
        + terms.sum(axis=1)
        + DRIFT_PER_MONTH * b["cohort_month"]
        + rng.normal(0, NOISE_SD, len(b))
    )
    out = b.copy()
    for c in terms:
        out[f"rule_{c}"] = terms[c]
    out["true_pd"] = _sigmoid(logit)
    out["default"] = (rng.random(len(b)) < out["true_pd"]).astype(np.int8)
    return out


# --- wallet history ----------------------------------------------------------------------


class _Events:
    """Accumulates transaction columns as arrays; one call per kind of flow."""

    def __init__(self):
        self.parts: list[dict[str, np.ndarray]] = []

    def add(self, owner, day, amount, category: str, counterparty) -> None:
        n = len(owner)
        cp = np.asarray(counterparty, dtype=object)
        self.parts.append(
            {
                "owner": np.asarray(owner, dtype=np.int32),
                "day": np.clip(np.asarray(day), 0, HISTORY_DAYS - 1).astype(np.int16),
                "amount": np.round(np.asarray(amount, dtype=float), 2),
                "category": np.full(n, category, dtype=object),
                "counterparty": cp if cp.ndim else np.full(n, counterparty, dtype=object),
            }
        )

    def frame(self) -> pd.DataFrame:
        cols = {k: np.concatenate([p[k] for p in self.parts]) for k in self.parts[0]}
        df = pd.DataFrame(cols)
        for c in ("category", "counterparty"):
            df[c] = df[c].astype("category")
        return df


def _repeat(counts) -> tuple[np.ndarray, np.ndarray]:
    """Owner index for each event and the event's rank within its owner."""
    counts = np.asarray(counts, dtype=np.int64)
    owner = np.repeat(np.arange(len(counts)), counts)
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    return owner, np.arange(counts.sum()) - starts


def _split(rng, owner_month: np.ndarray, totals: np.ndarray) -> np.ndarray:
    """Split each owner-month total across its events with random (exponential) weights."""
    w = rng.exponential(1.0, len(owner_month))
    sums = np.bincount(owner_month, weights=w, minlength=len(totals))
    return totals[owner_month] * w / sums[owner_month]


def _history(rng: np.random.Generator, b: pd.DataFrame) -> pd.DataFrame:
    n = len(b)
    occ = b["occupation"].to_numpy()
    income = b["latent_income"].to_numpy()
    r = b["latent_regularity"].to_numpy()
    v = b["latent_volatility"].to_numpy()
    urban = (b["urban_rural"] == "urban").to_numpy()
    ev = _Events()

    # Monthly income: noise with the borrower's volatility, harvest seasonality, shocks.
    mult = rng.lognormal(0, v[:, None], (n, MONTHS))
    mult /= mult.mean(axis=1, keepdims=True)
    first = b["application_date"] - pd.Timedelta(days=HISTORY_DAYS)
    cal_month = (first.dt.month.to_numpy()[:, None] - 1 + np.arange(MONTHS)) % 12 + 1
    harvest = np.isin(cal_month, HARVEST_MONTHS)
    farmer = occ == "farmer"
    mult[farmer] *= np.where(harvest[farmer], 2.0, 0.67)
    shocked = b["latent_shock"].to_numpy()
    shock_month = rng.integers(MONTHS - 3, MONTHS, n)
    mult[shocked, shock_month[shocked]] *= 0.35
    month_income = income[:, None] * mult  # (n, 12)

    # Income credits. Salary: one per month on payday, punctual if regular.
    counts = np.zeros((n, MONTHS), dtype=np.int64)
    sal = occ == "salaried"
    counts[sal] = 1
    for o in ("trader", "farmer", "day_labourer"):
        m = occ == o
        lam = np.full((m.sum(), MONTHS), INCOME_EVENTS[o])
        if o == "farmer":
            lam = lam + harvest[m]  # an extra harvest sale
        counts[m] = rng.poisson(lam)
    om, _ = _repeat(counts.ravel())  # owner-month index
    owner, month = om // MONTHS, om % MONTHS
    amount = _split(rng, om, month_income.ravel())
    payday = rng.integers(24, 28, n)
    jitter = np.round(rng.normal(0, 0.5 + 4 * (1 - r[owner])))
    day_in_month = np.where(sal[owner], payday[owner] + jitter, rng.integers(0, 30, len(om)))
    day = month * 30 + np.clip(day_in_month, 0, 29)
    o_occ = occ[owner]
    labour_emp = rng.integers(1, 4, len(om))
    customer = rng.integers(0, 40, len(om))
    cp = np.select(
        [o_occ == "salaried", o_occ == "day_labourer", o_occ == "trader"],
        [
            np.full(len(om), "EMPLOYER"),
            np.char.add("EMPLOYER-", labour_emp.astype(str)),
            np.char.add("CUSTOMER-", customer.astype(str)),
        ],
        "AGENT",
    )
    for cat, mask in (
        ("Income", np.isin(o_occ, ["salaried", "day_labourer"])),
        ("Sales", o_occ == "trader"),
        ("Cash In", o_occ == "farmer"),
    ):
        ev.add(owner[mask], day[mask], amount[mask], cat, cp[mask])

    # Remittances from family and friends.
    n_in_contacts = 1 + rng.poisson(1 + 4 * b["latent_social"].to_numpy())
    owner, _ = _repeat(rng.poisson(0.6 * MONTHS, n))
    ev.add(
        owner,
        rng.integers(0, HISTORY_DAYS, len(owner)),
        rng.lognormal(np.log(0.12 * income[owner]), 0.6),
        "Remittance In",
        np.char.add("CONTACT-", rng.integers(0, n_in_contacts[owner]).astype(str)),
    )

    # Airtime: cadence evenly spaced for regular earners, bunched for irregular ones.
    rate = rng.uniform(3, 9, n)
    owner, rank = _repeat(rng.poisson(rate * MONTHS))
    total = np.bincount(owner, minlength=n)
    spacing = HISTORY_DAYS / np.maximum(total[owner], 1)
    day = (rank + 0.5) * spacing + rng.normal(0, 2 + 14 * (1 - r[owner]))
    base = np.clip(income[owner] * 0.004, 15, 300)
    ev.add(
        owner,
        day,
        -np.maximum(5, np.round(rng.lognormal(np.log(base), 0.35) / 5) * 5),
        "Airtime",
        "AIRTIME",
    )

    # Bills: electricity (most urban, some rural), water (urban), on schedule.
    for name, share_u, share_r, dom, pct, floor in (
        ("EEU", 0.95, 0.35, 10, 0.015, 30),
        ("WATER", 0.80, 0.05, 18, 0.006, 20),
    ):
        has = rng.random(n) < np.where(urban, share_u, share_r)
        owner, month = _repeat(np.where(has, MONTHS, 0))
        late = np.round(np.abs(rng.normal(0, 1 + 3 * (1 - r[owner]))))
        amt = np.maximum(floor, pct * income[owner] * rng.lognormal(0, 0.25, len(owner)))
        ev.add(owner, month * 30 + dom + late, -amt, "Bills", name)

    # Iddir (burial society) and equb (rotating savings): fixed monthly contributions.
    iddir = b["latent_iddir"].to_numpy()
    fee = rng.choice([50, 100, 150, 200], n)
    owner, month = _repeat(np.where(iddir, MONTHS, 0))
    ev.add(owner, month * 30 + 2 + rng.integers(-1, 2, len(owner)), -fee[owner], "Iddir", "IDDIR")
    equb = b["latent_equb"].to_numpy()
    contribution = np.maximum(100, np.round(income * rng.uniform(0.05, 0.15, n), -2))
    owner, month = _repeat(np.where(equb, MONTHS, 0))
    dom = np.where(sal[owner], payday[owner] + 1, 5) + np.round(rng.normal(0, 2 * (1 - r[owner])))
    ev.add(owner, month * 30 + np.clip(dom, 0, 29), -contribution[owner], "Equb", "EQUB")
    members = np.flatnonzero(equb)
    ev.add(
        members,
        rng.integers(0, MONTHS, len(members)) * 30 + 15,
        contribution[members] * MONTHS,
        "Equb Payout",
        "EQUB",
    )

    # Merchant payments and P2P transfers out.
    owner, _ = _repeat(rng.poisson(5 * MONTHS, n))
    ev.add(
        owner,
        rng.integers(0, HISTORY_DAYS, len(owner)),
        -rng.lognormal(np.log(0.015 * income[owner]), 0.8),
        "Merchant",
        np.char.add("MERCHANT-", rng.integers(0, 200, len(owner)).astype(str)),
    )
    n_out_contacts = np.minimum(30, 1 + rng.poisson(2 + 6 * b["latent_social"].to_numpy()))
    owner, _ = _repeat(rng.poisson(2.5 * MONTHS, n))
    ev.add(
        owner,
        rng.integers(0, HISTORY_DAYS, len(owner)),
        -rng.lognormal(np.log(0.04 * income[owner]), 0.7),
        "P2P Out",
        np.char.add("CONTACT-", rng.integers(0, n_out_contacts[owner]).astype(str)),
    )

    # Agent cash-out tops outflows up to the borrower's spending ratio, month by month.
    tx = ev.frame()
    om = tx["owner"].to_numpy() * MONTHS + tx["day"].to_numpy() // 30
    amt = tx["amount"].to_numpy()
    inflow = np.bincount(om, weights=np.maximum(amt, 0), minlength=n * MONTHS)
    outflow = np.bincount(om, weights=np.maximum(-amt, 0), minlength=n * MONTHS)
    spend = np.repeat(b["latent_spend_ratio"].to_numpy(), MONTHS)
    remaining = np.maximum(0, spend * inflow - outflow)
    counts = np.where(remaining > 0, 1 + rng.poisson(1.5, n * MONTHS), 0)
    om, _ = _repeat(counts)
    ev.add(
        om // MONTHS,
        (om % MONTHS) * 30 + rng.integers(0, 30, len(om)),
        -_split(rng, om, remaining),
        "Cash Out",
        "AGENT",
    )
    return ev.frame()


def _balances(tx: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Running wallet balance. The opening balance is set so the lowest point equals the
    borrower's buffer: a wallet can't go negative, and how low it gets is the trait."""
    tx = tx.sort_values(["owner", "day", "amount"], ascending=[True, True, False], kind="stable")
    cum = tx.groupby("owner", sort=False)["amount"].cumsum().to_numpy()
    owner = tx["owner"].to_numpy()
    low = np.minimum(0, pd.Series(cum).groupby(owner).min().reindex(range(len(b))).fillna(0))
    buffer = (b["latent_buffer_months"] * b["latent_income"]).to_numpy()
    opening = buffer - low.to_numpy()
    tx = tx.assign(balance=(opening[owner] + cum).astype("float32"))
    return tx.reset_index(drop=True), opening


def generate(n: int = N_BORROWERS, seed: int = SEED) -> SyntheticBook:
    """Borrowers (with the hidden label) and their wallet transactions. Fully seeded."""
    rng = np.random.default_rng(seed)
    b = _label(rng, _borrowers(rng, n))
    tx, opening = _balances(_history(rng, b), b)
    b["opening_balance"] = np.round(opening, 2)
    start = (b["application_date"] - pd.Timedelta(days=HISTORY_DAYS)).to_numpy()
    tx.insert(0, "borrower_id", b["borrower_id"].to_numpy()[tx["owner"].to_numpy()])
    tx.insert(1, "date", start[tx["owner"].to_numpy()] + pd.to_timedelta(tx["day"], unit="D"))
    tx = tx.drop(columns="owner")
    tx["amount"] = tx["amount"].astype("float32")
    b["source"] = "synthetic"
    tx["source"] = "synthetic"
    return SyntheticBook(borrowers=b, transactions=tx)
