import gzip

import numpy as np
import pandas as pd
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine

from crisk.config import get_settings
from crisk.data.columns import COLUMNS, STRING_COLUMNS
from tests import ROOT

# (issue month, term, loan_status, expected outcome in the modelling table)
LOANS = [
    ("Jun-2010", 36, "Fully Paid", "train"),
    ("Mar-2012", 60, "Charged Off", "train"),
    ("Feb-2010", 36, "Does not meet the credit policy. Status:Charged Off", "train"),
    ("Aug-2014", 36, "Charged Off", "valid"),
    ("Jan-2015", 36, "Fully Paid", "valid"),
    ("Jun-2015", 36, "Fully Paid", "test"),
    ("Jan-2016", 36, "Default", "test"),
    ("Mar-2016", 36, "Fully Paid", "test"),  # last matured 36-month vintage
    ("Apr-2016", 36, "Fully Paid", "immature"),
    ("Jun-2014", 60, "Fully Paid", "immature"),  # 60 months -> Jun-2019, after the cut
    ("Jan-2013", 36, "Current", "unresolved"),
    ("Jan-2017", 36, "Late (31-120 days)", "unresolved"),
]
FOOTER = [
    "Total amount funded in policy code 1: 1000",
    "Total amount funded in policy code 2: 0",
]


def synthetic_loans(seed: int = 0) -> pd.DataFrame:
    """A tiny Lending-Club-shaped file: every source column, plausible values, no real data."""
    rng = np.random.default_rng(seed)
    n = len(LOANS)
    df = pd.DataFrame({c: rng.uniform(0, 20, n).round(1) for c in COLUMNS})
    df["id"] = [str(1000 + i) for i in range(n)]
    df["member_id"] = np.nan
    df["issue_d"] = [loan[0] for loan in LOANS]
    df["term"] = [f" {loan[1]} months" for loan in LOANS]
    df["loan_status"] = [loan[2] for loan in LOANS]
    df["loan_amnt"] = rng.integers(10, 350, n) * 100.0
    df["funded_amnt"] = df["loan_amnt"]
    df["int_rate"] = rng.uniform(6, 25, n).round(2)
    df["installment"] = (df["loan_amnt"] / 30).round(2)
    df["grade"] = rng.choice(list("ABCDEFG"), n)
    df["sub_grade"] = df["grade"] + rng.choice(list("12345"), n)
    df["emp_length"] = rng.choice(["< 1 year", "3 years", "10+ years", None], n)
    df["home_ownership"] = rng.choice(["RENT", "MORTGAGE", "OWN"], n)
    df["annual_inc"] = rng.integers(20, 200, n) * 1000.0
    df["verification_status"] = rng.choice(["Verified", "Not Verified"], n)
    df["purpose"] = rng.choice(["debt_consolidation", "credit_card", "car"], n)
    df["zip_code"] = "123xx"
    df["addr_state"] = rng.choice(["CA", "NY", "TX"], n)
    df["earliest_cr_line"] = rng.choice(["Aug-1999", "Jan-2005", "Nov-2010"], n)
    df["sec_app_earliest_cr_line"] = None
    df["fico_range_low"] = rng.integers(132, 168, n) * 5.0
    df["fico_range_high"] = df["fico_range_low"] + 4
    df["sec_app_fico_range_low"] = np.nan
    df["revol_util"] = rng.uniform(0, 100, n).round(1)
    df["initial_list_status"] = rng.choice(["w", "f"], n)
    df["application_type"] = "Individual"
    df["verification_status_joint"] = None
    df["disbursement_method"] = "Cash"
    df["last_pymnt_d"] = "Feb-2019"
    assert all(df[c].dtype != "float64" for c in STRING_COLUMNS), "every text column is set"
    return df[list(COLUMNS)]


@pytest.fixture
def lc_csv(tmp_path):
    """Synthetic loans written like the real file: gzip CSV with summary lines at the end."""
    path = tmp_path / "accepted.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
        synthetic_loans().to_csv(fh, index=False)
        for line in FOOTER:
            fh.write(f'"{line}"\n')
    return path


@pytest.fixture
def pg_url():
    url = get_settings().test_database_url
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    return url


@pytest.fixture
def migrated(pg_url):
    """A test database migrated to head, torn back down afterwards."""
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.attributes["url"] = pg_url
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")
    engine = create_engine(pg_url)
    yield engine
    engine.dispose()
    command.downgrade(cfg, "base")
