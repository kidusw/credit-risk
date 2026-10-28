"""Stream Lending Club's rejected applications (27.6M rows, 1.7 GB CSV) into parquet.

The file has nine columns. Five of them are shared with the accepted loans, and they are
renamed to the accepted file's names so one model can score both:

| Rejected file | Accepted file | Note |
|---|---|---|
| Amount Requested | loan_amnt | requested vs funded amount; the same for nearly all accepted |
| Application Date | issue_d | application date vs issue date: a few weeks apart |
| Risk_Score | fico_range_low | LC: FICO before 5 Nov 2013, VantageScore after (see README) |
| Debt-To-Income Ratio | dti | "10%" strings; the same sentinels as accepted are cleaned |
| Employment Length | emp_length_years | the same text categories |

State and zip are kept for reference only (Phase 5 keeps geography out of every model).
Only applications dated inside the train/valid/test windows are written.
"""

import gzip
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from crisk.data.build import _emp_years, assign_split

SOURCE = {
    "Amount Requested": "loan_amnt",
    "Application Date": "application_date",
    "Risk_Score": "fico_range_low",
    "Debt-To-Income Ratio": "dti",
    "Zip Code": "zip_code",
    "State": "addr_state",
    "Employment Length": "emp_length",
}
SCHEMA = pa.schema(
    [
        ("application_date", pa.timestamp("ns")),
        ("application_quarter", pa.string()),
        ("split", pa.string()),
        ("loan_amnt", pa.float32()),
        ("fico_range_low", pa.float32()),
        ("dti", pa.float32()),
        ("emp_length_years", pa.float32()),
        ("addr_state", pa.string()),
        ("zip_code", pa.string()),
    ]
)


@dataclass
class RejectedReport:
    rows_read: int
    rows_written: int
    by_split: dict


def transform(raw: pd.DataFrame) -> pd.DataFrame:
    """One chunk of the raw file -> typed rows inside a split window."""
    df = raw.rename(columns=SOURCE)
    df["application_date"] = pd.to_datetime(df["application_date"], errors="coerce")
    month = df["application_date"].dt.to_period("M").dt.to_timestamp()
    df["split"] = assign_split(month)
    df = df[df["split"].notna()].copy()
    dti = pd.to_numeric(df["dti"].astype(str).str.rstrip("%"), errors="coerce")
    df["dti"] = dti.where((dti >= 0) & (dti < 999)).astype("float32")  # as for accepted
    df["loan_amnt"] = pd.to_numeric(df["loan_amnt"], errors="coerce").astype("float32")
    df["fico_range_low"] = pd.to_numeric(df["fico_range_low"], errors="coerce").astype("float32")
    df["emp_length_years"] = _emp_years(df["emp_length"].astype("string"))
    df["application_quarter"] = df["application_date"].dt.to_period("Q").astype(str)
    df["split"] = df["split"].astype(str)
    return df[SCHEMA.names]


def rejected_to_parquet(src: Path, dest: Path, chunksize: int = 500_000) -> RejectedReport:
    dest.parent.mkdir(parents=True, exist_ok=True)
    read = written = 0
    counts: dict[str, int] = {}
    with gzip.open(src, "rt", encoding="utf-8") as fh, pq.ParquetWriter(dest, SCHEMA) as out:
        for chunk in pd.read_csv(fh, usecols=list(SOURCE), dtype=str, chunksize=chunksize):
            read += len(chunk)
            df = transform(chunk)
            if len(df):
                out.write_table(pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False))
                written += len(df)
                for k, v in df["split"].value_counts().items():
                    counts[k] = counts.get(k, 0) + int(v)
    return RejectedReport(rows_read=read, rows_written=written, by_split=counts)
