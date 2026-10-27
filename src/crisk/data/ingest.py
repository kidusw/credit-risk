"""Stream the accepted-loans CSV (1.7 GB uncompressed) into a column-pruned parquet file.

The file never sits in memory whole: it is read in chunks, dropped columns are never parsed,
numerics are float32, and each chunk is appended to a parquet file with a fixed schema.
"""

import gzip
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from crisk.data.columns import COLUMNS, READ_COLUMNS, STRING_COLUMNS


class SourceSchemaError(ValueError):
    pass


@dataclass
class IngestReport:
    rows_read: int
    rows_written: int
    footer_rows: int
    chunks: int
    peak_rss_mb: float
    parquet_mb: float


def _rss_mb() -> float:
    try:
        import psutil
    except ImportError:  # optional: only used for the memory report
        return float("nan")
    return psutil.Process(os.getpid()).memory_info().rss / 2**20


def read_header(path: Path) -> list[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return pd.read_csv(fh, nrows=0).columns.tolist()


def check_header(header: list[str]) -> None:
    """Fail loudly if the source gained or lost columns since the audit was written."""
    unclassified = [c for c in header if c not in COLUMNS]
    missing = [c for c in COLUMNS if c not in header]
    if unclassified or missing:
        raise SourceSchemaError(
            f"leakage audit out of date: unclassified={unclassified} missing={missing}"
        )


ARROW_SCHEMA = pa.schema(
    [(c, pa.string() if c in STRING_COLUMNS else pa.float32()) for c in READ_COLUMNS]
)


def csv_to_parquet(src: Path, dest: Path, chunksize: int = 200_000) -> IngestReport:
    check_header(read_header(src))
    dtypes = {c: ("string" if c in STRING_COLUMNS else "float32") for c in READ_COLUMNS}
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    rows_read = rows_written = chunks = 0
    peak = _rss_mb()
    with pq.ParquetWriter(tmp, ARROW_SCHEMA, compression="zstd") as writer:
        reader = pd.read_csv(
            src, usecols=READ_COLUMNS, dtype=dtypes, chunksize=chunksize, low_memory=False
        )
        for chunk in reader:
            rows_read += len(chunk)
            # The file ends with summary lines ("Total amount funded in policy code ...")
            # that only fill the id column.
            chunk = chunk[chunk["loan_amnt"].notna()]
            table = pa.Table.from_pandas(
                chunk[READ_COLUMNS], schema=ARROW_SCHEMA, preserve_index=False
            )
            writer.write_table(table, row_group_size=chunksize)
            rows_written += len(chunk)
            chunks += 1
            peak = max(peak, _rss_mb())
    tmp.replace(dest)
    return IngestReport(
        rows_read=rows_read,
        rows_written=rows_written,
        footer_rows=rows_read - rows_written,
        chunks=chunks,
        peak_rss_mb=round(peak, 1),
        parquet_mb=round(dest.stat().st_size / 2**20, 1),
    )
