"""Relational schema: matured Lending Club loans, the vintage table and API decisions.

The full feature matrix lives in parquet; Postgres holds the loan-level facts the API,
dashboard and reports query (amounts, grade, outcome, split) plus the vintage summary.
"""

from datetime import date, datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    Numeric,
    SmallInteger,
    String,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Deterministic constraint names keep Alembic migrations stable across databases.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

# Where a row came from. Synthetic rows (Phase 6) must always be distinguishable.
DATA_SOURCES = ("lending_club", "synthetic")
SPLIT_NAMES = ("train", "valid", "test")


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class Loan(Base):
    __tablename__ = "loans"
    __table_args__ = (
        CheckConstraint("source IN ('lending_club', 'synthetic')", name="source_valid"),
        CheckConstraint("split IN ('train', 'valid', 'test')", name="split_valid"),
        CheckConstraint("term_months IN (36, 60)", name="term_valid"),
        Index("ix_loans_issue_d", "issue_d"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    source: Mapped[str] = mapped_column(String(16))
    issue_d: Mapped[date] = mapped_column(Date)
    issue_quarter: Mapped[str] = mapped_column(String(6))
    split: Mapped[str] = mapped_column(String(5))
    term_months: Mapped[int] = mapped_column(SmallInteger)
    loan_amnt: Mapped[float] = mapped_column(Numeric(12, 2))
    funded_amnt: Mapped[float] = mapped_column(Numeric(12, 2))
    int_rate: Mapped[float] = mapped_column(Float)
    installment: Mapped[float] = mapped_column(Numeric(10, 2))
    grade: Mapped[str] = mapped_column(String(1))
    sub_grade: Mapped[str] = mapped_column(String(2))
    purpose: Mapped[str] = mapped_column(String(32))
    home_ownership: Mapped[str] = mapped_column(String(16))
    addr_state: Mapped[str] = mapped_column(String(2))
    annual_inc: Mapped[float | None] = mapped_column(Float)
    dti: Mapped[float | None] = mapped_column(Float)
    fico_range_low: Mapped[float] = mapped_column(Float)
    loan_status: Mapped[str] = mapped_column(String(64))
    is_default: Mapped[bool] = mapped_column(Boolean)
    total_pymnt: Mapped[float | None] = mapped_column(Numeric(12, 2))
    recoveries: Mapped[float | None] = mapped_column(Numeric(12, 2))


class Vintage(Base):
    """Default rate per issue quarter and term, for every resolved loan (matured or not)."""

    __tablename__ = "vintages"
    __table_args__ = (CheckConstraint("defaults <= loans", name="defaults_le_loans"),)

    issue_quarter: Mapped[str] = mapped_column(String(6), primary_key=True)
    term_months: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    matured: Mapped[bool] = mapped_column(Boolean)
    loans: Mapped[int] = mapped_column(Integer)
    defaults: Mapped[int] = mapped_column(Integer)
    default_rate: Mapped[float] = mapped_column(Float)


TRACKS = ("lending_club", "mobile_money")


class Decision(Base):
    """One row per scoring request the API answered (Phase 8): the audit trail."""

    __tablename__ = "decisions"
    __table_args__ = (
        CheckConstraint("track IN ('lending_club', 'mobile_money')", name="track_valid"),
        CheckConstraint("decision IN ('approve', 'decline')", name="decision_valid"),
        CheckConstraint("pd >= 0 AND pd <= 1", name="pd_range"),
        Index("ix_decisions_created_at", "created_at"),
    )

    # BIGINT in Postgres; INTEGER in SQLite (the tests), where only that autoincrements.
    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    track: Mapped[str] = mapped_column(String(16))
    model_version: Mapped[str] = mapped_column(String(64))
    synthetic: Mapped[bool] = mapped_column(Boolean)
    applicant_ref: Mapped[str | None] = mapped_column(String(64))
    pd: Mapped[float] = mapped_column(Float)
    score: Mapped[int] = mapped_column(Integer)
    risk_band: Mapped[str] = mapped_column(String(2))
    decision: Mapped[str] = mapped_column(String(8))
    cutoff: Mapped[float] = mapped_column(Float)
    reasons: Mapped[list] = mapped_column(JSON)
    applicant: Mapped[dict] = mapped_column(JSON)
