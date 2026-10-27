"""loans and vintages

Revision ID: 0001
Revises:
Create Date: 2026-10-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "loans",
        sa.Column("id", sa.BigInteger(), autoincrement=False, nullable=False),
        sa.Column("source", sa.String(16), nullable=False),
        sa.Column("issue_d", sa.Date(), nullable=False),
        sa.Column("issue_quarter", sa.String(6), nullable=False),
        sa.Column("split", sa.String(5), nullable=False),
        sa.Column("term_months", sa.SmallInteger(), nullable=False),
        sa.Column("loan_amnt", sa.Numeric(12, 2), nullable=False),
        sa.Column("funded_amnt", sa.Numeric(12, 2), nullable=False),
        sa.Column("int_rate", sa.Float(), nullable=False),
        sa.Column("installment", sa.Numeric(10, 2), nullable=False),
        sa.Column("grade", sa.String(1), nullable=False),
        sa.Column("sub_grade", sa.String(2), nullable=False),
        sa.Column("purpose", sa.String(32), nullable=False),
        sa.Column("home_ownership", sa.String(16), nullable=False),
        sa.Column("addr_state", sa.String(2), nullable=False),
        sa.Column("annual_inc", sa.Float(), nullable=True),
        sa.Column("dti", sa.Float(), nullable=True),
        sa.Column("fico_range_low", sa.Float(), nullable=False),
        sa.Column("loan_status", sa.String(64), nullable=False),
        sa.Column("is_default", sa.Boolean(), nullable=False),
        sa.Column("total_pymnt", sa.Numeric(12, 2), nullable=True),
        sa.Column("recoveries", sa.Numeric(12, 2), nullable=True),
        sa.CheckConstraint(
            "source IN ('lending_club', 'synthetic')", name=op.f("ck_loans_source_valid")
        ),
        sa.CheckConstraint(
            "split IN ('train', 'valid', 'test')", name=op.f("ck_loans_split_valid")
        ),
        sa.CheckConstraint("term_months IN (36, 60)", name=op.f("ck_loans_term_valid")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_loans")),
    )
    op.create_index("ix_loans_issue_d", "loans", ["issue_d"])
    op.create_table(
        "vintages",
        sa.Column("issue_quarter", sa.String(6), nullable=False),
        sa.Column("term_months", sa.SmallInteger(), nullable=False),
        sa.Column("matured", sa.Boolean(), nullable=False),
        sa.Column("loans", sa.Integer(), nullable=False),
        sa.Column("defaults", sa.Integer(), nullable=False),
        sa.Column("default_rate", sa.Float(), nullable=False),
        sa.CheckConstraint("defaults <= loans", name=op.f("ck_vintages_defaults_le_loans")),
        sa.PrimaryKeyConstraint("issue_quarter", "term_months", name=op.f("pk_vintages")),
    )


def downgrade() -> None:
    op.drop_table("vintages")
    op.drop_index("ix_loans_issue_d", table_name="loans")
    op.drop_table("loans")
