"""decisions: the API's audit trail

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-28
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "decisions",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("track", sa.String(16), nullable=False),
        sa.Column("model_version", sa.String(64), nullable=False),
        sa.Column("synthetic", sa.Boolean(), nullable=False),
        sa.Column("applicant_ref", sa.String(64), nullable=True),
        sa.Column("pd", sa.Float(), nullable=False),
        sa.Column("score", sa.Integer(), nullable=False),
        sa.Column("risk_band", sa.String(2), nullable=False),
        sa.Column("decision", sa.String(8), nullable=False),
        sa.Column("cutoff", sa.Float(), nullable=False),
        sa.Column("reasons", sa.JSON(), nullable=False),
        sa.Column("applicant", sa.JSON(), nullable=False),
        sa.CheckConstraint(
            "track IN ('lending_club', 'mobile_money')", name=op.f("ck_decisions_track_valid")
        ),
        sa.CheckConstraint(
            "decision IN ('approve', 'decline')", name=op.f("ck_decisions_decision_valid")
        ),
        sa.CheckConstraint("pd >= 0 AND pd <= 1", name=op.f("ck_decisions_pd_range")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_decisions")),
    )
    op.create_index("ix_decisions_created_at", "decisions", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_decisions_created_at", table_name="decisions")
    op.drop_table("decisions")
