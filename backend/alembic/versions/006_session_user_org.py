"""Add user_id and organization_id to user_sessions.

Revision ID: 006_session_user_org
Revises: 005_qa_user_org
Create Date: 2026-06-15 00:00:00
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "006_session_user_org"
down_revision: Union[str, None] = "005_qa_user_org"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "user_sessions",
        sa.Column("user_id", sa.Text, nullable=True),
    )
    op.add_column(
        "user_sessions",
        sa.Column("organization_id", sa.Text, nullable=True),
    )
    op.create_index(
        "ix_user_sessions_user_id", "user_sessions", ["user_id"],
    )
    op.create_index(
        "ix_user_sessions_organization_id",
        "user_sessions",
        ["organization_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_user_sessions_organization_id", table_name="user_sessions")
    op.drop_index("ix_user_sessions_user_id", table_name="user_sessions")
    op.drop_column("user_sessions", "organization_id")
    op.drop_column("user_sessions", "user_id")
