"""Add user_id and organization_id to search_history.

Revision ID: 005_qa_user_org
Revises: 004_processing_fields
Create Date: 2026-06-15 00:00:00
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "005_qa_user_org"
down_revision: Union[str, None] = "004_processing_fields"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "search_history",
        sa.Column("user_id", sa.Text, nullable=True),
    )
    op.add_column(
        "search_history",
        sa.Column("organization_id", sa.Text, nullable=True),
    )
    op.create_index(
        "ix_search_history_user_id", "search_history", ["user_id"],
    )
    op.create_index(
        "ix_search_history_organization_id",
        "search_history",
        ["organization_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_search_history_organization_id", table_name="search_history")
    op.drop_index("ix_search_history_user_id", table_name="search_history")
    op.drop_column("search_history", "organization_id")
    op.drop_column("search_history", "user_id")
