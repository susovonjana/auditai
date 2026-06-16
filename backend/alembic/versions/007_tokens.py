"""Add prompt_tokens, completion_tokens, total_tokens to search_history.

Revision ID: 007_tokens
Revises: 006_session_user_org
Create Date: 2026-06-15 00:00:00
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "007_tokens"
down_revision: Union[str, None] = "006_session_user_org"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "search_history",
        sa.Column("prompt_tokens", sa.Integer, nullable=True),
    )
    op.add_column(
        "search_history",
        sa.Column("completion_tokens", sa.Integer, nullable=True),
    )
    op.add_column(
        "search_history",
        sa.Column("total_tokens", sa.Integer, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("search_history", "total_tokens")
    op.drop_column("search_history", "completion_tokens")
    op.drop_column("search_history", "prompt_tokens")
