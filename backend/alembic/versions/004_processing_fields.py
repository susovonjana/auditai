"""Add error_message column for background processing failures.

Revision ID: 004_processing_fields
Revises: 003_chunk_metadata
Create Date: 2025-01-04 00:00:00
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "004_processing_fields"
down_revision: Union[str, None] = "003_chunk_metadata"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("error_message", sa.Text, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("documents", "error_message")
