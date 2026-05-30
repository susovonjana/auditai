"""Add structural metadata columns to document_chunks.

Revision ID: 003_chunk_metadata
Revises: 002_fts_index
Create Date: 2025-01-03 00:00:00
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = "003_chunk_metadata"
down_revision: Union[str, None] = "002_fts_index"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "document_chunks",
        sa.Column("page_number", sa.Integer, nullable=True),
    )
    op.add_column(
        "document_chunks",
        sa.Column("section_heading", sa.Text, nullable=True),
    )
    op.add_column(
        "document_chunks",
        sa.Column(
            "chunk_type",
            sa.Text,
            nullable=False,
            server_default="text",
        ),
    )


def downgrade() -> None:
    op.drop_column("document_chunks", "chunk_type")
    op.drop_column("document_chunks", "section_heading")
    op.drop_column("document_chunks", "page_number")
