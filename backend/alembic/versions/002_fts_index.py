"""Add full-text search index for hybrid retrieval.

Revision ID: 002_fts_index
Revises: 001_initial_schema
Create Date: 2025-01-02 00:00:00
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "002_fts_index"
down_revision: Union[str, None] = "001_initial_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # GIN expression index on the English tsvector of document_chunks.content.
    # Powers BM25-style keyword retrieval in the hybrid search layer.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS document_chunks_content_fts_idx
        ON document_chunks
        USING GIN (to_tsvector('english', content))
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS document_chunks_content_fts_idx")
