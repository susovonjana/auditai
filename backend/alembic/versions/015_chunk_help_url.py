"""aura_document_chunks: help_url — the help-center page of the chunk's
article, captured at ingest from "Help page:" / "صفحة المساعدة:" marker lines
in the prepared KB export. Lets answers end with a clickable "Learn more" link.

Revision ID: 015_chunk_help_url
Revises: 014_aura_table_prefix
Create Date: 2026-07-09

Idempotent (IF NOT EXISTS) like 010-014 — safe to re-run.
"""
from alembic import op

revision = "015_chunk_help_url"
down_revision = "014_aura_table_prefix"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE aura_document_chunks ADD COLUMN IF NOT EXISTS help_url TEXT"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE aura_document_chunks DROP COLUMN IF EXISTS help_url")
