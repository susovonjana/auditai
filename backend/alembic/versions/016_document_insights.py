"""aura_document_insights — per-document AI reading (the document reader /
document_extraction agent). One row per (audit file, document, UI language),
fingerprinted by content_key so a re-upload is re-read.

Revision ID: 016_document_insights
Revises: 015_chunk_help_url
Create Date: 2026-10-05

Idempotent (IF NOT EXISTS) like 010-015 — safe to re-run.
"""
from alembic import op

revision = "016_document_insights"
down_revision = "015_chunk_help_url"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS aura_document_insights (
            id UUID PRIMARY KEY,
            organization_id TEXT,
            audit_file_id INTEGER NOT NULL,
            document_id INTEGER NOT NULL,
            document_reference TEXT,
            document_name TEXT,
            mime_type TEXT,
            language VARCHAR(8) NOT NULL DEFAULT 'en',
            content_key VARCHAR(128) NOT NULL,
            doc_type VARCHAR(64),
            summary_short TEXT,
            insight JSONB NOT NULL DEFAULT '{}'::jsonb,
            extracted_text TEXT,
            read_method VARCHAR(32),
            pages INTEGER,
            model TEXT,
            created_by TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_document_insights_file_doc_lang "
        "ON aura_document_insights (audit_file_id, document_id, language)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_document_insights_org "
        "ON aura_document_insights (organization_id)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS aura_document_insights")
