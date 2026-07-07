"""Add search_history.audit_file_id (nullable).

Revision ID: 009_search_history_audit_file
Revises: 008_ai_credit_metering
Create Date: 2026-07-02 00:00:00

Purely ADDITIVE — one nullable column + its index. Records which audit file a chat
question was asked INSIDE (file-grounded /copilot/chat); NULL for the general /ask
chat. Powers per-file tagging in the UI and keeps file turns out of the general
chat's conversation memory. No backfill (old rows read as general/NULL). Idempotent
(IF NOT EXISTS) so it is safe whether or not create_all() already added the column.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op


revision: str = "009_search_history_audit_file"
down_revision: Union[str, None] = "008_ai_credit_metering"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE search_history ADD COLUMN IF NOT EXISTS audit_file_id INTEGER"
    )
    # Matches SQLAlchemy's default name for index=True on this column, so
    # create_all() and this migration never collide.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_search_history_audit_file_id "
        "ON search_history (audit_file_id)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_search_history_audit_file_id")
    op.execute("ALTER TABLE search_history DROP COLUMN IF EXISTS audit_file_id")
