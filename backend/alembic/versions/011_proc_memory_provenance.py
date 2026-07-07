"""proc_memory provenance + dedupe — source, source_key, content_hash.

Revision ID: 011_proc_memory_provenance
Revises: 010_agent_step_approved_payload
Create Date: 2026-07-06

Purely ADDITIVE — three nullable TEXT columns plus a partial unique index.
`source` records how a row entered the memory (confirm | agent_approved | seed),
`source_key` points back at the origin (agent run/section or seed section) so a
future unlearn-on-undo is possible, and `content_hash` + the per-org partial
unique index dedupe re-ingestion of the same procedure text (markup/whitespace
differences hash identically). Idempotent (IF NOT EXISTS) so it is safe whether
or not create_all() already added the columns locally.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op


revision: str = "011_proc_memory_provenance"
down_revision: Union[str, None] = "010_agent_step_approved_payload"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE proc_memory ADD COLUMN IF NOT EXISTS source TEXT")
    op.execute("ALTER TABLE proc_memory ADD COLUMN IF NOT EXISTS source_key TEXT")
    op.execute("ALTER TABLE proc_memory ADD COLUMN IF NOT EXISTS content_hash TEXT")
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_proc_memory_org_hash "
        "ON proc_memory (organization_id, content_hash) "
        "WHERE content_hash IS NOT NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_proc_memory_org_hash")
    op.execute("ALTER TABLE proc_memory DROP COLUMN IF EXISTS content_hash")
    op.execute("ALTER TABLE proc_memory DROP COLUMN IF EXISTS source_key")
    op.execute("ALTER TABLE proc_memory DROP COLUMN IF EXISTS source")
