"""Add agent_steps.approved_payload — feedback capture for write checkpoints.

Revision ID: 010_agent_step_approved_payload
Revises: 009_agent_runtime
Create Date: 2026-07-06

Purely ADDITIVE — one nullable JSONB column. It stores the payload that actually
ran when the auditor approved a write checkpoint (edited or not), so
proposed_write vs approved_payload diffs can grade how much auditors change the
agent's drafts (Phase-1 feedback loop for the procedure build-out agent).
Idempotent (IF NOT EXISTS) so it is safe whether or not create_all() already
added the column locally.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op


revision: str = "010_agent_step_approved_payload"
down_revision: Union[str, None] = "009_agent_runtime"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE agent_steps ADD COLUMN IF NOT EXISTS approved_payload JSONB"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE agent_steps DROP COLUMN IF EXISTS approved_payload")
