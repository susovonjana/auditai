"""Add the agent runtime tables: agent_runs + agent_steps.

Revision ID: 009_agent_runtime
Revises: 008_ai_credit_metering
Create Date: 2026-06-28 00:00:00

Purely ADDITIVE — two new tables only; no existing column/table is altered.
Made idempotent (IF NOT EXISTS) so it is safe to run whether or not
create_all() already built the tables locally.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op


revision: str = "009_agent_runtime"
down_revision: Union[str, None] = "008_ai_credit_metering"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1) one row per supervised agent run (goal, plan, status, audit trail)
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_runs (
            id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            organization_id   TEXT,
            user_id           TEXT,
            audit_file_id     INTEGER      NOT NULL,
            agent_type        VARCHAR(64)  NOT NULL,
            goal              TEXT,
            status            VARCHAR(24)  NOT NULL DEFAULT 'planning',
            plan              JSONB        NOT NULL DEFAULT '[]'::jsonb,
            current_step_idx  INTEGER      NOT NULL DEFAULT 0,
            credits_used      INTEGER      NOT NULL DEFAULT 0,
            result_summary    JSONB,
            error_message     TEXT,
            created_by        TEXT,
            created_at        TIMESTAMPTZ  NOT NULL DEFAULT now(),
            updated_at        TIMESTAMPTZ  NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_agent_runs_org_created "
        "ON agent_runs (organization_id, created_at)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_agent_runs_audit_file "
        "ON agent_runs (audit_file_id)"
    )

    # 2) one row per planned step (tool, input/output, approval checkpoint)
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_steps (
            id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            run_id             UUID NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
            idx                INTEGER     NOT NULL,
            title              TEXT,
            type               VARCHAR(16) NOT NULL,
            tool               VARCHAR(64),
            input              JSONB       NOT NULL DEFAULT '{}'::jsonb,
            output             JSONB,
            status             VARCHAR(24) NOT NULL DEFAULT 'pending',
            requires_approval  BOOLEAN     NOT NULL DEFAULT false,
            proposed_write     JSONB,
            approved_by        TEXT,
            approved_at        TIMESTAMPTZ,
            created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at         TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_agent_steps_run_id "
        "ON agent_steps (run_id)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS agent_steps")
    op.execute("DROP TABLE IF EXISTS agent_runs")
