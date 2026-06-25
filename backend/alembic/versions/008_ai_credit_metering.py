"""Add per-org AI usage metering: ai_usage_ledger + ai_org_quota.

Revision ID: 008_ai_credit_metering
Revises: 007_tokens
Create Date: 2026-06-24 00:00:00

Purely ADDITIVE — two new tables only; no existing column/table is altered.
This is the exact DDL approved in the plan, made idempotent (IF NOT EXISTS) so
it is safe to run whether or not create_all() already built the tables locally.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op


revision: str = "008_ai_credit_metering"
down_revision: Union[str, None] = "007_tokens"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1) append-only per-call usage ledger (billing source of truth)
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS ai_usage_ledger (
            id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            organization_id  VARCHAR(64),
            user_id          VARCHAR(64),
            feature          VARCHAR(32)  NOT NULL,
            tier             VARCHAR(16)  NOT NULL,
            model            VARCHAR(128) NOT NULL,
            input_tokens     INTEGER      NOT NULL DEFAULT 0,
            output_tokens    INTEGER      NOT NULL DEFAULT 0,
            credits          INTEGER      NOT NULL DEFAULT 0,
            request_id       VARCHAR(80),
            created_at       TIMESTAMPTZ  NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_ai_usage_ledger_org_time "
        "ON ai_usage_ledger (organization_id, created_at)"
    )
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_ai_usage_ledger_request "
        "ON ai_usage_ledger (request_id) WHERE request_id IS NOT NULL"
    )

    # 2) per-org allowance + current-period running counter
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS ai_org_quota (
            organization_id           VARCHAR(64) PRIMARY KEY,
            monthly_credit_allowance  INTEGER     NOT NULL,
            period_start              DATE        NOT NULL,
            credits_used_this_period  INTEGER     NOT NULL DEFAULT 0,
            status                    VARCHAR(16) NOT NULL DEFAULT 'active',
            updated_at                TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS ai_usage_ledger")
    op.execute("DROP TABLE IF EXISTS ai_org_quota")
