"""agent_runs: dismissed flag (remove a run from the draft-history list)

Revision ID: 013_agent_run_dismissed
Revises: 012_agent_run_wp_template
Create Date: 2026-07-07

Idempotent (IF NOT EXISTS) like 010-012 — safe to re-run.
"""
from alembic import op

revision = "013_agent_run_dismissed"
down_revision = "012_agent_run_wp_template"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS dismissed BOOLEAN "
        "NOT NULL DEFAULT FALSE"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE agent_runs DROP COLUMN IF EXISTS dismissed")
