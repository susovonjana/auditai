"""agent_runs: working_paper_id + is_template (per-WP draft history + template mode)

Revision ID: 012_agent_run_wp_template
Revises: 011_proc_memory_provenance
Create Date: 2026-07-07

Idempotent (IF NOT EXISTS) like 010/011 — safe to re-run on a DB where the
columns already exist.
"""
from alembic import op

revision = "012_agent_run_wp_template"
down_revision = "011_proc_memory_provenance"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS working_paper_id INTEGER"
    )
    op.execute(
        "ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS is_template BOOLEAN "
        "NOT NULL DEFAULT FALSE"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_agent_runs_file_agent_created "
        "ON agent_runs (audit_file_id, agent_type, created_at)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_agent_runs_file_agent_created")
    op.execute("ALTER TABLE agent_runs DROP COLUMN IF EXISTS is_template")
    op.execute("ALTER TABLE agent_runs DROP COLUMN IF EXISTS working_paper_id")
