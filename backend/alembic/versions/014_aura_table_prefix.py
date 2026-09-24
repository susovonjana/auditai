"""aura_ table prefix for every auditai table (aura_admin_users, ...).

Revision ID: 014_aura_table_prefix
Revises: 009_search_history_audit_file, 013_agent_run_dismissed
Create Date: 2026-07-08

Product decision: every table in the auditai Postgres DB carries the `aura_`
prefix so they are recognisable next to other schemas, and every NEW table must
be created with it (set it directly in the model's __tablename__).

This is a pure RENAME — data, indexes, constraints and sequences all follow the
table; only the table names change. Index/constraint names are left as-is (they
don't collide and renaming them buys nothing). Also a MERGE point: it joins the
two migration heads (009_search_history_audit_file + 013_agent_run_dismissed)
into one.

Idempotent via IF EXISTS + a target-exists guard — safe to re-run, and safe on
a dev DB where create_all() already made the aura_* names directly.
"""
from alembic import op

revision = "014_aura_table_prefix"
down_revision = ("009_search_history_audit_file", "013_agent_run_dismissed")
branch_labels = None
depends_on = None

# every table of this project, in FK-safe order (renames don't cascade anyway,
# but keep parents first for readability)
TABLES = [
    "documents",
    "document_chunks",
    "user_sessions",
    "search_history",
    "admin_users",
    "proc_memory",
    "tb_mapping_memory",
    "ai_usage_ledger",
    "ai_org_quota",
    "audit_file_chunks",
    "audit_file_index",
    "audit_file_cache_state",
    "agent_runs",
    "agent_steps",
]


def _rename(old: str, new: str) -> None:
    # rename only when the source exists and the target doesn't — re-runnable,
    # and a no-op on DBs that already carry the new name
    op.execute(
        f"""
        DO $$
        BEGIN
            IF to_regclass('public.{old}') IS NOT NULL
               AND to_regclass('public.{new}') IS NULL THEN
                ALTER TABLE {old} RENAME TO {new};
            END IF;
        END $$;
        """
    )


def upgrade() -> None:
    for t in TABLES:
        _rename(t, f"aura_{t}")


def downgrade() -> None:
    for t in TABLES:
        _rename(f"aura_{t}", t)
