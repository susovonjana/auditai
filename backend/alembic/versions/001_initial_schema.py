"""Initial schema — five tables + pgvector + ivfflat index.

Revision ID: 001_initial_schema
Revises:
Create Date: 2025-01-01 00:00:00
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = "001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ---- pgvector extension ----
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ---- documents ----
    op.create_table(
        "documents",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("filename", sa.Text, nullable=False),
        sa.Column("file_type", sa.Text, nullable=False),
        sa.Column("category", sa.Text, nullable=True),
        sa.Column("file_path", sa.Text, nullable=False),
        sa.Column("status", sa.Text, nullable=False, server_default="processing"),
        sa.Column("total_chunks", sa.Integer, nullable=False, server_default="0"),
        sa.Column("file_size_bytes", sa.BigInteger, nullable=False, server_default="0"),
        sa.Column("uploaded_by", sa.Text, nullable=False, server_default="admin"),
        sa.Column(
            "uploaded_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    # ---- document_chunks ----
    op.create_table(
        "document_chunks",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "document_id",
            sa.dialects.postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("embedding", Vector(384), nullable=False),  # all-MiniLM-L6-v2
        sa.Column("char_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_document_chunks_document_id",
        "document_chunks",
        ["document_id"],
    )
    # Fast approximate-NN index for cosine similarity
    op.execute(
        "CREATE INDEX document_chunks_embedding_idx "
        "ON document_chunks USING ivfflat (embedding vector_cosine_ops) "
        "WITH (lists = 100)"
    )

    # ---- user_sessions ----
    op.create_table(
        "user_sessions",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("session_token", sa.Text, nullable=False, unique=True),
        sa.Column("user_identifier", sa.Text, nullable=False, server_default="anonymous"),
        sa.Column("ip_address", sa.Text, nullable=True),
        sa.Column("user_agent", sa.Text, nullable=True),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "last_active_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("total_questions", sa.Integer, nullable=False, server_default="0"),
    )
    op.create_index("ix_user_sessions_session_token", "user_sessions", ["session_token"])

    # ---- search_history ----
    op.create_table(
        "search_history",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "session_id",
            sa.dialects.postgresql.UUID(as_uuid=True),
            sa.ForeignKey("user_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("question", sa.Text, nullable=False),
        sa.Column("question_embedding", Vector(384), nullable=True),  # all-MiniLM-L6-v2
        sa.Column("ai_answer", sa.Text, nullable=False, server_default=""),
        sa.Column(
            "chunks_used",
            sa.dialects.postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "documents_referenced",
            sa.dialects.postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "similarity_scores",
            sa.dialects.postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("response_time_ms", sa.Integer, nullable=False, server_default="0"),
        sa.Column("was_answered", sa.Boolean, nullable=False, server_default=sa.text("false")),
        sa.Column("user_feedback", sa.Text, nullable=True),
        sa.Column(
            "asked_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_search_history_session_id", "search_history", ["session_id"])
    op.create_index("ix_search_history_asked_at", "search_history", ["asked_at"])

    # ---- admin_users ----
    op.create_table(
        "admin_users",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("username", sa.String(64), nullable=False, unique=True),
        sa.Column("password_hash", sa.Text, nullable=False),
        sa.Column("role", sa.String(32), nullable=False, server_default="superadmin"),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_admin_users_username", "admin_users", ["username"])


def downgrade() -> None:
    op.drop_index("ix_admin_users_username", table_name="admin_users")
    op.drop_table("admin_users")

    op.drop_index("ix_search_history_asked_at", table_name="search_history")
    op.drop_index("ix_search_history_session_id", table_name="search_history")
    op.drop_table("search_history")

    op.drop_index("ix_user_sessions_session_token", table_name="user_sessions")
    op.drop_table("user_sessions")

    op.execute("DROP INDEX IF EXISTS document_chunks_embedding_idx")
    op.drop_index("ix_document_chunks_document_id", table_name="document_chunks")
    op.drop_table("document_chunks")

    op.drop_table("documents")
