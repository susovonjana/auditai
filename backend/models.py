"""
SQLAlchemy ORM models for AuditAI.

Five tables, exactly as defined in the project brief:
  - documents
  - document_chunks   (with VECTOR(1536) embedding column via pgvector)
  - user_sessions
  - search_history    (with VECTOR(1536) question_embedding column)
  - admin_users
"""
import uuid
from datetime import datetime
from typing import Optional, List, Any

from sqlalchemy import (
    String,
    Text,
    Integer,
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    BigInteger,
    Index,
    func,
    text as text_sql,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from database import Base
from config import EMBEDDING_DIMENSIONS


# ---------------------------------------------------------------------------
# Table 1: documents
# ---------------------------------------------------------------------------
class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    file_type: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        Text, nullable=False, default="queued"
    )  # "queued" | "parsing" | "embedding" | "active" | "error"
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    total_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    uploaded_by: Mapped[str] = mapped_column(Text, nullable=False, default="admin")
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    chunks: Mapped[List["DocumentChunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


# ---------------------------------------------------------------------------
# Table 2: document_chunks  (the heart of the knowledge base)
# ---------------------------------------------------------------------------
class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[Any] = mapped_column(
        Vector(EMBEDDING_DIMENSIONS), nullable=False
    )
    char_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # New in migration 003 — optional structural metadata about the chunk.
    page_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    section_heading: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    chunk_type: Mapped[str] = mapped_column(
        Text, nullable=False, default="text"
    )  # "text" | "table"
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    document: Mapped["Document"] = relationship(back_populates="chunks")


# ---------------------------------------------------------------------------
# Table 3: user_sessions
# ---------------------------------------------------------------------------
class UserSession(Base):
    __tablename__ = "user_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_token: Mapped[str] = mapped_column(
        Text, unique=True, nullable=False, index=True
    )
    user_identifier: Mapped[str] = mapped_column(
        Text, nullable=False, default="anonymous"
    )
    ip_address: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_active_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    ended_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    total_questions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Identity passed in from the embedding app (e.g. 1audit) at session
    # creation. Nullable so anonymous traffic still saves.
    user_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True, index=True)
    organization_id: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, index=True
    )

    history: Mapped[List["SearchHistory"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


# ---------------------------------------------------------------------------
# Table 4: search_history
# ---------------------------------------------------------------------------
class SearchHistory(Base):
    __tablename__ = "search_history"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("user_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    question_embedding: Mapped[Optional[Any]] = mapped_column(
        Vector(EMBEDDING_DIMENSIONS), nullable=True
    )
    ai_answer: Mapped[str] = mapped_column(Text, nullable=False, default="")
    chunks_used: Mapped[Any] = mapped_column(JSONB, nullable=False, default=list)
    documents_referenced: Mapped[Any] = mapped_column(
        JSONB, nullable=False, default=list
    )
    similarity_scores: Mapped[Any] = mapped_column(JSONB, nullable=False, default=list)
    response_time_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    was_answered: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    user_feedback: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # "helpful" | "not_helpful" | NULL
    # Identity passed in from the embedding app (e.g. 1audit). Nullable so
    # legacy/anonymous traffic still saves.
    user_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True, index=True)
    organization_id: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, index=True
    )
    # LLM token usage for this answer. 0 / NULL when no Gemini call happened
    # (small-talk replies, cache hits, empty-KB short-circuits).
    prompt_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    asked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    session: Mapped["UserSession"] = relationship(back_populates="history")


# ---------------------------------------------------------------------------
# Table 5: admin_users
# ---------------------------------------------------------------------------
class AdminUser(Base):
    __tablename__ = "admin_users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    username: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, index=True
    )
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str] = mapped_column(
        String(32), nullable=False, default="superadmin"
    )  # "superadmin" | "admin"
    last_login_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


# ---------------------------------------------------------------------------
# Table 6: proc_memory  (ticket A-1b — firm "house style" procedure memory)
# ---------------------------------------------------------------------------
class ProcMemory(Base):
    """An accepted audit procedure, embedded by (audit_area + risk_summary) so a
    future draft can retrieve THIS firm's closest past procedures as few-shot
    examples and increasingly match their house style.

    Strictly org-scoped and purely additive: rows live only in auditai's own
    Postgres, are never read or written by 1audit, and are only appended to (no
    update/delete in app code)."""
    __tablename__ = "proc_memory"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    organization_id: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, index=True
    )
    client_sector: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    audit_area: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    risk_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    assertions: Mapped[Any] = mapped_column(JSONB, nullable=False, default=list)
    procedure_html: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[Any] = mapped_column(
        Vector(EMBEDDING_DIMENSIONS), nullable=False
    )
    confirmed_by: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    confirmed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )


class TbMappingMemory(Base):
    """A confirmed trial-balance account -> chart-of-account decision, embedded by
    its normalised account text so a future "AI auto map" run can retrieve the
    firm's closest past mappings (ticket C-2).

    This is the learning store behind both signals the feature uses:
      * "previous data"  — this client's own prior confirmed mappings.
      * "organization trend" — the whole firm's confirmed mappings (same org),
        preferring the same client sector.
    A reserved organization_id may also hold a curated "prime"/golden seed so a
    brand-new org/client still gets suggestions on day one (cold start).

    Strictly org-scoped and purely additive: rows live only in auditai's own
    Postgres, are never read or written by 1audit, and are only appended to (no
    update/delete in app code)."""
    __tablename__ = "tb_mapping_memory"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # Mandatory retrieval scope. A reserved id (the "prime" seed) is also stored
    # here, so cold-start lookups reuse the exact same search path.
    organization_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    client_sector: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # The normalised "name (+ second language) + code" text we embed and match on.
    account_text: Mapped[str] = mapped_column(Text, nullable=False)
    # Raw account code, kept verbatim for exact-code lookups and the btree index.
    account_code: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # The confirmed mapping target (1audit ChartOfAccount.id).
    coa_original_id: Mapped[int] = mapped_column(Integer, nullable=False)
    coa_label: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    embedding: Mapped[Any] = mapped_column(
        Vector(EMBEDDING_DIMENSIONS), nullable=False
    )
    confirmed_by: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    confirmed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    __table_args__ = (
        # Fast exact/near lookups by code within an org (Tier-1 "previous data").
        Index("tb_mapping_memory_org_code_idx", "organization_id", "account_code"),
    )


# ---------------------------------------------------------------------------
# Table 8: ai_usage_ledger  — per-call LLM usage (billing source of truth)
# ---------------------------------------------------------------------------
class AiUsageLedger(Base):
    """One append-only row per Bedrock call: which org/feature/tier/model, the
    token counts, and the normalised credit cost. This is the source of truth
    for per-org AI spend and analytics (procedure/findings/tb-tail never write
    to search_history, so the ledger captures every feature uniformly).

    Purely additive: a new table in auditai's own Postgres, never read/written
    by 1audit. The canonical DDL also lives in
    migrations/004_ai_credit_metering.sql (used for prod / manual apply)."""
    __tablename__ = "ai_usage_ledger"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    organization_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    user_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    feature: Mapped[str] = mapped_column(String(32), nullable=False)
    tier: Mapped[str] = mapped_column(String(16), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    credits: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Idempotency key — a retried request with the same id is recorded once.
    request_id: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_ai_usage_ledger_org_time", "organization_id", "created_at"),
        # Partial unique index: idempotency for non-null request ids only.
        Index(
            "ux_ai_usage_ledger_request",
            "request_id",
            unique=True,
            postgresql_where=text_sql("request_id IS NOT NULL"),
        ),
    )


# ---------------------------------------------------------------------------
# Table 9: ai_org_quota  — per-org monthly allowance + current-period counter
# ---------------------------------------------------------------------------
class AiOrgQuota(Base):
    """One row per organization holding its monthly AI-credit allowance and a
    running counter for the current period (reset on the first call of a new
    month). A denormalised counter so the pre-flight cap check is a single cheap
    read; the ai_usage_ledger remains the auditable source of truth.

    Orgs with no row fall back to AI_MONTHLY_CREDIT_ALLOWANCE_DEFAULT; a row is
    created lazily on first use. Purely additive (auditai Postgres only)."""
    __tablename__ = "ai_org_quota"

    organization_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    monthly_credit_allowance: Mapped[int] = mapped_column(Integer, nullable=False)
    period_start: Mapped[Any] = mapped_column(Date, nullable=False)
    credits_used_this_period: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


# ---------------------------------------------------------------------------
# Table 10: audit_file_chunks  (phase-2 per-file RAG — semantic search over ONE
# audit file's working-paper narrative)
# ---------------------------------------------------------------------------
class AuditFileChunk(Base):
    """A chunk of a single audit file's working-paper NARRATIVE (procedure
    questions, the auditor's notes/free-text answers, titles, comments), embedded
    so the copilot's search_file tool can semantically find "which WP discusses
    going concern / related parties / …" without sweeping every WP.

    Scoped strictly to one audit_file_id. Built lazily and REPLACED wholesale on
    reindex (the file's content changes as auditors work), so rows are transient
    cache, never a source of truth — 1audit-be remains authoritative. Purely
    additive: lives only in auditai's own Postgres."""
    __tablename__ = "audit_file_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    audit_file_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[Any] = mapped_column(
        Vector(EMBEDDING_DIMENSIONS), nullable=False
    )
    # Where this chunk came from in the file, e.g. "C1 - Checklist Final" — shown
    # to the model so it can cite / drill into the working paper.
    source_ref: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


# ---------------------------------------------------------------------------
# Table 11: audit_file_index  (freshness bookkeeping for the per-file RAG index)
# ---------------------------------------------------------------------------
class AuditFileIndex(Base):
    """One row per indexed audit file: the change-signature it was built from, a
    content hash (to skip re-embedding when nothing material changed), the chunk
    count and when it was built. Lets search_file decide cheaply whether to reuse
    the existing chunks or rebuild. Purely additive (auditai Postgres only)."""
    __tablename__ = "audit_file_index"

    audit_file_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    signature: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    content_hash: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    indexed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


# ---------------------------------------------------------------------------
# Table 12: audit_file_cache_state  — per-file "data changed" marker for the
# copilot's live-data cache invalidation
# ---------------------------------------------------------------------------
class AuditFileCacheState(Base):
    """One row per audit file recording WHEN its data last changed in 1audit.
    1audit-be pings auditai on any edit → ``changed_at`` is bumped; each file-chat
    request compares it to what this worker last applied and clears that file's
    in-memory data cache when it's newer. Shared source of truth so the
    invalidation is correct across multiple auditai workers. Purely additive
    (auditai Postgres only)."""
    __tablename__ = "audit_file_cache_state"

    audit_file_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    changed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
