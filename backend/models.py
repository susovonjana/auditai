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
    DateTime,
    ForeignKey,
    BigInteger,
    func,
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
        Text, nullable=False, default="processing"
    )  # "processing" | "active" | "error"
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
