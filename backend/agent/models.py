"""
SQLAlchemy ORM models for the agent runtime.

Two tables, matching the style of ``models.py`` (UUID PKs, ``Mapped[...]`` /
``mapped_column``, JSONB, timezone-aware timestamps):

  - agent_runs    one row per agent run (the goal, plan, status, audit trail)
  - agent_steps   one row per planned step (tool, input/output, approval)
"""
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    String,
    Text,
    Integer,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    func,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


# ---------------------------------------------------------------------------
# agent_runs — one supervised run of one agent over one audit file
# ---------------------------------------------------------------------------
class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    organization_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    user_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    audit_file_id: Mapped[int] = mapped_column(Integer, nullable=False)
    agent_type: Mapped[str] = mapped_column(String(64), nullable=False)
    goal: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # planning | awaiting_approval | running | done | aborted | error
    status: Mapped[str] = mapped_column(
        String(24), nullable=False, default="planning"
    )
    plan: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    current_step_idx: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    # running total of AI credits spent by this run (per-run ceiling guardrail)
    credits_used: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    result_summary: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_by: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    steps: Mapped[list["AgentStep"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="AgentStep.idx",
    )

    __table_args__ = (
        Index("ix_agent_runs_org_created", "organization_id", "created_at"),
        Index("ix_agent_runs_audit_file", "audit_file_id"),
    )


# ---------------------------------------------------------------------------
# agent_steps — one planned step (read / compute / analysis / write)
# ---------------------------------------------------------------------------
class AgentStep(Base):
    __tablename__ = "agent_steps"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agent_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    idx: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    type: Mapped[str] = mapped_column(String(16), nullable=False)  # read|compute|analysis|write
    tool: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    input: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    output: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    # pending | running | done | awaiting_approval | approved | rejected | skipped | error
    # (24, not 16 — "awaiting_approval" is 17 chars; matches AgentRun.status width)
    status: Mapped[str] = mapped_column(
        String(24), nullable=False, default="pending"
    )
    requires_approval: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    # the exact change previewed to the auditor at a write checkpoint
    proposed_write: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    approved_by: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    approved_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    run: Mapped["AgentRun"] = relationship(back_populates="steps")
