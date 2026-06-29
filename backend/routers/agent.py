"""
Agent router — start and supervise multi-step AI agent runs inside a 1audit
audit file.

Endpoints (all gated by AGENT_FEATURE_ENABLED + an org allowlist, metered, and
rate-limited exactly like the copilot routes):
  POST /agent/run                          start a run -> AgentRunOut
  GET  /agent/run/{run_id}                 poll the run (FE shows live progress)
  POST /agent/run/{run_id}/step/{idx}/approve
  POST /agent/run/{run_id}/step/{idx}/reject
  POST /agent/run/{run_id}/abort

Every mutating call carries a FRESH copilot grant; the runtime binds it to the
run's file/org before any read or write (least privilege across pauses).

NOTE: No `from __future__ import annotations` here — slowapi's decorator
interferes with FastAPI's resolution of stringified forward references (same
reason as routers/user.py and routers/copilot.py).
"""
import asyncio
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

import config
from config import RATE_LIMIT_ASK
from database import get_db
from rate_limit import limiter
from routers.user import _load_session
import schemas
import usage_meter
from copilot_tools import CopilotContext, CopilotGrantError

import agent.definitions  # noqa: F401  (importing registers every AgentDefinition)
from agent.models import AgentRun, AgentStep
from agent.runtime import AgentRuntime
from agent.types import AGENT_DEFINITIONS, RunContext

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agent", tags=["agent"])

_runtime = AgentRuntime()
_ACTIVE = ("planning", "running", "awaiting_approval")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _gate(organization_id: Optional[str]) -> None:
    """403 unless the feature is on AND this org is allowlisted (pilot gate).
    ``AGENT_ALLOWED_ORG_IDS=*`` opens it to every org (dev / open beta)."""
    if not config.AGENT_FEATURE_ENABLED:
        raise HTTPException(status_code=403, detail="The AI agent is not enabled.")
    allow = config.AGENT_ALLOWED_ORG_IDS
    if "*" in allow:
        return
    if not organization_id or organization_id not in allow:
        raise HTTPException(
            status_code=403, detail="The AI agent is not enabled for this organization."
        )


def _validate_grant(audit_file_id: int, grant: str) -> CopilotContext:
    """Build + validate this request's copilot context (maps a bad grant to 401)."""
    ctx = CopilotContext(audit_file_id, grant)
    try:
        ctx.validate_grant_local()
    except CopilotGrantError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    return ctx


async def _get_run(db: AsyncSession, run_id: str) -> AgentRun:
    try:
        rid = uuid.UUID(str(run_id))
    except (ValueError, TypeError):
        raise HTTPException(status_code=404, detail="Run not found.")
    res = await db.execute(select(AgentRun).where(AgentRun.id == rid))
    run = res.scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    return run


def _authorize_run(run: AgentRun, *, organization_id: Optional[str], audit_file_id: Optional[int] = None) -> None:
    """Decision D: the caller's org (and grant's file, when given) must match."""
    if run.organization_id and organization_id and run.organization_id != organization_id:
        raise HTTPException(status_code=403, detail="Run belongs to another organization.")
    if audit_file_id is not None and int(run.audit_file_id) != int(audit_file_id):
        raise HTTPException(status_code=403, detail="Grant is for a different audit file.")


async def _run_out(db: AsyncSession, run: AgentRun) -> schemas.AgentRunOut:
    res = await db.execute(
        select(AgentStep).where(AgentStep.run_id == run.id).order_by(AgentStep.idx)
    )
    steps = list(res.scalars().all())
    awaiting_idx = None
    proposed = None
    for s in steps:
        if s.status == "awaiting_approval":
            awaiting_idx = s.idx
            proposed = s.proposed_write
            break
    return schemas.AgentRunOut(
        id=run.id,
        status=run.status,
        agent_type=run.agent_type,
        goal=run.goal,
        audit_file_id=run.audit_file_id,
        plan=run.plan,
        steps=[schemas.AgentStepOut.model_validate(s) for s in steps],
        result_summary=run.result_summary,
        credits_used=run.credits_used or 0,
        error_message=run.error_message,
        awaiting_step_idx=awaiting_idx,
        proposed_write=proposed,
    )


async def _meter(db: AsyncSession, run: AgentRun, payload) -> None:
    """Write one org-ledger row for the LLM tokens this request spent (if any)."""
    usage = getattr(run, "_agent_usage", None) or {}
    await usage_meter.record_usage(
        db,
        organization_id=payload.organization_id,
        user_id=payload.user_id,
        feature="agent",
        tier="smart",
        usage=usage,
        request_id=uuid.uuid4().hex,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.post("/run")
@limiter.limit(RATE_LIMIT_ASK)
async def start_run(
    request: Request,
    payload: schemas.AgentRunRequest,
    db: AsyncSession = Depends(get_db),
):
    _gate(payload.organization_id)
    await _load_session(db, payload.session_token)
    await usage_meter.ensure_credits(db, payload.organization_id)

    definition = AGENT_DEFINITIONS.get(payload.agent_type)
    if definition is None:
        raise HTTPException(status_code=404, detail=f"Unknown agent type '{payload.agent_type}'.")
    _validate_grant(payload.audit_file_id, payload.copilot_grant)

    # Decision C: one active run per file — return the in-flight one if present.
    res = await db.execute(
        select(AgentRun)
        .where(AgentRun.audit_file_id == payload.audit_file_id, AgentRun.status.in_(_ACTIVE))
        .order_by(AgentRun.created_at.desc())
    )
    existing = res.scalars().first()
    if existing is not None:
        return await _run_out(db, existing)

    run = await _runtime.start(
        db,
        definition=definition,
        audit_file_id=payload.audit_file_id,
        grant=payload.copilot_grant,
        organization_id=payload.organization_id,
        user_id=payload.user_id,
        goal=payload.goal,
        language=payload.language,
    )
    await _meter(db, run, payload)
    return await _run_out(db, run)


@router.get("/run/{run_id}")
async def get_run(
    run_id: str,
    session_token: str,
    organization_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    await _load_session(db, session_token)
    run = await _get_run(db, run_id)
    _authorize_run(run, organization_id=(str(organization_id) if organization_id else None))
    return await _run_out(db, run)


@router.post("/run/{run_id}/step/{idx}/approve")
@limiter.limit(RATE_LIMIT_ASK)
async def approve_step(
    request: Request,
    run_id: str,
    idx: int,
    payload: schemas.AgentStepActionRequest,
    db: AsyncSession = Depends(get_db),
):
    await _load_session(db, payload.session_token)
    run = await _get_run(db, run_id)
    _gate(payload.organization_id or run.organization_id)
    await usage_meter.ensure_credits(db, payload.organization_id)
    ctx_copilot = _validate_grant(run.audit_file_id, payload.copilot_grant)
    _authorize_run(run, organization_id=payload.organization_id, audit_file_id=ctx_copilot.audit_file_id)

    run = await _runtime.approve(
        db,
        run,
        idx,
        RunContext(copilot=ctx_copilot, audit_file_id=run.audit_file_id, organization_id=run.organization_id),
        edited_payload=payload.edited_payload,
        approved_by=payload.user_id,
    )
    await _meter(db, run, payload)
    return await _run_out(db, run)


@router.post("/run/{run_id}/step/{idx}/reject")
@limiter.limit(RATE_LIMIT_ASK)
async def reject_step(
    request: Request,
    run_id: str,
    idx: int,
    payload: schemas.AgentStepActionRequest,
    db: AsyncSession = Depends(get_db),
):
    await _load_session(db, payload.session_token)
    run = await _get_run(db, run_id)
    _gate(payload.organization_id or run.organization_id)
    ctx_copilot = _validate_grant(run.audit_file_id, payload.copilot_grant)
    _authorize_run(run, organization_id=payload.organization_id, audit_file_id=ctx_copilot.audit_file_id)

    run = await _runtime.reject(
        db,
        run,
        idx,
        RunContext(copilot=ctx_copilot, audit_file_id=run.audit_file_id, organization_id=run.organization_id),
        note=payload.note,
    )
    await _meter(db, run, payload)
    return await _run_out(db, run)


@router.post("/run/{run_id}/abort")
@limiter.limit(RATE_LIMIT_ASK)
async def abort_run(
    request: Request,
    run_id: str,
    payload: schemas.AgentStepActionRequest,
    db: AsyncSession = Depends(get_db),
):
    await _load_session(db, payload.session_token)
    run = await _get_run(db, run_id)
    _authorize_run(run, organization_id=payload.organization_id)
    run = await _runtime.abort(db, run)
    return await _run_out(db, run)
