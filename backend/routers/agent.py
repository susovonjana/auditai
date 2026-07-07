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
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

import config
from config import RATE_LIMIT_ASK
from database import AsyncSessionLocal, get_db
import proc_memory
from rate_limit import limiter
from routers.user import _load_session
import schemas
import usage_meter
from copilot_tools import CopilotContext, CopilotGrantError

from models import ProcMemory

import agent.definitions  # noqa: F401  (importing registers every AgentDefinition)
from agent.models import AgentRun, AgentStep
from agent.registry import REGISTRY
from agent.runtime import AgentRuntime
from agent.types import AGENT_DEFINITIONS, RunContext

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agent", tags=["agent"])

_runtime = AgentRuntime()
_ACTIVE = ("planning", "running", "awaiting_approval")

# Strong refs to in-flight background advances (a bare create_task can be GC'd).
_BG_TASKS: set = set()


async def _advance_in_background(run_id, ctx: RunContext, payload: "schemas.AgentRunRequest") -> None:
    """Execute a freshly planned run's steps OUTSIDE the start request, in a
    session of its own. advance() commits after every step, so the FE's poll
    shows each step flip pending -> running -> done live instead of receiving
    the whole run at once when the checkpoint is reached."""
    async with AsyncSessionLocal() as db:
        try:
            res = await db.execute(select(AgentRun).where(AgentRun.id == run_id))
            run = res.scalar_one_or_none()
            if run is None:
                return
            await _runtime.advance(db, run, ctx)
            await _meter(db, run, payload)
        except Exception:  # noqa: BLE001 — never let a bg failure go unrecorded
            logger.exception("background agent advance failed (run %s)", run_id)
            try:
                res = await db.execute(select(AgentRun).where(AgentRun.id == run_id))
                run = res.scalar_one_or_none()
                if run is not None and run.status not in ("done", "aborted", "error"):
                    run.status = "error"
                    run.error_message = "internal error while executing the run"
                    await db.commit()
            except Exception:  # noqa: BLE001
                logger.exception("could not mark run %s as errored", run_id)


def _spawn_bg_advance(run_id, ctx: RunContext, payload: "schemas.AgentRunRequest") -> None:
    task = asyncio.create_task(_advance_in_background(run_id, ctx, payload))
    _BG_TASKS.add(task)
    task.add_done_callback(_BG_TASKS.discard)


# An org with fewer stored procedures than this gets a one-off seed import.
_SEED_MIN_ROWS = 10


def _step_output(steps, tool: str) -> dict:
    for s in steps:
        if s.tool == tool and isinstance(s.output, dict):
            return s.output
    return {}


async def _ingest_approved_program(run_id, user_id: Optional[str]) -> None:
    """Feedback loop (Phase 2): after an approved procedure write, store the
    APPROVED — i.e. auditor-edited — procedures in this org's proc_memory so
    future drafts converge on the firm's house style. Best-effort in its own
    session; idempotent via the content-hash dedupe."""
    async with AsyncSessionLocal() as db:
        try:
            res = await db.execute(select(AgentRun).where(AgentRun.id == run_id))
            run = res.scalar_one_or_none()
            if run is None or run.agent_type != "procedure_buildout":
                return
            sres = await db.execute(
                select(AgentStep).where(AgentStep.run_id == run.id).order_by(AgentStep.idx)
            )
            steps = list(sres.scalars().all())
            write_step = next(
                (
                    s for s in steps
                    if s.tool == "bulk_create_program_sections"
                    and s.status == "approved"
                    and isinstance(s.approved_payload, dict)
                ),
                None,
            )
            if write_step is None:
                return
            wp = _step_output(steps, "get_working_paper").get("working_paper") or {}
            risks = _step_output(steps, "get_risks").get("risks") or []
            sector = _step_output(steps, "get_audit_file_summary").get("sector")
            rows = proc_memory.extract_memory_rows(
                write_step.approved_payload,
                run_id=str(run.id),
                audit_area=(wp.get("name") if isinstance(wp, dict) else None),
                risk_summary=proc_memory.summarize_risks(risks) or None,
                client_sector=(str(sector) if sector else None),
                confirmed_by=(str(user_id) if user_id else None),
            )
            added = await proc_memory.add_memories(
                db, rows, organization_id=run.organization_id, source="agent_approved"
            )
            if added:
                logger.info(
                    "proc_memory ingest: run %s stored %d approved procedure(s)", run_id, added
                )
        except Exception:  # noqa: BLE001 — learning must never affect the approve
            logger.exception("proc_memory ingest failed (run %s)", run_id)


def _spawn_proc_memory_ingest(run_id, user_id: Optional[str]) -> None:
    task = asyncio.create_task(_ingest_approved_program(run_id, user_id))
    _BG_TASKS.add(task)
    task.add_done_callback(_BG_TASKS.discard)


async def _seed_org_proc_memory(copilot: CopilotContext, organization_id: Optional[str]) -> None:
    """One-off per org (Layer B seed): import the org's OWN existing procedures
    (templates first, then recent files — via the grant-scoped seed endpoint)
    into proc_memory, so even the very first drafts retrieve real house-style
    examples. Skips itself once the org has memory; content-hash dedupe makes
    re-runs no-ops. Best-effort in its own session."""
    if not organization_id:
        return
    async with AsyncSessionLocal() as db:
        try:
            res = await db.execute(
                select(func.count())
                .select_from(ProcMemory)
                .where(ProcMemory.organization_id == str(organization_id))
            )
            if int(res.scalar() or 0) >= _SEED_MIN_ROWS:
                return
            seed = await asyncio.to_thread(copilot.get, "org_procedure_seed")
            raw_rows = seed.get("rows") if isinstance(seed, dict) else None
            if not raw_rows:
                return
            rows = [
                {
                    "audit_area": r.get("audit_area"),
                    "risk_summary": None,
                    "client_sector": r.get("client_sector"),
                    "assertions": r.get("assertions") or [],
                    "procedure_html": r.get("procedure_html") or "",
                    "confirmed_by": None,
                    "source_key": f"seed:{r.get('source_audit_file_id')}:{r.get('section_id')}",
                }
                for r in raw_rows
                if isinstance(r, dict)
            ]
            added = await proc_memory.add_memories(
                db, rows, organization_id=organization_id, source="seed"
            )
            logger.info("proc_memory seed: org %s stored %d row(s)", organization_id, added)
        except Exception:  # noqa: BLE001 — seeding is an enhancement, never a blocker
            logger.exception("proc_memory seeding failed (org %s)", organization_id)


def _spawn_proc_memory_seed(copilot: CopilotContext, organization_id: Optional[str]) -> None:
    task = asyncio.create_task(_seed_org_proc_memory(copilot, organization_id))
    _BG_TASKS.add(task)
    task.add_done_callback(_BG_TASKS.discard)


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


async def _procedure_style_examples(
    db: AsyncSession, copilot: CopilotContext, payload: "schemas.AgentRunRequest"
) -> list:
    """House-style few-shots for the procedure agent: this firm's closest past
    procedures from proc_memory, keyed on the working paper's name. Best-effort —
    any failure (or an org with no history yet) just means no examples."""
    try:
        content = await asyncio.to_thread(
            copilot.get, f"working_papers/{int(payload.working_paper_id)}/content"
        )
        wp_name = ((content or {}).get("working_paper") or {}).get("name") if isinstance(content, dict) else None
        if not wp_name:
            return []
        rows = await proc_memory.search_examples(
            db,
            organization_id=payload.organization_id,
            client_sector=None,
            audit_area=str(wp_name),
            risk_summary=None,
            k=3,
        )
        return [r.procedure_html for r in rows if getattr(r, "procedure_html", None)]
    except Exception as exc:  # noqa: BLE001 — examples are an enhancement, never a blocker
        logger.warning("procedure style examples unavailable: %s", exc)
        return []


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
    copilot_ctx = _validate_grant(payload.audit_file_id, payload.copilot_grant)

    # Working-paper-scoped agents (procedure_buildout) can't plan without their WP.
    if getattr(definition, "requires_working_paper", False) and not payload.working_paper_id:
        raise HTTPException(
            status_code=422,
            detail=f"Agent '{payload.agent_type}' requires working_paper_id.",
        )

    # Decision C: one active run per file — return the in-flight one if present.
    res = await db.execute(
        select(AgentRun)
        .where(AgentRun.audit_file_id == payload.audit_file_id, AgentRun.status.in_(_ACTIVE))
        .order_by(AgentRun.created_at.desc())
    )
    existing = res.scalars().first()
    if existing is not None:
        return await _run_out(db, existing)

    # Firm house-style few-shots for the procedure drafter (best-effort), plus a
    # one-off background seed of the org's procedure memory on first use.
    style_examples: list = []
    if payload.agent_type == "procedure_buildout" and payload.working_paper_id:
        style_examples = await _procedure_style_examples(db, copilot_ctx, payload)
        _spawn_proc_memory_seed(copilot_ctx, payload.organization_id)

    # Persist the plan and respond IMMEDIATELY; the steps execute in a
    # background task (own session) and the FE watches them via its poll.
    run = await _runtime.start(
        db,
        definition=definition,
        audit_file_id=payload.audit_file_id,
        grant=payload.copilot_grant,
        organization_id=payload.organization_id,
        user_id=payload.user_id,
        goal=payload.goal,
        language=payload.language,
        working_paper_id=payload.working_paper_id,
        style_examples=style_examples,
        is_template=payload.is_template,
        defer_advance=True,
    )
    if run.status == "running":  # planned OK — execute in the background
        bg_ctx = RunContext(
            copilot=copilot_ctx,
            audit_file_id=payload.audit_file_id,
            language=payload.language,
            organization_id=payload.organization_id,
            working_paper_id=payload.working_paper_id,
            style_examples=style_examples,
            is_template=payload.is_template,
        )
        _spawn_bg_advance(run.id, bg_ctx, payload)
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
        RunContext(
            copilot=ctx_copilot,
            audit_file_id=run.audit_file_id,
            organization_id=run.organization_id,
            is_template=bool(run.is_template),
        ),
        edited_payload=payload.edited_payload,
        approved_by=payload.user_id,
    )
    await _meter(db, run, payload)
    # Feedback loop: learn the APPROVED procedures in the background (never
    # affects the approve response; the ingest re-checks step status itself).
    if run.agent_type == "procedure_buildout":
        _spawn_proc_memory_ingest(run.id, payload.user_id)
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
        RunContext(
            copilot=ctx_copilot,
            audit_file_id=run.audit_file_id,
            organization_id=run.organization_id,
            is_template=bool(run.is_template),
        ),
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


@router.post("/run/{run_id}/undo")
@limiter.limit(RATE_LIMIT_ASK)
async def undo_run(
    request: Request,
    run_id: str,
    payload: schemas.AgentStepActionRequest,
    db: AsyncSession = Depends(get_db),
):
    """One-click undo for the procedure build-out agent: soft-delete every
    section this run created (they all carry the run id in their config stamp).
    Idempotent — a second undo deletes nothing. Manual sections are untouched."""
    await _load_session(db, payload.session_token)
    run = await _get_run(db, run_id)
    _gate(payload.organization_id or run.organization_id)
    if not payload.copilot_grant:
        raise HTTPException(status_code=401, detail="A copilot grant is required to undo.")
    ctx_copilot = _validate_grant(run.audit_file_id, payload.copilot_grant)
    _authorize_run(run, organization_id=payload.organization_id, audit_file_id=ctx_copilot.audit_file_id)
    if run.agent_type != "procedure_buildout":
        raise HTTPException(status_code=409, detail="This agent's runs cannot be undone.")

    # Only a run whose write checkpoint actually ran has anything to undo.
    res = await db.execute(
        select(AgentStep).where(
            AgentStep.run_id == run.id,
            AgentStep.tool == "bulk_create_program_sections",
            AgentStep.status == "approved",
        )
    )
    write_step = res.scalars().first()
    if write_step is None:
        raise HTTPException(status_code=409, detail="This run has not written anything to undo.")
    wp_id = (
        (write_step.output or {}).get("working_paper_id")
        or (write_step.approved_payload or {}).get("working_paper_id")
        or (write_step.input or {}).get("working_paper_id")
    )
    if not wp_id:
        raise HTTPException(status_code=409, detail="Could not resolve the working paper for this run.")

    impls = REGISTRY.impls_for(ctx_copilot, ["undo_program_sections_run"])
    out = await asyncio.to_thread(
        impls["undo_program_sections_run"],
        {"working_paper_id": int(wp_id), "ai_run_id": str(run.id)},
    )
    if isinstance(out, dict) and "error" in out:
        raise HTTPException(status_code=502, detail=str(out.get("error")))

    deleted = int(out.get("deleted_count", 0)) if isinstance(out, dict) else 0
    # Record the undo on the run so the FE can reflect it after refetch, and
    # keep the EXACT deleted section ids — redo restores precisely these rows
    # (never sections the auditor deleted or added by hand in between).
    summary = dict(run.result_summary or {})
    summary["undone"] = True
    summary["undone_count"] = deleted
    if isinstance(out, dict) and out.get("section_ids"):
        summary["undone_section_ids"] = [int(s) for s in out["section_ids"]]
    run.result_summary = summary
    await db.commit()
    return {"run_id": str(run.id), "undone": True, "deleted_count": deleted}


@router.post("/run/{run_id}/redo")
@limiter.limit(RATE_LIMIT_ASK)
async def redo_run(
    request: Request,
    run_id: str,
    payload: schemas.AgentStepActionRequest,
    db: AsyncSession = Depends(get_db),
):
    """Reverse an undo: restore exactly the sections that undo soft-deleted
    (recorded on the run as undone_section_ids). Sections the auditor deleted
    or created by hand are never touched; idempotent per undo."""
    await _load_session(db, payload.session_token)
    run = await _get_run(db, run_id)
    _gate(payload.organization_id or run.organization_id)
    if not payload.copilot_grant:
        raise HTTPException(status_code=401, detail="A copilot grant is required to redo.")
    ctx_copilot = _validate_grant(run.audit_file_id, payload.copilot_grant)
    _authorize_run(run, organization_id=payload.organization_id, audit_file_id=ctx_copilot.audit_file_id)
    if run.agent_type != "procedure_buildout":
        raise HTTPException(status_code=409, detail="This agent's runs cannot be redone.")

    summary = dict(run.result_summary or {})
    section_ids = summary.get("undone_section_ids") or []
    if not summary.get("undone") or not section_ids:
        raise HTTPException(status_code=409, detail="This run has no undone draft to restore.")

    wp_id = run.working_paper_id
    if not wp_id:  # legacy run rows predate the working_paper_id column
        res = await db.execute(
            select(AgentStep).where(
                AgentStep.run_id == run.id,
                AgentStep.tool == "bulk_create_program_sections",
                AgentStep.status == "approved",
            )
        )
        write_step = res.scalars().first()
        if write_step is not None:
            wp_id = (
                (write_step.output or {}).get("working_paper_id")
                or (write_step.input or {}).get("working_paper_id")
            )
    if not wp_id:
        raise HTTPException(status_code=409, detail="Could not resolve the working paper for this run.")

    impls = REGISTRY.impls_for(ctx_copilot, ["restore_program_sections_run"])
    out = await asyncio.to_thread(
        impls["restore_program_sections_run"],
        {
            "working_paper_id": int(wp_id),
            "ai_run_id": str(run.id),
            "section_ids": [int(s) for s in section_ids],
        },
    )
    if isinstance(out, dict) and "error" in out:
        raise HTTPException(status_code=502, detail=str(out.get("error")))

    restored = int(out.get("restored_count", 0)) if isinstance(out, dict) else 0
    summary["undone"] = False
    summary["redone_count"] = restored
    run.result_summary = summary
    await db.commit()
    return {"run_id": str(run.id), "restored": True, "restored_count": restored}


@router.post("/run/{run_id}/dismiss")
@limiter.limit(RATE_LIMIT_ASK)
async def dismiss_run(
    request: Request,
    run_id: str,
    payload: schemas.AgentStepActionRequest,
    db: AsyncSession = Depends(get_db),
):
    """Remove a run from the draft-history list. Refuses while the run's draft
    is still LIVE in the working paper (undo it first — we never orphan active
    sections). A still-running / awaiting-approval run is aborted first, then
    hidden; a removed (undone) run's soft-deleted sections stay soft-deleted."""
    await _load_session(db, payload.session_token)
    run = await _get_run(db, run_id)
    _gate(payload.organization_id or run.organization_id)
    _authorize_run(run, organization_id=payload.organization_id)

    rs = run.result_summary if isinstance(run.result_summary, dict) else {}
    created = int(rs.get("sections_created") or 0)
    live_in_paper = run.status == "done" and created > 0 and not bool(rs.get("undone"))
    if live_in_paper:
        raise HTTPException(
            status_code=409,
            detail="This draft is still in the working paper. Undo it first, then remove it from history.",
        )
    if run.status in _ACTIVE:  # cancel a planning/running/awaiting run before hiding it
        run = await _runtime.abort(db, run)

    run.dismissed = True
    await db.commit()
    return {"run_id": str(run.id), "dismissed": True}


@router.get("/runs")
async def list_runs(
    session_token: str,
    audit_file_id: int,
    organization_id: str,
    working_paper_id: Optional[int] = None,
    agent_type: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Per-working-paper draft HISTORY: past runs with their undo/redo state,
    so the auditor can remove or restore ANY earlier AI draft — not only the
    one in the currently open panel session. Lightweight rows (no steps)."""
    await _load_session(db, session_token)
    _gate(organization_id)
    q = select(AgentRun).where(
        AgentRun.audit_file_id == int(audit_file_id),
        AgentRun.organization_id == str(organization_id),
        AgentRun.dismissed.is_(False),
    )
    if working_paper_id:
        q = q.where(AgentRun.working_paper_id == int(working_paper_id))
    if agent_type:
        q = q.where(AgentRun.agent_type == agent_type)
    q = q.order_by(AgentRun.created_at.desc()).limit(20)
    rows = (await db.execute(q)).scalars().all()
    runs = [_history_row(r) for r in rows]
    return {"runs": runs, "count": len(runs)}


def _history_row(r: AgentRun) -> dict:
    """One compact history entry (no steps). can_undo: the run finished and its
    draft is still in the working paper; can_redo: an undo happened and the
    exact deleted section ids were recorded. Pure — unit-tested offline."""
    rs = r.result_summary if isinstance(r.result_summary, dict) else {}
    created = int(rs.get("sections_created") or 0)
    undone = bool(rs.get("undone"))
    return {
        "id": str(r.id),
        "status": r.status,
        "agent_type": r.agent_type,
        "working_paper_id": r.working_paper_id,
        "is_template": bool(r.is_template),
        "goal": r.goal,
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "created_by": r.created_by,
        "sections_created": created,
        "summary": rs.get("summary"),
        "undone": undone,
        "undone_count": rs.get("undone_count"),
        "can_undo": r.status == "done" and created > 0 and not undone,
        "can_redo": undone and bool(rs.get("undone_section_ids")),
        # safe to remove from history only when nothing this run created is
        # still live in the working paper — a live draft must be undone first
        "can_delete": not (r.status == "done" and created > 0 and not undone),
    }
