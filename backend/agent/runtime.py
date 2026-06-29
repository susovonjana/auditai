"""
The agent runtime — plan -> execute -> checkpoint, resumable and auditable.

``AgentRuntime`` drives an ``AgentDefinition`` over one audit file:
  * start    create the run, persist one step per planned step, then advance.
  * advance  execute steps in order, COMMITTING each status transition so a
             concurrent poll sees live progress; PAUSE before any approval-gated
             write (status -> awaiting_approval) and return.
  * approve  run the held write tool once (idempotent) with the approved/edited
             payload, then advance.
  * reject   skip the checkpoint (v1: skip, don't re-plan), then advance.
  * abort    stop the run.

Guardrails: plan length <= AGENT_STEP_LIMIT, per-advance wall-clock budget
AGENT_RUN_TIMEOUT_SEC, and a per-run credit ceiling AGENT_MAX_RUN_CREDITS checked
before every LLM step. The runtime NEVER computes a financial figure; numbers
come from tool outputs or the definition's pure-Python compute steps. Any tool
returning ``{"error": ...}`` on a required step ends the run as ``error`` with
nothing half-written.
"""
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

import config
from agent.models import AgentRun, AgentStep
from agent.registry import REGISTRY
from agent.types import AgentDefinition, PlannedStep, RunContext, StepResult
from copilot_tools import CopilotContext

_TERMINAL_STEP = ("done", "approved", "skipped")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _jsonsafe(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def _is_error(value: Any) -> bool:
    return isinstance(value, dict) and "error" in value


def _credits_from_usage(usage: Dict[str, Any], tier: str = "smart") -> int:
    """Mirror usage_meter's credit formula so the per-run ceiling uses the same
    unit as billing (the authoritative ledger is still written by the route)."""
    in_rate, out_rate = config.AI_CREDIT_TIER_RATES.get(
        tier, config.AI_CREDIT_TIER_RATES["smart"]
    )
    inp = int(usage.get("input", 0) or 0)
    out = int(usage.get("output", 0) or 0)
    return math.ceil(inp / 1000.0 * in_rate + out / 1000.0 * out_rate)


def _accumulate_usage(run: AgentRun, usage: Dict[str, Any]) -> None:
    """Sum an LLM call's tokens onto the run object as a transient attribute so
    the router can write ONE org-ledger row per request (real billing) without a
    new column. Not persisted; rebuilt per request, which is what we want."""
    if not usage:
        return
    tot = getattr(run, "_agent_usage", None) or {"input": 0, "output": 0, "model": ""}
    tot["input"] += int(usage.get("input", 0) or 0)
    tot["output"] += int(usage.get("output", 0) or 0)
    if usage.get("model"):
        tot["model"] = usage["model"]
    run._agent_usage = tot


def _call_kwargs(fn, kwargs):
    return fn(**(kwargs or {}))


def _call_payload(fn, payload):
    return fn(payload or {})


def _planned_from(step: AgentStep) -> PlannedStep:
    return PlannedStep(
        title=step.title or "",
        type=step.type,
        tool=step.tool,
        args=dict(step.input or {}),
        requires_approval=bool(step.requires_approval),
    )


class AgentRuntime:
    # ------------------------------------------------------------------ start
    async def start(
        self,
        db: AsyncSession,
        *,
        definition: AgentDefinition,
        audit_file_id: int,
        grant: str,
        organization_id: Optional[str],
        user_id: Optional[str],
        goal: Optional[str],
        language: str = "en",
        base_url: Optional[str] = None,
    ) -> AgentRun:
        copilot = CopilotContext(audit_file_id, grant, base_url=base_url)
        run_ctx = RunContext(
            copilot=copilot,
            audit_file_id=audit_file_id,
            language=language,
            organization_id=organization_id,
        )

        run = AgentRun(
            organization_id=organization_id,
            user_id=user_id,
            audit_file_id=audit_file_id,
            agent_type=definition.agent_type,
            goal=goal or definition.default_goal(audit_file_id),
            status="planning",
            created_by=user_id,
        )
        db.add(run)
        await db.flush()  # assign run.id

        plan: List[PlannedStep] = definition.build_plan(run_ctx)
        if len(plan) > config.AGENT_STEP_LIMIT:
            run.status = "error"
            run.error_message = (
                f"plan has {len(plan)} steps, exceeds AGENT_STEP_LIMIT="
                f"{config.AGENT_STEP_LIMIT}"
            )
            await db.commit()
            return run

        run.plan = [asdict(p) for p in plan]
        for i, p in enumerate(plan):
            db.add(
                AgentStep(
                    run_id=run.id,
                    idx=i,
                    title=p.title,
                    type=p.type,
                    tool=p.tool,
                    input=p.args or {},
                    requires_approval=bool(p.requires_approval),
                    status="pending",
                )
            )
        run.status = "running"
        await db.commit()

        return await self.advance(db, run, run_ctx)

    # ---------------------------------------------------------------- advance
    async def advance(
        self, db: AsyncSession, run: AgentRun, ctx: RunContext
    ) -> AgentRun:
        if run.status in ("done", "aborted", "error"):
            return run

        steps = await self._load_steps(db, run.id)
        ctx.results = self._rebuild_results(steps)
        deadline = time.monotonic() + config.AGENT_RUN_TIMEOUT_SEC

        for step in steps:
            if step.idx < run.current_step_idx or step.status in _TERMINAL_STEP:
                continue

            # Pause before an approval-gated write that hasn't been approved yet.
            if (
                step.type == "write"
                and step.requires_approval
                and step.status != "approved"
            ):
                # Populate the preview the auditor approves, derived from earlier
                # steps by the definition (if it computes one and none is set yet).
                if step.proposed_write is None:
                    definition = _definition_for(run.agent_type)
                    prepare = getattr(definition, "prepare_write", None)
                    if prepare is not None:
                        try:
                            payload = await asyncio.to_thread(prepare, _planned_from(step), ctx)
                            step.proposed_write = _jsonsafe(payload)
                        except Exception as exc:  # noqa: BLE001
                            return await self._fail(
                                db, run, step,
                                f"prepare_write failed: {type(exc).__name__}: {exc}",
                            )
                step.status = "awaiting_approval"
                run.status = "awaiting_approval"
                await db.commit()
                return run

            if time.monotonic() > deadline:
                return await self._fail(db, run, step, "run wall-clock budget exceeded")

            # Per-run credit ceiling, checked before any LLM (analysis) step.
            if step.type == "analysis" and run.credits_used >= config.AGENT_MAX_RUN_CREDITS:
                return await self._fail(
                    db, run, step,
                    f"run credit ceiling {config.AGENT_MAX_RUN_CREDITS} reached",
                )

            step.status = "running"
            await db.commit()

            try:
                out = await self._execute(step, ctx, run)
            except Exception as exc:  # noqa: BLE001
                # A WRITE failing is fatal (never half-write). A read/compute/
                # analysis failing is recorded and the run continues — a review
                # stays resilient and reports what it couldn't get as a gap.
                if step.type == "write":
                    return await self._fail(db, run, step, f"{type(exc).__name__}: {exc}")
                out = {"error": f"{type(exc).__name__}: {exc}"}

            if _is_error(out) and step.type == "write":
                return await self._fail(db, run, step, str(out.get("error")), output=out)

            safe = _jsonsafe(out)
            step.output = safe
            step.status = "done"
            run.current_step_idx = step.idx + 1
            ctx.results.append(
                StepResult(step.idx, step.title or "", step.type, step.tool, dict(step.input or {}), safe)
            )
            await db.commit()

        # All steps complete -> synthesize the final result.
        if run.credits_used >= config.AGENT_MAX_RUN_CREDITS:
            run.status = "error"
            run.error_message = f"run credit ceiling {config.AGENT_MAX_RUN_CREDITS} reached"
            await db.commit()
            return run
        try:
            definition = _definition_for(run.agent_type)
            ctx.usage_out.clear()
            summary = await asyncio.to_thread(definition.synthesize, ctx)
            if ctx.usage_out:
                run.credits_used += _credits_from_usage(ctx.usage_out)
                _accumulate_usage(run, ctx.usage_out)
        except Exception as exc:  # noqa: BLE001
            run.status = "error"
            run.error_message = f"synthesize failed: {type(exc).__name__}: {exc}"
            await db.commit()
            return run

        run.result_summary = _jsonsafe(summary)
        run.status = "done"
        await db.commit()
        return run

    # ---------------------------------------------------------------- approve
    async def approve(
        self,
        db: AsyncSession,
        run: AgentRun,
        idx: int,
        ctx: RunContext,
        *,
        edited_payload: Optional[dict] = None,
        approved_by: Optional[str] = None,
    ) -> AgentRun:
        step = await self._step(db, run.id, idx)
        # Idempotent: only act if this step is the live checkpoint.
        if step is None or step.status != "awaiting_approval":
            return run

        payload = edited_payload if edited_payload is not None else step.proposed_write
        impls = REGISTRY.impls_for(ctx.copilot, [step.tool])
        try:
            out = await asyncio.to_thread(_call_payload, impls[step.tool], payload)
        except Exception as exc:  # noqa: BLE001
            return await self._fail(db, run, step, f"write failed: {type(exc).__name__}: {exc}")
        if _is_error(out):
            return await self._fail(db, run, step, str(out.get("error")), output=out)

        step.output = _jsonsafe(out)
        step.status = "approved"
        step.approved_by = approved_by
        step.approved_at = _now()
        run.current_step_idx = idx + 1
        run.status = "running"
        await db.commit()
        return await self.advance(db, run, ctx)

    # ----------------------------------------------------------------- reject
    async def reject(
        self,
        db: AsyncSession,
        run: AgentRun,
        idx: int,
        ctx: RunContext,
        *,
        note: Optional[str] = None,
    ) -> AgentRun:
        step = await self._step(db, run.id, idx)
        if step is None or step.status != "awaiting_approval":
            return run
        step.status = "skipped"
        step.output = {"rejected": True, "note": note}
        run.current_step_idx = idx + 1
        run.status = "running"
        await db.commit()
        return await self.advance(db, run, ctx)

    # ------------------------------------------------------------------ abort
    async def abort(self, db: AsyncSession, run: AgentRun) -> AgentRun:
        if run.status not in ("done", "error"):
            run.status = "aborted"
            await db.commit()
        return run

    # ----------------------------------------------------------------- helpers
    async def _execute(self, step: AgentStep, ctx: RunContext, run: AgentRun) -> Any:
        if step.type in ("read", "write"):
            impls = REGISTRY.impls_for(ctx.copilot, [step.tool])
            fn = impls[step.tool]
            if step.type == "read":
                return await asyncio.to_thread(_call_kwargs, fn, step.input)
            return await asyncio.to_thread(_call_payload, fn, step.proposed_write or step.input)

        # compute / analysis -> the definition's logic
        definition = _definition_for(run.agent_type)
        ctx.usage_out.clear()
        out = await asyncio.to_thread(definition.execute_step, _planned_from(step), ctx)
        if ctx.usage_out:
            run.credits_used += _credits_from_usage(ctx.usage_out)
            _accumulate_usage(run, ctx.usage_out)
        return out

    async def _fail(
        self,
        db: AsyncSession,
        run: AgentRun,
        step: AgentStep,
        message: str,
        *,
        output: Optional[dict] = None,
    ) -> AgentRun:
        step.status = "error"
        step.output = output if output is not None else {"error": message}
        run.status = "error"
        run.error_message = message
        await db.commit()
        return run

    async def _load_steps(self, db: AsyncSession, run_id) -> List[AgentStep]:
        res = await db.execute(
            select(AgentStep).where(AgentStep.run_id == run_id).order_by(AgentStep.idx)
        )
        return list(res.scalars().all())

    async def _step(self, db: AsyncSession, run_id, idx: int) -> Optional[AgentStep]:
        res = await db.execute(
            select(AgentStep).where(AgentStep.run_id == run_id, AgentStep.idx == idx)
        )
        return res.scalar_one_or_none()

    @staticmethod
    def _rebuild_results(steps: List[AgentStep]) -> List[StepResult]:
        out: List[StepResult] = []
        for s in steps:
            if s.status in ("done", "approved") and s.output is not None:
                out.append(
                    StepResult(s.idx, s.title or "", s.type, s.tool, dict(s.input or {}), s.output)
                )
        return out


def _definition_for(agent_type: str) -> AgentDefinition:
    # Imported lazily so importing the runtime never pulls every definition.
    from agent.types import AGENT_DEFINITIONS

    definition = AGENT_DEFINITIONS.get(agent_type)
    if definition is None:
        raise KeyError(f"no agent definition registered for '{agent_type}'")
    return definition
