"""Unit tests for the agent runtime (offline — no DB, no HTTP, no Bedrock).

A tiny fake AsyncSession holds AgentRun/AgentStep objects in memory; the runtime
mutates them in place and "commits" are no-ops. Tool callables and the agent
definition are fakes registered under test-only names, so nothing real is hit.
"""
from __future__ import annotations

import uuid

import pytest

import config
from agent.models import AgentRun, AgentStep
from agent.registry import REGISTRY, RegisteredTool, ToolKind
from agent.runtime import AgentRuntime
from agent.types import PlannedStep, RunContext, register_definition
from copilot_tools import CopilotContext


# --------------------------------------------------------------------------
# Fake persistence: just enough of AsyncSession for the runtime.
# --------------------------------------------------------------------------
def _apply_defaults(obj) -> None:
    for col in obj.__table__.columns:
        if getattr(obj, col.name, None) is None and col.default is not None:
            arg = col.default.arg
            if callable(arg):
                # SQLAlchemy wraps callable defaults to accept a context arg.
                try:
                    value = arg(None)
                except TypeError:
                    value = arg()
            else:
                value = arg
            setattr(obj, col.name, value)


class FakeSession:
    def __init__(self) -> None:
        self.runs: list = []
        self.steps: list = []

    def add(self, obj) -> None:
        _apply_defaults(obj)
        if isinstance(obj, AgentRun):
            self.runs.append(obj)
        elif isinstance(obj, AgentStep):
            self.steps.append(obj)

    async def flush(self) -> None:
        for o in list(self.runs) + list(self.steps):
            _apply_defaults(o)

    async def commit(self) -> None:
        pass


class _Runtime(AgentRuntime):
    """Runtime with the two DB reads pointed at the in-memory FakeSession."""

    async def _load_steps(self, db, run_id):
        return sorted([s for s in db.steps if s.run_id == run_id], key=lambda s: s.idx)

    async def _step(self, db, run_id, idx):
        for s in db.steps:
            if s.run_id == run_id and s.idx == idx:
                return s
        return None


# --------------------------------------------------------------------------
# Fake tools + definition, registered under test-only names.
# --------------------------------------------------------------------------
WRITE_CALLS: list = []


def _b_read_ok(ctx):
    def f(**_):
        return {"read_ok": True}
    return f


def _b_read_err(ctx):
    def f(**_):
        return {"error": "boom"}
    return f


def _b_write(ctx):
    def f(payload):
        WRITE_CALLS.append(payload)
        return {"written": payload}
    return f


def _b_write_err(ctx):
    def f(payload):
        WRITE_CALLS.append(payload)
        return {"error": "write boom"}
    return f


REGISTRY.register(RegisteredTool("t_read_ok", ToolKind.READ, False, _b_read_ok))
REGISTRY.register(RegisteredTool("t_read_err", ToolKind.READ, False, _b_read_err))
REGISTRY.register(RegisteredTool("t_write", ToolKind.WRITE, True, _b_write))
REGISTRY.register(RegisteredTool("t_write_err", ToolKind.WRITE, True, _b_write_err))


class _FakeAgent:
    agent_type = "test_runtime_agent"
    allowed_tools = ["t_read_ok", "t_write"]

    def __init__(self) -> None:
        self._plan: list = []

    def default_goal(self, audit_file_id: int) -> str:
        return "test goal"

    def build_plan(self, ctx: RunContext):
        return list(self._plan)

    def execute_step(self, step: PlannedStep, ctx: RunContext):
        if step.tool == "bump_credits":
            ctx.usage_out.update(input=1_000_000, output=0, model="fake")
            return {"bumped": True}
        if step.tool == "echo":
            return {"echo": dict(step.args)}
        raise ValueError(f"unknown compute tool {step.tool}")

    def prepare_write(self, step: PlannedStep, ctx: RunContext):
        return {"proposed": True, "tool": step.tool}

    def synthesize(self, ctx: RunContext):
        return {"summary": "ok", "n_steps": len(ctx.results)}


FAKE = _FakeAgent()
register_definition(FAKE)


# --------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _reset():
    WRITE_CALLS.clear()
    FAKE._plan = []
    yield


def _ctx(file_id: int = 99) -> RunContext:
    return RunContext(copilot=CopilotContext(file_id, "grant"), audit_file_id=file_id)


async def _start(db, plan):
    FAKE._plan = plan
    rt = _Runtime()
    run = await rt.start(
        db,
        definition=FAKE,
        audit_file_id=99,
        grant="grant",
        organization_id="11",
        user_id="u1",
        goal=None,
    )
    return rt, run


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
async def test_planning_persists_steps_and_runs_reads():
    db = FakeSession()
    plan = [
        PlannedStep("read a", "read", "t_read_ok"),
        PlannedStep("read b", "read", "t_read_ok"),
        PlannedStep("write c", "write", "t_write", requires_approval=True),
    ]
    rt, run = await _start(db, plan)

    steps = await rt._load_steps(db, run.id)
    assert [s.idx for s in steps] == [0, 1, 2]
    assert [s.type for s in steps] == ["read", "read", "write"]
    # reads executed and stored output
    assert steps[0].status == "done" and steps[0].output == {"read_ok": True}
    assert steps[1].status == "done"
    # paused at the write checkpoint with a populated preview
    assert run.status == "awaiting_approval"
    assert steps[2].status == "awaiting_approval"
    assert steps[2].proposed_write == {"proposed": True, "tool": "t_write"}
    assert WRITE_CALLS == []  # nothing written yet


async def test_approve_runs_write_once_and_finishes():
    db = FakeSession()
    plan = [
        PlannedStep("read a", "read", "t_read_ok"),
        PlannedStep("write b", "write", "t_write", requires_approval=True),
    ]
    rt, run = await _start(db, plan)
    assert run.status == "awaiting_approval"

    run = await rt.approve(db, run, 1, _ctx(), approved_by="u1")
    assert WRITE_CALLS == [{"proposed": True, "tool": "t_write"}]
    steps = await rt._load_steps(db, run.id)
    assert steps[1].status == "approved"
    assert steps[1].approved_by == "u1" and steps[1].approved_at is not None
    assert run.status == "done"
    assert run.result_summary == {"summary": "ok", "n_steps": 2}


async def test_approve_with_edited_payload_uses_edit():
    db = FakeSession()
    plan = [PlannedStep("write b", "write", "t_write", requires_approval=True)]
    rt, run = await _start(db, plan)

    run = await rt.approve(db, run, 0, _ctx(), edited_payload={"edited": 1})
    assert WRITE_CALLS == [{"edited": 1}]  # edited, NOT the proposed preview
    assert run.status == "done"


async def test_repeated_approve_is_noop():
    db = FakeSession()
    plan = [PlannedStep("write b", "write", "t_write", requires_approval=True)]
    rt, run = await _start(db, plan)

    await rt.approve(db, run, 0, _ctx())
    await rt.approve(db, run, 0, _ctx())  # second approve must not write again
    assert len(WRITE_CALLS) == 1


async def test_reject_skips_and_continues():
    db = FakeSession()
    plan = [PlannedStep("write b", "write", "t_write", requires_approval=True)]
    rt, run = await _start(db, plan)

    run = await rt.reject(db, run, 0, _ctx(), note="not now")
    steps = await rt._load_steps(db, run.id)
    assert steps[0].status == "skipped"
    assert steps[0].output == {"rejected": True, "note": "not now"}
    assert WRITE_CALLS == []
    assert run.status == "done"  # nothing left after the skipped step


async def test_step_limit_refused(monkeypatch):
    monkeypatch.setattr(config, "AGENT_STEP_LIMIT", 2)
    db = FakeSession()
    plan = [
        PlannedStep("r1", "read", "t_read_ok"),
        PlannedStep("r2", "read", "t_read_ok"),
        PlannedStep("r3", "read", "t_read_ok"),
    ]
    rt, run = await _start(db, plan)
    assert run.status == "error"
    assert "AGENT_STEP_LIMIT" in (run.error_message or "")
    assert db.steps == []  # no steps persisted for a refused plan


async def test_credit_ceiling_stops_run(monkeypatch):
    monkeypatch.setattr(config, "AGENT_MAX_RUN_CREDITS", 1)
    db = FakeSession()
    plan = [
        PlannedStep("a0", "analysis", "bump_credits"),
        PlannedStep("a1", "analysis", "bump_credits"),
    ]
    rt, run = await _start(db, plan)
    steps = await rt._load_steps(db, run.id)
    assert steps[0].status == "done"          # first analysis ran
    assert run.credits_used >= 1
    assert run.status == "error"              # ceiling hit before the second
    assert steps[1].status == "error"
    assert "ceiling" in (run.error_message or "")


async def test_read_error_is_recorded_but_run_continues():
    # A failed READ is a data gap, not a hard stop — the run continues to the
    # next checkpoint and records the error as that step's output.
    db = FakeSession()
    plan = [
        PlannedStep("bad read", "read", "t_read_err"),
        PlannedStep("write b", "write", "t_write", requires_approval=True),
    ]
    rt, run = await _start(db, plan)
    steps = await rt._load_steps(db, run.id)
    assert steps[0].status == "done"
    assert steps[0].output == {"error": "boom"}      # recorded for synthesize to report
    assert run.status == "awaiting_approval"          # continued to the write checkpoint
    assert steps[1].status == "awaiting_approval"
    assert WRITE_CALLS == []


async def test_write_error_on_approve_fails_run():
    # A failed WRITE IS fatal — the run stops as error (never a half-written file).
    db = FakeSession()
    plan = [PlannedStep("write b", "write", "t_write_err", requires_approval=True)]
    rt, run = await _start(db, plan)
    assert run.status == "awaiting_approval"
    run = await rt.approve(db, run, 0, _ctx())
    assert run.status == "error"
    assert "write boom" in (run.error_message or "")
