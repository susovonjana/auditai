"""Router-level tests for the agent feature gate (offline — DB dependency stubbed).

The runtime itself is covered by test_agent_runtime.py; here we only prove the
HTTP gate: the /agent/run endpoint refuses callers unless the feature flag is on
AND the org is allowlisted. The gate fires before any DB use, so a stub session
is enough.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

import config
import main
from database import get_db


async def _fake_db():
    yield None


main.app.dependency_overrides[get_db] = _fake_db
client = TestClient(main.app)

_BODY = {
    "session_token": "s",
    "audit_file_id": 1,
    "copilot_grant": "g",
    "agent_type": "file_review",
    "organization_id": "11",
}


def test_run_forbidden_when_feature_off(monkeypatch):
    monkeypatch.setattr(config, "AGENT_FEATURE_ENABLED", False)
    r = client.post("/agent/run", json=_BODY)
    assert r.status_code == 403


def test_run_forbidden_when_org_not_allowlisted(monkeypatch):
    monkeypatch.setattr(config, "AGENT_FEATURE_ENABLED", True)
    monkeypatch.setattr(config, "AGENT_ALLOWED_ORG_IDS", {"99"})
    r = client.post("/agent/run", json=_BODY)
    assert r.status_code == 403


def test_run_unknown_agent_type_404(monkeypatch):
    # feature on + org allowed, but session check must pass first; stub it + grant.
    monkeypatch.setattr(config, "AGENT_FEATURE_ENABLED", True)
    monkeypatch.setattr(config, "AGENT_ALLOWED_ORG_IDS", {"11"})

    import routers.agent as agent_router
    import usage_meter
    from copilot_tools import CopilotContext

    async def _ok_session(db, token):
        return object()

    async def _ok_credits(db, org):
        return None

    monkeypatch.setattr(agent_router, "_load_session", _ok_session)
    monkeypatch.setattr(usage_meter, "ensure_credits", _ok_credits)
    monkeypatch.setattr(CopilotContext, "validate_grant_local", lambda self: None)

    body = dict(_BODY, agent_type="does_not_exist")
    r = client.post("/agent/run", json=body)
    assert r.status_code == 404


def test_agent_run_request_accepts_is_template():
    """The FE flag rides the start payload and defaults off."""
    import schemas

    req = schemas.AgentRunRequest(**dict(_BODY, working_paper_id=7, is_template=True))
    assert req.is_template is True
    assert schemas.AgentRunRequest(**_BODY).is_template is False


def test_history_row_undo_redo_derivation():
    """can_undo: finished + sections still in the WP; can_redo: undone with the
    exact deleted ids recorded. Pure serializer, no DB."""
    from types import SimpleNamespace

    from routers.agent import _history_row

    def run(**over):
        base = dict(
            id="4b6a0d3e-0000-0000-0000-000000000001", status="done",
            agent_type="procedure_buildout", working_paper_id=10197,
            is_template=False, goal="g", created_at=None, created_by="27",
            result_summary={"sections_created": 10, "summary": "Created 10"},
        )
        base.update(over)
        return SimpleNamespace(**base)

    fresh = _history_row(run())
    # live in the paper: can undo, cannot delete from history (undo first)
    assert fresh["can_undo"] is True and fresh["can_redo"] is False
    assert fresh["can_delete"] is False

    undone = _history_row(run(result_summary={
        "sections_created": 10, "undone": True, "undone_count": 10,
        "undone_section_ids": [1, 2, 3],
    }))
    assert undone["can_undo"] is False and undone["can_redo"] is True
    assert undone["can_delete"] is True  # nothing live → removable

    legacy_undone = _history_row(run(result_summary={
        "sections_created": 10, "undone": True, "undone_count": 10,
    }))
    assert legacy_undone["can_redo"] is False  # no ids recorded pre-round-7
    assert legacy_undone["can_delete"] is True

    unwritten = _history_row(run(status="done", result_summary={"sections_created": 0}))
    assert unwritten["can_undo"] is False
    assert unwritten["can_delete"] is True  # awaiting/never-wrote → removable

    errored = _history_row(run(status="error", result_summary=None))
    assert errored["can_undo"] is False and errored["sections_created"] == 0
    assert errored["can_delete"] is True


def test_spawn_proc_memory_ingest_schedules_background_task(monkeypatch):
    """Approving a procedure run must schedule the learn-from-approved task
    without ever blocking or failing the approve path."""
    import asyncio

    import routers.agent as agent_router

    called = {}

    async def fake_ingest(run_id, user_id):
        called["args"] = (run_id, user_id)

    monkeypatch.setattr(agent_router, "_ingest_approved_program", fake_ingest)

    async def go():
        agent_router._spawn_proc_memory_ingest("run-1", "user-9")
        await asyncio.sleep(0)  # yield so the task body runs

    asyncio.run(go())
    assert called["args"] == ("run-1", "user-9")
