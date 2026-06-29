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
