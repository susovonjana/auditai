"""Router-level tests for /copilot/document/* (offline — DB stubbed).

Proves the gates fire before any work: feature switch, session, grant, and the
document handle requirement; plus the request schemas' id stringification.
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
    "audit_file_id": 56,
    "copilot_grant": "g",
    "document_id": 7,
    "organization_id": 11,
    "user_id": 3,
}


def test_insight_forbidden_when_feature_off(monkeypatch):
    monkeypatch.setattr(config, "DOC_READER_ENABLED", False)
    r = client.post("/copilot/document/insight", json=_BODY)
    assert r.status_code == 403


def test_insight_requires_document_handle(monkeypatch):
    monkeypatch.setattr(config, "DOC_READER_ENABLED", True)
    import routers.document as doc_router

    async def _ok_session(db, token):
        return object()

    monkeypatch.setattr(doc_router, "_load_session", _ok_session)
    body = dict(_BODY)
    body.pop("document_id")
    r = client.post("/copilot/document/insight", json=body)
    assert r.status_code == 422


def test_insight_rejects_bad_grant(monkeypatch):
    monkeypatch.setattr(config, "DOC_READER_ENABLED", True)
    import routers.document as doc_router

    async def _ok_session(db, token):
        return object()

    monkeypatch.setattr(doc_router, "_load_session", _ok_session)
    # "g" is not a JWT → validate_grant_local raises → 401
    r = client.post("/copilot/document/insight", json=_BODY)
    assert r.status_code == 401


def _grant(feature=None, sub_feature=None):
    """Unsigned grant with the claims 1audit-be sets (verify_signature is off when
    COPILOT_GRANT_SECRET is empty, so the local check reads the claims only)."""
    import time
    import jwt
    return jwt.encode(
        {"scope": "copilot", "audit_file_id": 56, "user_id": 3, "organization_id": 11,
         "feature": feature, "sub_feature": sub_feature, "exp": int(time.time()) + 600},
        "x", algorithm="HS256",
    )


def _stub_session_and_credits(monkeypatch):
    import routers.document as doc_router
    import usage_meter

    async def _ok_session(db, token):
        return object()

    async def _ok_credits(db, org):
        return None

    monkeypatch.setattr(doc_router, "_load_session", _ok_session)
    monkeypatch.setattr(usage_meter, "ensure_credits", _ok_credits)
    monkeypatch.setattr(config, "COPILOT_GRANT_SECRET", "")
    import copilot_tools
    monkeypatch.setattr(copilot_tools, "COPILOT_GRANT_SECRET", "")


def test_insight_refuses_grant_of_another_feature(monkeypatch):
    """The prime-admin gate: a grant minted for chat/agent cannot drive the reader."""
    monkeypatch.setattr(config, "DOC_READER_ENABLED", True)
    _stub_session_and_credits(monkeypatch)
    r = client.post("/copilot/document/insight", json={**_BODY, "copilot_grant": _grant("agent")})
    assert r.status_code == 403
    r = client.post("/copilot/document/insights", json={**_BODY, "copilot_grant": _grant("chat")})
    assert r.status_code == 403


def test_ask_requires_ask_document_sub_feature(monkeypatch):
    monkeypatch.setattr(config, "DOC_READER_ENABLED", True)
    _stub_session_and_credits(monkeypatch)
    import routers.document as doc_router

    async def _no_row(db, **kw):
        return None

    monkeypatch.setattr(doc_router.store, "get_insight", _no_row)
    # parent feature only → 403 (the prime admin may have unticked "Ask about a document")
    r = client.post("/copilot/document/ask", json={**_BODY, "copilot_grant": _grant("document_ai"), "question": "q"})
    assert r.status_code == 403
    # parent + sub-feature → passes the gate (409 = no reading yet, i.e. past auth)
    r = client.post("/copilot/document/ask", json={**_BODY, "copilot_grant": _grant("document_ai", "ask_document"), "question": "q"})
    assert r.status_code == 409


def test_ask_requires_prior_reading(monkeypatch):
    monkeypatch.setattr(config, "DOC_READER_ENABLED", True)
    import routers.document as doc_router
    import usage_meter
    from copilot_tools import CopilotContext

    async def _ok_session(db, token):
        return object()

    async def _ok_credits(db, org):
        return None

    async def _no_row(db, **kw):
        return None

    monkeypatch.setattr(doc_router, "_load_session", _ok_session)
    monkeypatch.setattr(usage_meter, "ensure_credits", _ok_credits)
    monkeypatch.setattr(CopilotContext, "validate_grant_local", lambda self: None)
    monkeypatch.setattr(CopilotContext, "require_feature", lambda self, f, sf=None: None)
    monkeypatch.setattr(doc_router.store, "get_insight", _no_row)
    r = client.post("/copilot/document/ask", json={**_BODY, "question": "What is the total?"})
    assert r.status_code == 409


def test_schemas_stringify_ids():
    import schemas

    req = schemas.DocumentInsightRequest(**_BODY)
    assert req.organization_id == "11" and req.user_id == "3"
    assert req.refresh is False and req.cached_only is False
    ask = schemas.DocumentAskRequest(**{**_BODY, "question": "q", "history": [{"role": "user", "content": "hi"}]})
    assert ask.history[0].role == "user"


def test_document_agent_refuses_whole_file_scan(monkeypatch):
    """The agent route 422s without document_ids — reading is selection-only."""
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
    r = client.post("/agent/run", json={"session_token": "s", "audit_file_id": 56, "copilot_grant": "g",
                                        "agent_type": "document_extraction", "organization_id": "11"})
    assert r.status_code == 422
    import schemas
    assert schemas.AgentRunRequest(session_token="s", audit_file_id=1, copilot_grant="g", agent_type="x", document_ids=[1, 2]).document_ids == [1, 2]
