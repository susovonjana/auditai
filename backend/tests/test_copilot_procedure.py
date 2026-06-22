"""Endpoint test for POST /copilot/procedure (offline — DB, embeddings and
Gemini are all stubbed)."""
import json

import pytest
from fastapi.testclient import TestClient

import main
import qa
import structured
import routers.copilot as copilot
from database import get_db


class _FakeSession:
    id = "00000000-0000-0000-0000-000000000000"


@pytest.fixture
def client(monkeypatch):
    async def _fake_db():
        yield None

    main.app.dependency_overrides[get_db] = _fake_db

    async def _fake_load_session(db, token):
        return _FakeSession()

    async def _fake_embed(query):
        return [0.0] * 384

    async def _fake_retrieve(*args, **kwargs):
        return []

    async def _fake_astream(system, prompt, **kwargs):
        for piece in (
            "<p>Perform the following substantive procedures.</p>",
            "<ol><li>Obtain the aged receivables listing.</li>",
            "<li>Select a sample and confirm balances directly.</li></ol>",
        ):
            yield piece

    monkeypatch.setattr(copilot, "_load_session", _fake_load_session)
    monkeypatch.setattr(copilot, "embed_query", _fake_embed)
    monkeypatch.setattr(qa, "retrieve_chunks", _fake_retrieve)
    monkeypatch.setattr(structured, "astream_text", _fake_astream)

    c = TestClient(main.app)
    yield c
    main.app.dependency_overrides.clear()


def test_procedure_stream_yields_html_with_ordered_list(client):
    body = {
        "session_token": "tok",
        "section_title": "Trade receivables — substantive testing",
        "risks": [
            {
                "title": "Overstatement of receivables",
                "description": "Trade receivables may be overstated.",
                "assessment_level": "significant",
            }
        ],
        "assertions": ["Existence", "Valuation"],
        "client_sector": "Retail",
        "audit_area": "Receivables",
        "language": "en",
    }
    resp = client.post("/copilot/procedure", json=body)
    assert resp.status_code == 200

    events = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    types = [e["type"] for e in events]
    assert types[0] == "meta"
    assert types[-1] == "done"

    html = "".join(e["text"] for e in events if e["type"] == "delta")
    assert "<ol>" in html
    assert "<li>" in html


def test_procedure_strips_code_fences(client, monkeypatch):
    async def _fenced_stream(system, prompt, **kwargs):
        yield "```html\n<ol><li>Step one</li></ol>\n```"

    monkeypatch.setattr(structured, "astream_text", _fenced_stream)
    body = {"session_token": "tok", "audit_area": "Cash", "language": "en"}
    resp = client.post("/copilot/procedure", json=body)
    assert resp.status_code == 200
    events = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    html = "".join(e["text"] for e in events if e["type"] == "delta")
    assert "```" not in html
    assert "<ol>" in html
