"""Offline tests for the support-desk assist (POST /support/assist).

Same harness as test_agent_router.py: the DB dependency is stubbed, and the
retrieval + LLM seams are monkeypatched, so nothing here needs Postgres,
Bedrock or the embedding model. The pure helpers in prompts/support.py are
tested directly.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

import config
import main
from database import get_db
from prompts import support as support_prompt


async def _fake_db():
    yield None


main.app.dependency_overrides[get_db] = _fake_db
client = TestClient(main.app)

_SECRET = "test-support-secret"
_HEADERS = {"X-Support-Secret": _SECRET}

_WRITE_BODY = {
    "mode": "write",
    "language": "en",
    "draft": "pls check ur smtp setting in company profile > email. we fixed it yesterday",
    "ticket": {
        "number": "TKT-20260924-AB12CD",
        "subject": "Emails are not sending",
        "product": "1audit",
        "customer_name": "Sara",
        "agent_name": "Ali",
    },
    "conversation": [
        {"role": "customer", "name": "Sara", "text": "My invoices are not being emailed."},
    ],
    "caller_user_id": 7,
}


def _lines(response):
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def _stub_llm(monkeypatch, pieces, *, usage=(10, 5)):
    """Replace the Bedrock stream with canned pieces and silence the ledger."""
    import routers.support as support_router

    async def _astream(system, prompt, **kwargs):
        usage_out = kwargs.get("usage_out")
        if usage_out is not None:
            usage_out.update(input=usage[0], output=usage[1], model="stub")
        for piece in pieces:
            yield piece

    recorded = {}

    async def _record(db, **kwargs):
        recorded.update(kwargs)

    monkeypatch.setattr(support_router.structured, "astream_text", _astream)
    monkeypatch.setattr(support_router.usage_meter, "record_usage", _record)
    return recorded


def _stub_retrieval(monkeypatch, chunks):
    import routers.support as support_router

    calls = {"count": 0}

    async def _embed(question):
        return [0.0, 0.0, 0.0]

    async def _retrieve(db, question, embedding, top_k, language="en"):
        calls["count"] += 1
        return list(chunks)

    monkeypatch.setattr(support_router, "embed_query", _embed)
    monkeypatch.setattr(support_router.qa, "retrieve_chunks", _retrieve)
    monkeypatch.setattr(support_router.qa, "HELP_CENTER_URL_TEMPLATE", "https://1audit.com/{lang}/helps/{slug}")
    return calls


def _chunk(**overrides):
    base = dict(
        chunk_id=1,
        document_id=1,
        document_filename="help-manual.docx",
        content="Open Clients in the side menu and choose Add client.",
        similarity=0.9,
        fts_rank=0.0,
        rerank_score=1.2,
        page_number=None,
        section_heading="Clients",
        chunk_type="text",
        help_url="clients",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------
def test_503_when_secret_unset(monkeypatch):
    monkeypatch.setattr(config, "SUPPORT_SHARED_SECRET", "")
    r = client.post("/support/assist", json=_WRITE_BODY, headers=_HEADERS)
    assert r.status_code == 503


def test_403_wrong_secret(monkeypatch):
    monkeypatch.setattr(config, "SUPPORT_SHARED_SECRET", _SECRET)
    r = client.post("/support/assist", json=_WRITE_BODY, headers={"X-Support-Secret": "nope"})
    assert r.status_code == 403


def test_422_on_missing_inputs(monkeypatch):
    monkeypatch.setattr(config, "SUPPORT_SHARED_SECRET", _SECRET)
    _stub_llm(monkeypatch, ["x"])
    _stub_retrieval(monkeypatch, [])
    ticket = _WRITE_BODY["ticket"]
    assert client.post("/support/assist", json={"mode": "ask", "ticket": ticket}, headers=_HEADERS).status_code == 422
    assert client.post("/support/assist", json={"mode": "write", "ticket": ticket}, headers=_HEADERS).status_code == 422
    assert client.post(
        "/support/assist", json={"mode": "write", "translate": True, "ticket": ticket}, headers=_HEADERS
    ).status_code == 422


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def test_ask_streams_sources_and_plain_text(monkeypatch):
    monkeypatch.setattr(config, "SUPPORT_SHARED_SECRET", _SECRET)
    # A standards chunk (no help_url) ranked first must not displace the manual.
    calls = _stub_retrieval(
        monkeypatch,
        [_chunk(help_url=None, document_filename="ISA-315.pdf", section_heading=None), _chunk()],
    )
    recorded = _stub_llm(
        monkeypatch,
        ["To create a client, open **Clients** in the side menu", " and choose Add client.\n\n",
         "Learn more: https://1audit.com/en/helps/clients"],
    )
    body = {
        "mode": "ask",
        "language": "en",
        "question": "How do I create a client?",
        "ticket": _WRITE_BODY["ticket"],
        "caller_user_id": "7",
    }
    r = client.post("/support/assist", json=body, headers=_HEADERS)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")
    lines = _lines(r)
    assert [l["type"] for l in lines] == ["meta", "delta", "delta", "delta", "done"]
    assert calls["count"] == 1
    assert lines[0]["mode"] == "ask"
    assert lines[0]["sources"] == [{"title": "Clients", "url": "https://1audit.com/en/helps/clients"}]
    done = lines[-1]
    assert "**" not in done["text"]
    assert done["text"].startswith("To create a client, open Clients")
    assert done["text"].endswith("Learn more: https://1audit.com/en/helps/clients")
    assert done["was_answered"] is True
    assert done["usage"] == {"input": 10, "output": 5}
    assert recorded["organization_id"] == config.SUPPORT_ASSIST_ORG_ID
    assert recorded["user_id"] == "support:7"
    assert recorded["feature"] == "support_ask"


def test_ask_no_answer_is_flagged(monkeypatch):
    monkeypatch.setattr(config, "SUPPORT_SHARED_SECRET", _SECRET)
    _stub_retrieval(monkeypatch, [])
    _stub_llm(monkeypatch, [support_prompt.NO_ANSWER_PREFIXES["ar"] + " يمكن التحقق من ذلك مع فريق المنتج."])
    body = {"mode": "ask", "language": "ar", "question": "كيف أغيّر لون الواجهة؟", "ticket": {}}
    lines = _lines(client.post("/support/assist", json=body, headers=_HEADERS))
    assert lines[-1]["was_answered"] is False
    assert lines[-1]["text"].startswith("لم أجد هذا")


def test_write_polish_skips_retrieval(monkeypatch):
    monkeypatch.setattr(config, "SUPPORT_SHARED_SECRET", _SECRET)
    calls = _stub_retrieval(monkeypatch, [_chunk()])
    recorded = _stub_llm(monkeypatch, ["Hi Sara,\n\nPlease check the SMTP settings under Company profile > Email.\n\nBest regards,\nAli"])
    lines = _lines(client.post("/support/assist", json=_WRITE_BODY, headers=_HEADERS))
    assert calls["count"] == 0  # a draft with no instruction is a polish: no manual lookup
    assert lines[0]["sources"] == []
    assert lines[-1]["text"].startswith("Hi Sara,")
    assert recorded["feature"] == "support_write"


def test_write_from_scratch_retrieves(monkeypatch):
    monkeypatch.setattr(config, "SUPPORT_SHARED_SECRET", _SECRET)
    calls = _stub_retrieval(monkeypatch, [_chunk()])
    _stub_llm(monkeypatch, ["Hi Sara,\n\nHere is how to add a client.\n\nBest regards,\nAli"])
    body = dict(_WRITE_BODY, draft="", instruction="explain how to add a client")
    lines = _lines(client.post("/support/assist", json=body, headers=_HEADERS))
    assert calls["count"] == 1
    assert lines[-1]["type"] == "done"


def test_translate_skips_retrieval_and_records_feature(monkeypatch):
    monkeypatch.setattr(config, "SUPPORT_SHARED_SECRET", _SECRET)
    calls = _stub_retrieval(monkeypatch, [_chunk()])
    recorded = _stub_llm(monkeypatch, ["مرحباً سارة،\n\nيرجى التحقق من إعدادات SMTP.\n\nمع تحياتنا،\nعلي"])
    body = dict(_WRITE_BODY, language="ar", translate=True)
    lines = _lines(client.post("/support/assist", json=body, headers=_HEADERS))
    assert calls["count"] == 0
    assert lines[0]["translate"] is True
    assert lines[-1]["text"] == "مرحباً سارة،\n\nيرجى التحقق من إعدادات SMTP.\n\nمع تحياتنا،\nعلي"
    assert recorded["feature"] == "support_translate"


def test_llm_failure_emits_error_line(monkeypatch):
    monkeypatch.setattr(config, "SUPPORT_SHARED_SECRET", _SECRET)
    _stub_retrieval(monkeypatch, [])
    import routers.support as support_router

    async def _boom(system, prompt, **kwargs):
        raise RuntimeError("bedrock down")
        yield  # pragma: no cover — makes this an async generator

    async def _record(db, **kwargs):
        return None

    monkeypatch.setattr(support_router.structured, "astream_text", _boom)
    monkeypatch.setattr(support_router.usage_meter, "record_usage", _record)
    lines = _lines(client.post("/support/assist", json=_WRITE_BODY, headers=_HEADERS))
    assert [l["type"] for l in lines] == ["meta", "error"]
    assert "retry" in lines[-1]["message"].lower()


def _break_retrieval(monkeypatch):
    """The lookup itself fails (unresolvable database host, embedding error)."""
    import routers.support as support_router

    async def _embed_boom(question):
        raise OSError("nodename nor servname provided, or not known")

    monkeypatch.setattr(support_router, "embed_query", _embed_boom)


def test_ask_fails_fast_when_manual_unavailable(monkeypatch):
    monkeypatch.setattr(config, "SUPPORT_SHARED_SECRET", _SECRET)
    _break_retrieval(monkeypatch)
    import routers.support as support_router

    llm_calls = {"count": 0}

    async def _astream(system, prompt, **kwargs):
        llm_calls["count"] += 1
        yield "must not be called"

    async def _record(db, **kwargs):
        return None

    monkeypatch.setattr(support_router.structured, "astream_text", _astream)
    monkeypatch.setattr(support_router.usage_meter, "record_usage", _record)
    body = {"mode": "ask", "language": "en", "question": "How do I create a client?", "ticket": {}}
    lines = _lines(client.post("/support/assist", json=body, headers=_HEADERS))
    assert [l["type"] for l in lines] == ["meta", "error"]
    assert lines[0]["retrieval"] == "unavailable"
    assert "help manual" in lines[-1]["message"].lower()
    assert llm_calls["count"] == 0  # no "I couldn't find this" masquerading as an answer


def test_write_from_scratch_continues_when_manual_unavailable(monkeypatch):
    monkeypatch.setattr(config, "SUPPORT_SHARED_SECRET", _SECRET)
    _break_retrieval(monkeypatch)
    _stub_llm(monkeypatch, ["Hi Sara,\n\nWe are looking into it and will update you today.\n\nBest regards,\nAli"])
    body = {"mode": "write", "language": "en", "instruction": "tell her we are looking into it", "ticket": {}}
    lines = _lines(client.post("/support/assist", json=body, headers=_HEADERS))
    assert lines[0]["retrieval"] == "unavailable"
    assert lines[-1]["type"] == "done"
    assert lines[-1]["text"].startswith("Hi Sara,")


def test_reply_mode_grounds_on_last_customer_message(monkeypatch):
    """The one-click auto reply: no draft, no instruction from the agent; the
    lookup is keyed by what the customer last said and the model gets the
    built-in reply instruction."""
    monkeypatch.setattr(config, "SUPPORT_SHARED_SECRET", _SECRET)
    import routers.support as support_router

    seen = {}

    async def _embed(question):
        seen["query"] = question
        return [0.0, 0.0, 0.0]

    async def _retrieve(db, question, embedding, top_k, language="en"):
        return [_chunk()]

    prompts = {}

    async def _astream(system, prompt, **kwargs):
        prompts["system"], prompts["user"] = system, prompt
        yield "Hi Sara,\n\nOpen Clients in the side menu and choose Add client.\n\nBest regards,\nAli"

    recorded = {}

    async def _record(db, **kwargs):
        recorded.update(kwargs)

    monkeypatch.setattr(support_router, "embed_query", _embed)
    monkeypatch.setattr(support_router.qa, "retrieve_chunks", _retrieve)
    monkeypatch.setattr(support_router.structured, "astream_text", _astream)
    monkeypatch.setattr(support_router.usage_meter, "record_usage", _record)
    body = {
        "mode": "reply",
        "language": "en",
        "ticket": {"subject": "Clients", "customer_name": "Sara", "agent_name": "Ali"},
        "conversation": [
            {"role": "customer", "name": "Sara", "text": "How do I create a client?"},
            {"role": "agent", "name": "Ali", "text": "Let me check."},
            {"role": "customer", "name": "Sara", "text": "Any update on creating a client?"},
        ],
    }
    lines = _lines(client.post("/support/assist", json=body, headers=_HEADERS))
    assert lines[0]["mode"] == "reply" and lines[0]["retrieval"] == "ok"
    assert "creating a client" in seen["query"].lower()          # newest customer turn
    assert support_prompt.AUTO_REPLY_INSTRUCTION[:40] in prompts["user"]
    assert lines[-1]["type"] == "done" and lines[-1]["text"].startswith("Hi Sara,")
    assert recorded["feature"] == "support_reply"


def test_help_manual_false_skips_retrieval(monkeypatch):
    """Other products' tickets never get grounded in the 1audit manual."""
    monkeypatch.setattr(config, "SUPPORT_SHARED_SECRET", _SECRET)
    calls = _stub_retrieval(monkeypatch, [_chunk()])
    _stub_llm(monkeypatch, ["Hi Sara,\n\nThanks for the details.\n\nBest regards,\nAli"])
    body = {"mode": "reply", "language": "en", "help_manual": False, "ticket": {},
            "conversation": [{"role": "customer", "name": "Sara", "text": "How do I add a client?"}]}
    lines = _lines(client.post("/support/assist", json=body, headers=_HEADERS))
    assert calls["count"] == 0
    assert lines[0]["retrieval"] == "skipped"
    assert lines[-1]["type"] == "done"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def test_to_plain_text_strips_markdown_and_html():
    raw = (
        "## From your knowledge base\n"
        "Here is **how** to _do_ it:\n\n"
        "* First step\n"
        "+ Second `step`\n"
        "1. Numbered\n\n"
        "| Col A | Col B |\n|---|---|\n| one | two |\n\n"
        "<p>Para <strong>html</strong></p><ul><li>item</li></ul>\n"
        "See [the guide](https://1audit.com/en/helps/x).\n"
        "```\ncode\n```\n"
        "## Follow-up Questions\n- Q1?\n- Q2?\n"
    )
    out = support_prompt.to_plain_text(raw)
    assert out == (
        "Here is how to do it:\n\n"
        "- First step\n"
        "- Second step\n"
        "1. Numbered\n\n"
        "Col A – Col B\n"
        "one – two\n\n"
        "Para html\n"
        "item\n\n"
        "See the guide: https://1audit.com/en/helps/x.\n"
        "code"
    )


def test_to_plain_text_keeps_arabic_byte_identical():
    arabic = (
        "مرحباً أحمد،\n\n"
        "شكراً لتواصلك معنا. هل جرّبت إعادة تسجيل الدخول؟\n\n"
        "- الخطوة الأولى: افتح «الإعدادات»؛ ثم اختر البريد الإلكتروني.\n"
        "1. أعد المحاولة.\n\n"
        "مع تحياتنا،\n"
        "علي"
    )
    assert support_prompt.to_plain_text(arabic) == arabic


def test_guard_links_keeps_allowed_hosts_only():
    text = (
        "Learn more: https://1audit.com/en/helps/clients\n"
        "Also https://dev.1audit.com/ar/app/x?y=1.\n"
        "Never http://evil.example/phish, ok?"
    )
    out = support_prompt.guard_links(text)
    assert "https://1audit.com/en/helps/clients" in out
    assert "https://dev.1audit.com/ar/app/x?y=1." in out
    assert "evil.example" not in out
    assert "[link removed], ok?" in out


def test_was_answered_recognises_both_languages():
    assert support_prompt.was_answered("Open Clients and choose Add client.", "en") is True
    assert support_prompt.was_answered(support_prompt.NO_ANSWER_PREFIXES["en"] + " Ask the product team.", "en") is False
    assert support_prompt.was_answered(support_prompt.NO_ANSWER_PREFIXES["ar"], "ar") is False


def test_conversation_block_keeps_newest_within_budget():
    old = {"role": "customer", "name": "Sara", "text": "old " * 400}
    newer = {"role": "agent", "name": "Ali", "text": "new " * 400}
    newest = {"role": "customer", "name": "Sara", "text": "latest message"}
    prompt = support_prompt.build_write_prompt(
        draft="",
        instruction="reply",
        translate=False,
        excerpts=[],
        kb_articles=[],
        ticket={"subject": "s"},
        conversation=[old, newer, newest] * 4,
        language="en",
    )
    assert "latest message" in prompt
    assert prompt.index("<draft>") > prompt.index("<conversation>")
    assert len(prompt) <= 14000
