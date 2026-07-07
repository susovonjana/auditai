"""Unit tests for the Document Intelligence deterministic logic (offline — no Bedrock, no I/O).

The LLM classification and the real download/parse are not exercised; we test the
extension derivation and ``_extract_documents`` (which walks list_documents, fetches
each doc, and tallies text/links/gaps) with ``_fetch_and_parse`` monkeypatched.
"""
from __future__ import annotations

import agent.definitions.document_intelligence as di


class _Ctx:
    """Minimal RunContext stand-in: only .find('list_documents') is used here."""
    def __init__(self, listing):
        self._listing = listing

    def find(self, tool, **_kw):
        return self._listing if tool == "list_documents" else None


def test_ext_for():
    assert di._ext_for("invoice.PDF", "application/pdf") == ".pdf"
    assert di._ext_for("scan", "image/png") == ".png"
    assert di._ext_for("", "image/jpeg") == ".jpg"
    assert di._ext_for("noext", "application/octet-stream") == ".bin"


def test_extract_documents_builds_entries(monkeypatch):
    listing = {"documents": [
        {"reference": "A-1", "name": "inv.pdf", "mime_type": "application/pdf", "working_papers": ["WP1"]},
        {"reference": "A-2", "name": "scan.png", "mime_type": "image/png", "working_papers": []},
        {"name": "noref.pdf", "mime_type": "application/pdf", "working_papers": []},  # no reference -> gap
    ]}

    def fake_fetch(_ctx, ref, _name, _mime):
        if ref == "A-1":
            return {"text": "Invoice total 1000", "chars": 18, "error": None}
        return {"text": "", "error": "parse failed"}

    monkeypatch.setattr(di, "_fetch_and_parse", fake_fetch)
    out = di.DocumentIntelligenceAgent()._extract_documents(_Ctx(listing))

    assert out["total"] == 3
    assert out["considered"] == 2          # A-1 + A-2 (noref skipped before fetch)
    assert out["with_text"] == 1           # only A-1 yielded text
    assert [d["reference"] for d in out["docs"]] == ["A-1", "A-2"]
    assert out["docs"][0]["linked"] is True
    assert out["docs"][1]["linked"] is False
    assert any("noref.pdf" in g for g in out["gaps"])   # missing-reference gap
    assert any("scan.png" in g for g in out["gaps"])    # parse-failed gap


def test_extract_documents_caps(monkeypatch):
    listing = {"documents": [
        {"reference": f"R{i}", "name": f"d{i}.pdf", "mime_type": "application/pdf", "working_papers": []}
        for i in range(10)
    ]}
    monkeypatch.setattr(di, "_fetch_and_parse", lambda *_a: {"text": "x", "chars": 1, "error": None})
    out = di.DocumentIntelligenceAgent()._extract_documents(_Ctx(listing))

    assert out["total"] == 10
    assert out["considered"] == di._MAX_DOCS
    assert any("only the first" in g for g in out["gaps"])


def test_empty():
    out = di.DocumentIntelligenceAgent()._extract_documents(_Ctx({"documents": []}))
    assert out["total"] == 0 and out["docs"] == [] and out["with_text"] == 0
