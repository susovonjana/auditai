"""Unit tests for the document_extraction agent's deterministic logic (offline).

The reader itself (``document_reader.build_insight``) and the insight store are
stubbed; we test the plan, the batch loop's caps/skips/reuse, usage accumulation,
and the report synthesis.
"""
from __future__ import annotations

import config
import agent.definitions.document_extraction as de
from agent.types import AGENT_DEFINITIONS, PlannedStep


class _Copilot:
    pass


class _Ctx:
    def __init__(self, outputs, document_ids=None):
        self._outputs = outputs
        self.copilot = _Copilot()
        self.audit_file_id = 56
        self.language = "en"
        self.organization_id = "11"
        self.usage_out = {}
        self.document_ids = list(document_ids or [])

    def find(self, tool, **_kw):
        return self._outputs.get(tool)


def test_registered_and_plan_requires_selection():
    import pytest
    agent = AGENT_DEFINITIONS["document_extraction"]
    assert agent.requires_document_ids is True
    with pytest.raises(ValueError):
        agent.build_plan(_Ctx({}))  # never a whole-file scan
    plan = agent.build_plan(_Ctx({}, document_ids=[3, 1]))
    assert [p.tool for p in plan] == ["get_audit_file_summary", "list_working_papers", "list_documents", "read_documents"]
    assert plan[-1].args == {"document_ids": [3, 1]}  # selection persisted in the step args
    assert all(not p.requires_approval for p in plan)  # read-only agent
    assert "selected" in agent.default_goal(56)


def test_read_documents_reads_reuses_and_caps(monkeypatch):
    monkeypatch.setattr(config, "DOC_READER_MAX_BATCH_DOCS", 3)
    docs = [{"document_id": i, "reference": f"D-{i}", "name": f"d{i}.pdf", "working_papers": (["WP1"] if i == 1 else [])} for i in range(1, 6)]
    ctx = _Ctx({
        "get_audit_file_summary": {"client": "ACME", "period_start": "2026-01-01", "period_end": "2026-12-31"},
        "list_working_papers": {"working_papers": [{"reference": "C-1", "name": "Revenue"}]},
        "list_documents": {"documents": docs},
    })
    monkeypatch.setattr(de.document_reader, "fetch_document_meta",
                        lambda cp, document_id=None, reference=None: {"document_id": document_id, "reference": reference, "name": f"d{document_id}.pdf", "size": 1, "updated_at": "t"})
    monkeypatch.setattr(de.document_reader, "content_key_for", lambda m: f"key-{m['document_id']}")
    # doc 2 is already read with the same key → reused, no LLM
    monkeypatch.setattr(de.store, "lookup_cached_sync",
                        lambda **kw: ({"insight": {"doc_type": "contract", "summary_short": "cached", "confidence": 0.7}} if kw["document_id"] == 2 else None))
    persisted = []
    monkeypatch.setattr(de.store, "persist_from_thread", lambda **kw: persisted.append(kw) or {})

    def fake_build(cp, *, document_id=None, reference=None, language, file_context, working_papers, usage_out, meta):
        assert file_context["client"] == "ACME" and working_papers[0]["reference"] == "C-1"
        usage_out.update(input=100, output=10, model="m")
        return {"content_key": f"key-{document_id}", "read_method": "native_text", "pages": 1, "extracted_text": "txt",
                "insight": {"doc_type": "invoice", "summary_short": f"inv {document_id}", "confidence": 0.9,
                            "amounts": [{"label": "Total", "value": 1150, "currency": "SAR"}],
                            "audit_relevance": {"suggested_working_papers": ["C-1 · Revenue"]},
                            "red_flags": ["No VAT number"] if document_id == 3 else [],
                            "checks": [{"check": "vat_rate", "status": "warning", "detail": "VAT is 10%"}] if document_id == 3 else []}}

    monkeypatch.setattr(de.document_reader, "build_insight", fake_build)
    # selection: 1,2,3 (+ 9 which is not on the file) and NOT 4,5 → never read
    out = de.DocumentExtractionAgent()._read_documents(ctx, [1, 2, 3, 9])

    assert out["total"] == 3 and out["considered"] == 3
    assert any("not found" in g and "9" in g for g in out["gaps"])
    assert out["read_now"] == 2 and out["reused"] == 1
    assert [d["document_id"] for d in out["documents"]] == [1, 2, 3]
    assert out["documents"][1]["cached"] is True and out["documents"][1]["doc_type"] == "contract"
    assert out["documents"][0]["linked"] is True and out["documents"][2]["linked"] is False
    assert len(persisted) == 2 and persisted[0]["audit_file_id"] == 56 and persisted[0]["language"] == "en"
    assert ctx.usage_out == {"input": 200, "output": 20, "model": "m"}  # summed across the 2 reads
    assert not any("per-run cap" in g for g in out["gaps"])
    # red flags + warning checks merged per document
    assert out["documents"][2]["red_flags"] == ["No VAT number", "VAT is 10%"]


def test_read_documents_records_failures_as_gaps(monkeypatch):
    ctx = _Ctx({"list_documents": {"documents": [
        {"document_id": 1, "reference": "D-1", "name": "bad.pptx"},
        {"name": "noref.pdf"},
    ]}})
    monkeypatch.setattr(de.document_reader, "fetch_document_meta", lambda cp, **kw: {"document_id": 1, "size": 1, "updated_at": "t"})
    monkeypatch.setattr(de.store, "lookup_cached_sync", lambda **kw: None)

    def boom(*a, **kw):
        raise de.document_reader.DocumentReaderError("This file type is not supported")

    monkeypatch.setattr(de.document_reader, "build_insight", boom)
    out = de.DocumentExtractionAgent()._read_documents(ctx, [1, 2])
    assert out["documents"] == [] and out["read_now"] == 0
    assert any("not supported" in g for g in out["gaps"])
    assert any("not found" in g for g in out["gaps"])  # id 2 has no document_id row → reported, never guessed


def test_synthesize_report_shape():
    ctx = _Ctx({"read_documents": {
        "total": 4, "considered": 2, "read_now": 1, "reused": 1, "gaps": ["x"],
        "documents": [
            {"document_id": 1, "reference": "D-1", "name": "inv.pdf", "linked": True, "doc_type": "invoice",
             "summary_short": "Invoice from ACME", "confidence": 0.92, "suggested_working_papers": ["C-1 · Revenue"],
             "amounts": [{"label": "Total", "value": 1150, "currency": "SAR"}], "red_flags": ["No VAT number"]},
            {"document_id": 2, "reference": "D-2", "name": "lease.pdf", "linked": False, "doc_type": "lease_agreement",
             "summary_short": "5-year lease", "confidence": 0.8, "suggested_working_papers": [], "amounts": [], "red_flags": []},
        ],
    }})
    rep = de.DocumentExtractionAgent().synthesize(ctx)
    assert rep["documents_total"] == 4 and rep["documents_read"] == 2 and rep["documents_newly_read"] == 1
    assert rep["unlinked_documents"] == 1 and rep["red_flags_found"] == 1
    assert rep["document_summaries"][0]["status"] == "invoice"
    assert "Likely supports: C-1 · Revenue" in rep["document_summaries"][0]["action"]
    assert "1,150.00 SAR" in rep["document_summaries"][0]["metrics"] and "92% confidence" in rep["document_summaries"][0]["metrics"]
    assert rep["needs_attention"][0]["reason"] == "No VAT number"
    assert rep["data_gaps"] == ["x"]
    assert "invoice (1)" in rep["summary"] and "lease_agreement (1)" in rep["summary"]
