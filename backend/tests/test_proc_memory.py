"""Offline unit tests for proc_memory pure logic (ticket A-1b).

The embedding/pgvector search path needs a live DB + model, so it is exercised by
the end-to-end smoke test, not here. These cover the deterministic helpers and the
guard clauses that short-circuit BEFORE any DB/embedding work.
"""
import asyncio

import proc_memory


def test_memory_key_combines_area_and_risk():
    assert proc_memory.memory_key("Receivables", "existence") == "Receivables — existence"
    assert proc_memory.memory_key("Receivables", None) == "Receivables"
    assert proc_memory.memory_key(None, None) == "audit procedure"


def test_summarize_risks_formats_title_and_description():
    out = proc_memory.summarize_risks(
        [{"title": "R1", "description": "d1"}, {"title": "R2"}, {"description": "d3"}]
    )
    assert "R1: d1" in out
    assert "R2" in out
    assert "d3" in out


def test_add_memory_guards_short_circuit_without_db():
    # Empty procedure → None, and no DB/embedding is touched (db=None is safe).
    assert (
        asyncio.run(
            proc_memory.add_memory(
                None, organization_id="10", client_sector=None, audit_area="X",
                risk_summary="y", assertions=[], procedure_html="   ",
            )
        )
        is None
    )
    # Missing org scope → None (a firm's memory must be org-scoped).
    assert (
        asyncio.run(
            proc_memory.add_memory(
                None, organization_id=None, client_sector=None, audit_area="X",
                risk_summary="y", assertions=[], procedure_html="<ol><li>x</li></ol>",
            )
        )
        is None
    )


def test_search_examples_without_org_returns_empty():
    # No org → [] immediately (no DB/embedding work).
    assert asyncio.run(
        proc_memory.search_examples(
            None, organization_id=None, client_sector="Retail",
            audit_area="Receivables", risk_summary="existence",
        )
    ) == []


# ---------------------------------------------------------------------------
# Provenance + dedupe (Phase 2 feedback loop)
# ---------------------------------------------------------------------------
def test_content_hash_stable_under_markup_and_whitespace():
    a = proc_memory.content_hash("Inventory", "<p>Attend the  count.</p>")
    b = proc_memory.content_hash("Inventory", "<ol><li>Attend the count.</li></ol>")
    c = proc_memory.content_hash("Inventory", "ATTEND THE COUNT.")
    assert a == b == c
    # wording change -> different hash
    assert proc_memory.content_hash("Inventory", "<p>Attend the recount.</p>") != a
    # the area participates in the hash
    assert proc_memory.content_hash("Receivables", "<p>Attend the count.</p>") != a


def test_extract_memory_rows_policy():
    payload = {
        "working_paper_id": 77,
        "ai_run_id": "r1",
        "sections": [
            {"temp_id": "t1", "section_type": 1, "title": "General"},                       # title -> skipped
            {"temp_id": "t2", "section_type": 3, "description": "<p>Objective: evidence over inventory quantities.</p>"},  # comment -> skipped
            {"temp_id": "t3", "section_type": 2, "procedure": "<p>ok</p>"},                 # too short -> skipped
            {"temp_id": "t4", "section_type": 2,
             "procedure": "<p>Attend the year-end inventory count and observe the client's procedures.</p>",
             "assertions": ["Existence", " Valuation ", ""]},
        ],
    }
    rows = proc_memory.extract_memory_rows(
        payload, run_id="RUN", audit_area="Inventory", risk_summary="obsolescence risk",
        client_sector="Manufacturing", confirmed_by="27",
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["audit_area"] == "Inventory"
    assert row["risk_summary"] == "obsolescence risk"
    assert row["client_sector"] == "Manufacturing"
    assert row["confirmed_by"] == "27"
    assert row["assertions"] == ["Existence", "Valuation"]
    assert row["source_key"] == "agent:RUN:t4"
    assert row["procedure_html"].startswith("<p>Attend")


def test_extract_memory_rows_caps_and_empty():
    assert proc_memory.extract_memory_rows(
        None, run_id="r", audit_area=None, risk_summary=None, client_sector=None, confirmed_by=None,
    ) == []
    many = {
        "sections": [
            {"temp_id": f"t{i}", "section_type": 2,
             "procedure": f"<p>Perform detailed testing over balance number {i} with supporting evidence.</p>"}
            for i in range(60)
        ]
    }
    rows = proc_memory.extract_memory_rows(
        many, run_id="r", audit_area="X", risk_summary=None, client_sector=None, confirmed_by=None,
    )
    assert len(rows) == 40  # _MAX_ROWS_PER_INGEST
