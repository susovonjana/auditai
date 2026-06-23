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
