"""Unit tests for the Substantive Testing evaluation helper (offline — no Bedrock).

The LLM call (``synthesize``) is not exercised; we test the pure-Python ISA 450
work — per-sample status, coverage, misstatement/projected flags, and the
aggregate-vs-materiality roll-up. These are the numbers the LLM may not touch.
"""
from __future__ import annotations

from agent.definitions.substantive_testing import (
    _evaluate_testing,
    _build_conclusion_writes,
    _conclusion_html,
)


def _sampling():
    base = {"overall_materiality": 1_000_000, "performance_materiality": 600_000, "misstatement_trivial": 50_000}
    return {"samples": [
        # not started (population known)
        {**base, "reference": "a", "account_name": "Land", "account_code": "11101",
         "no_of_items": 50, "no_of_tested_items": 0, "misstatement_found": 0, "projected_misstatement": 0},
        # in progress, 40% coverage, misstatement below trivial -> no flag
        {**base, "reference": "b", "account_name": "Cash",
         "no_of_items": 50, "no_of_tested_items": 20, "misstatement_found": 30_000, "projected_misstatement": 40_000},
        # complete, misstatement above trivial AND projected exceeds performance materiality
        {**base, "reference": "c", "account_name": "Revenue",
         "no_of_items": 10, "no_of_tested_items": 10, "misstatement_found": 80_000, "projected_misstatement": 700_000},
        # tested, population size unknown
        {**base, "reference": "d", "account_name": "PPE",
         "no_of_items": 0, "no_of_tested_items": 29, "misstatement_found": 0, "projected_misstatement": 0},
    ]}


def test_status_coverage_and_flags():
    out = _evaluate_testing(_sampling())
    by_ref = {r["reference"]: r for r in out["samples"]}

    assert by_ref["a"]["status"] == "not_started"
    assert by_ref["a"]["coverage_pct"] == 0.0
    assert by_ref["a"]["flags"] == ["not yet tested"]

    assert by_ref["b"]["status"] == "in_progress"
    assert by_ref["b"]["coverage_pct"] == 40.0
    assert by_ref["b"]["flags"] == []

    assert by_ref["c"]["status"] == "complete"
    assert by_ref["c"]["coverage_pct"] == 100.0
    assert "misstatement above the trivial threshold" in by_ref["c"]["flags"]
    assert "projected misstatement exceeds performance materiality" in by_ref["c"]["flags"]

    assert by_ref["d"]["status"] == "tested"
    assert by_ref["d"]["coverage_pct"] is None


def test_aggregate_rollup():
    out = _evaluate_testing(_sampling())
    assert out["total"] == 4
    assert out["not_started"] == 1
    assert out["tested"] == 3
    assert out["attention_count"] == 2  # sample a (untested) + sample c (misstatements)
    assert out["aggregate_projected_misstatement"] == 740_000.0
    assert out["file_overall_materiality"] == 1_000_000.0
    assert out["aggregate_within_materiality"] is True


def test_aggregate_exceeds_materiality():
    s = _sampling()
    s["samples"][2]["projected_misstatement"] = 2_000_000  # > overall materiality
    out = _evaluate_testing(s)
    assert out["aggregate_within_materiality"] is False


def test_empty():
    out = _evaluate_testing({})
    assert out == {
        "samples": [], "total": 0, "tested": 0, "not_started": 0, "attention_count": 0,
        "aggregate_projected_misstatement": 0.0, "file_overall_materiality": 0.0,
        "aggregate_within_materiality": None,
    }


# ---------------------------------------------------------------------------
# Write-back payload: the "fill gaps, never overwrite, tested only" policy
# ---------------------------------------------------------------------------
def _sampling_for_writes():
    base = {"overall_materiality": 1_000_000, "performance_materiality": 600_000,
            "misstatement_trivial": 50_000, "no_of_items": 10, "no_of_tested_items": 10,
            "misstatement_found": 0, "projected_misstatement": 0}
    return {"samples": [
        # tested, one test, no existing conclusion -> WRITE TT
        {**base, "sample_id": 101, "reference": "a", "account_name": "Land", "tests": ["TT"], "concluded_tests": []},
        # not started -> skip
        {**base, "sample_id": 102, "reference": "b", "account_name": "Cash", "tests": ["CO"],
         "concluded_tests": [], "no_of_tested_items": 0},
        # tested but its only test already concluded -> skip
        {**base, "sample_id": 103, "reference": "c", "account_name": "Revenue", "tests": ["AA"], "concluded_tests": ["AA"]},
        # tested, two tests, one already concluded -> WRITE only CO
        {**base, "sample_id": 104, "reference": "d", "account_name": "PPE", "tests": ["TT", "CO"], "concluded_tests": ["TT"]},
        # tested but no sample_id from source -> skip
        {**base, "sample_id": None, "reference": "e", "account_name": "Stock", "tests": ["TT"], "concluded_tests": []},
        # tested but no test method set up -> skip
        {**base, "sample_id": 106, "reference": "f", "account_name": "Payables", "tests": [], "concluded_tests": []},
    ]}


def test_build_conclusion_writes_policy():
    samples = _evaluate_testing(_sampling_for_writes())["samples"]
    writes = _build_conclusion_writes(samples, {0: "Balance fairly stated.", 3: "PPE additions vouched."})

    pairs = {(c["sample_id"], c["test_type_name"]) for c in writes["conclusions"]}
    assert pairs == {(101, "TT"), (104, "CO")}            # gaps filled; existing/untested left alone
    assert all("1audit AI" in c["conclusion_comment"] for c in writes["conclusions"])
    assert len(writes["preview"]) == 2

    reasons = {s["reason"] for s in writes["skipped"]}
    assert reasons == {"not yet tested", "already concluded", "no sample id from source", "no test method set up"}


def test_conclusion_html_numbers_are_code_owned_and_note_escaped():
    r = {"flags": [], "tested_items": 10, "population_items": 10, "coverage_pct": 100.0,
         "misstatement_found": 0, "misstatement_trivial": 50_000, "performance_materiality": 600_000,
         "projected_misstatement": 0}
    out = _conclusion_html(r, "<b>see lead schedule</b>")
    assert "tested 10/10 (100.0%)" in out                  # code-built figures
    assert "No exceptions above the trivial threshold" in out
    assert "&lt;b&gt;see lead schedule&lt;/b&gt;" in out    # LLM note is HTML-escaped
    assert "Drafted with 1audit AI" in out

    flagged = _conclusion_html({**r, "flags": ["misstatement above the trivial threshold"]}, "")
    assert "Attention required" in flagged
