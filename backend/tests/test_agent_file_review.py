"""Unit tests for the File Review agent's DETERMINISTIC parts (no LLM).

Proves the core guarantee: period movements are computed in code from the
provided figures, large movements are flagged correctly, and missing data is
reported as a gap rather than guessed. synthesize() (which calls Bedrock) is not
exercised here — only the pure-Python compute path.
"""
from __future__ import annotations

from agent.definitions.file_review import (
    FileReviewAgent,
    _extract_lines,
    _num,
)
from agent.types import RunContext, StepResult
from copilot_tools import CopilotContext


def _ctx_with(bs=None, is_=None) -> RunContext:
    ctx = RunContext(copilot=CopilotContext(1, ""), audit_file_id=1)
    if bs is not None:
        ctx.results.append(
            StepResult(0, "bs", "read", "get_financial_statement",
                       {"statement_type": "balance_sheet"}, bs)
        )
    if is_ is not None:
        ctx.results.append(
            StepResult(1, "is", "read", "get_financial_statement",
                       {"statement_type": "income_statement"}, is_)
        )
    return ctx


def test_num_parses_messy_values():
    assert _num("1,500") == 1500.0
    assert _num("(200)") == -200.0
    assert _num(42) == 42.0
    assert _num("n/a") is None
    assert _num(True) is None


def test_extract_lines_walks_nested_payload():
    payload = {
        "sections": [
            {"title": "Assets", "lines": [
                {"name": "Cash", "current_year": 1500, "prior_year": 1000},
                {"name": "Receivables", "cy": "2,000", "py": "2,100"},
            ]},
        ]
    }
    rows = _extract_lines(payload)
    names = {r["name"]: (r["cy"], r["py"]) for r in rows}
    assert names["Cash"] == (1500.0, 1000.0)
    assert names["Receivables"] == (2000.0, 2100.0)


def test_compute_movements_flags_large_and_reports_gaps():
    bs = {"lines": [
        {"name": "Cash", "current_year": 1500, "prior_year": 1000},      # +50% -> flagged
        {"name": "Inventory", "current_year": 1010, "prior_year": 1000},  # +1%  -> not flagged
        {"name": "Suspense", "current_year": 50, "prior_year": 0},        # PY=0 -> flagged
    ]}
    agent = FileReviewAgent()
    out = agent._compute_movements(_ctx_with(bs=bs, is_=None))

    flagged = {m["line"]: m for m in out["movements"]}
    assert set(flagged) == {"Cash", "Suspense"}          # Inventory below threshold
    assert flagged["Cash"]["delta"] == 500.0 and flagged["Cash"]["pct"] == 50.0
    assert flagged["Suspense"]["py"] == 0.0 and flagged["Suspense"]["pct"] is None
    assert out["lines_parsed"] == 3
    # income statement was not provided -> reported as a gap, not invented
    assert any("income_statement" in g for g in out["gaps"])
    # every reported figure traces back to the input (no invented numbers)
    for m in out["movements"]:
        assert m["cy"] in (1500.0, 50.0) and m["py"] in (1000.0, 0.0)
