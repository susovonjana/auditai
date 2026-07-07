"""Unit tests for the Analytical Review deterministic helpers (offline — no Bedrock).

The LLM call (``synthesize``) is not exercised here; we test the pure-Python figure
work — fluctuation detection (dual threshold, ranking, direction, new-balance) and
ratio computation (correct values, close-label disambiguation, skip-when-missing).
These are exactly the numbers the LLM is forbidden from touching.
"""
from __future__ import annotations

from agent.definitions.analytical_review import _compute_fluctuations, _compute_ratios


def _row(name, cy, py):
    # Keys chosen to be understood by file_review._extract_lines.
    return {"name": name, "current_year": cy, "prior_year": py}


def _statements(bs_rows=None, is_rows=None):
    out = {}
    if bs_rows is not None:
        out["balance_sheet"] = {"lines": bs_rows}
    if is_rows is not None:
        out["income_statement"] = {"lines": is_rows}
    return out


# ---------------------------------------------------------------------------
# Fluctuations
# ---------------------------------------------------------------------------
def test_dual_threshold_filters_trivial_lines():
    # scale = 1,000,000 -> abs_floor = 0.5% = 5,000.
    statements = _statements(is_rows=[
        _row("Revenue", 1_000_000, 900_000),   # +11.1%, delta 100k  -> SIGNIFICANT
        _row("Misc income", 110, 100),          # +10% but delta 10  -> below value floor, dropped
        _row("Rent", 100_000, 99_500),          # +0.5%, delta 500   -> below pct threshold, dropped
    ])
    out = _compute_fluctuations(statements)
    lines = {f["line"] for f in out["fluctuations"]}
    assert lines == {"Revenue"}
    assert out["lines_analyzed"] == 3
    assert out["significant_count"] == 1
    assert out["abs_floor"] == 5000.0


def test_new_balance_and_direction_and_ranking():
    statements = _statements(bs_rows=[
        _row("Loan", 0, 0),                      # unchanged, ignored
        _row("New borrowings", 400_000, 0),      # py=0, cy!=0 -> flagged (new balance), increase
        _row("Cash", 200_000, 800_000),          # -75%, delta -600k -> decrease, biggest move
    ])
    out = _compute_fluctuations(statements)
    # Ranked by |delta| desc: Cash (600k) before New borrowings (400k).
    assert [f["line"] for f in out["fluctuations"]] == ["Cash", "New borrowings"]
    cash = out["fluctuations"][0]
    assert cash["direction"] == "decrease" and cash["pct"] == -75.0
    newb = out["fluctuations"][1]
    assert newb["direction"] == "increase" and newb["pct"] is None  # py=0 -> pct not defined


def test_missing_statement_is_a_gap_not_a_guess():
    out = _compute_fluctuations(_statements(is_rows=[_row("Revenue", 100, 50)]))
    assert any("balance sheet" in g for g in out["gaps"])  # BS absent
    assert out["fluctuations"][0]["line"] == "Revenue"


# ---------------------------------------------------------------------------
# Ratios
# ---------------------------------------------------------------------------
def _full_statements():
    bs = [
        _row("Total current assets", 600, 500),
        _row("Total assets", 1200, 1050),
        _row("Total current liabilities", 300, 250),
        _row("Total liabilities", 500, 450),
        _row("Total liabilities and equity", 1200, 1050),  # decoy: must NOT be read as total liabilities
        _row("Total equity", 700, 600),
    ]
    is_ = [
        _row("Revenue", 1000, 800),
        _row("Cost of sales", 600, 500),   # decoy: must NOT be read as revenue
        _row("Gross profit", 400, 300),
        _row("Net profit", 200, 150),
    ]
    return _statements(bs_rows=bs, is_rows=is_)


def test_ratios_values_and_label_disambiguation():
    ratios = {r["line"]: r for r in _compute_ratios(_full_statements())["ratios"]}
    assert ratios["Gross margin %"]["cy"] == 40.0 and ratios["Gross margin %"]["py"] == 37.5
    assert ratios["Current ratio"]["cy"] == 2.0 and ratios["Current ratio"]["py"] == 2.0
    # Debt-to-equity must use 'Total liabilities' (500/700), NOT the 'and equity' decoy (1200/700).
    assert ratios["Debt to equity"]["cy"] == round(500 / 700, 2)
    assert ratios["Return on assets %"]["cy"] == round(200 / 1200 * 100, 1)
    assert ratios["Net margin %"]["cy"] == 20.0


def test_ratio_skipped_when_component_missing():
    # No equity line -> debt-to-equity must be absent (skipped, never guessed).
    bs = [_row("Total current assets", 600, 500), _row("Total current liabilities", 300, 250)]
    names = {r["line"] for r in _compute_ratios(_statements(bs_rows=bs))["ratios"]}
    assert "Current ratio" in names
    assert "Debt to equity" not in names


def test_empty_statements():
    assert _compute_fluctuations({})["fluctuations"] == []
    assert _compute_ratios({})["ratios"] == []
