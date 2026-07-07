"""Unit tests for the Risk Assessment signal helper (offline — no Bedrock).

The LLM call (``synthesize``) is not exercised; we test the pure-Python signals —
the largest balances, biggest movements, and negatives the assessment reasons
from. These are the figures the LLM may cite but never invent.
"""
from __future__ import annotations

from agent.definitions.risk_assessment import _risk_signals


def _tb():
    return {"accounts": [
        {"account_name": "Loan", "account_code": "L1", "cy_amount": 1_000_000, "py_amount": 1_000_000},   # big balance, no move
        {"account_name": "Cash", "account_code": "C1", "cy_amount": 600_000, "py_amount": 100_000},        # big balance + big move
        {"account_name": "Provision", "account_code": "P1", "cy_amount": -300_000, "py_amount": -200_000},  # negative + big balance + move
        {"account_name": "Petty cash", "account_code": "PC", "cy_amount": 100, "py_amount": 90},            # immaterial -> excluded
    ]}


def test_thresholds_and_buckets():
    out = _risk_signals(_tb())
    assert out["accounts_considered"] == 4
    assert out["balance_floor"] == 50_000.0      # 5% of 1,000,000
    assert out["movement_floor"] == 5_000.0      # 0.5% of 1,000,000

    bal = [b["account"] for b in out["large_balances"]]
    assert bal[0] == "Loan (L1)"                  # largest first
    assert "Cash (C1)" in bal and "Provision (P1)" in bal
    assert "Petty cash (PC)" not in bal           # immaterial dropped

    mv = [m["account"] for m in out["large_movements"]]
    assert mv[0] == "Cash (C1)"                    # biggest delta first
    assert "Provision (P1)" in mv
    assert "Petty cash (PC)" not in mv             # delta below floor

    neg = [n["account"] for n in out["negative_balances"]]
    assert neg == ["Provision (P1)"]


def test_empty():
    out = _risk_signals({})
    assert out == {"accounts_considered": 0, "large_balances": [], "large_movements": [], "negative_balances": []}
