"""Unit tests for the TB column-detect post-processing (offline — no Bedrock).

The LLM call is not exercised here; we test the deterministic filtering that turns
the model's match list into the { mapping, confidence } the FE consumes — invalid
fields/columns dropped, each field mapped once, confidence rounded. Header labels
may be Arabic; only the column letters matter.
"""
from __future__ import annotations

from routers.copilot import _build_detect_result


def _headers():
    return [
        {"column": "A", "label": "Account Code"},
        {"column": "B", "label": "رواتب"},        # Arabic label — column letter is what matters
        {"column": "C", "label": "2017"},
        {"column": "D", "label": "2016"},
    ]


def test_keeps_valid_and_drops_invalid():
    matches = [
        {"field": "account_code", "column": "A", "confidence": 0.95},
        {"field": "account_name", "column": "B", "confidence": 0.8},
        {"field": "cy_amount", "column": "C", "confidence": 0.9},
        {"field": "not_a_field", "column": "D", "confidence": 0.9},   # invalid field -> dropped
        {"field": "py_amount", "column": "Z", "confidence": 0.9},     # column not in headers -> dropped
    ]
    out = _build_detect_result(matches, _headers())
    assert out["mapping"] == {"account_code": "A", "account_name": "B", "cy_amount": "C"}
    assert out["confidence"]["account_code"] == 0.95


def test_dedup_first_wins_and_rounds():
    matches = [
        {"field": "cy_amount", "column": "C", "confidence": 0.912},
        {"field": "cy_amount", "column": "D", "confidence": 0.4},     # same field again -> ignored
    ]
    out = _build_detect_result(matches, _headers())
    assert out["mapping"] == {"cy_amount": "C"}
    assert out["confidence"]["cy_amount"] == 0.91                     # rounded to 2dp


def test_empty():
    assert _build_detect_result([], _headers()) == {"mapping": {}, "confidence": {}}
