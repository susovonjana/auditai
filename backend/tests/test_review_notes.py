"""Unit tests for the Review Notes / EQR signal helper (offline — no Bedrock).

The LLM call (``synthesize``) is not exercised; we test the pure-Python counts —
working-paper sign-off buckets, the unsigned list, and the existing review-point
open/cleared split — that ground the reviewer.
"""
from __future__ import annotations

from agent.definitions.review_notes import _review_signals


def _wps():
    return {"working_papers": [
        {"working_paper_id": 1, "name": "WP1", "reference": "r1", "status": "completed", "section": "F1"},
        {"working_paper_id": 2, "name": "WP2", "reference": "r2", "status": "in_progress", "section": "F1"},
        {"working_paper_id": 3, "name": "WP3", "reference": "r3", "status": "not_started", "section": "F2"},
        {"working_paper_id": 4, "name": "WP4", "reference": "r4", "status": "", "section": "F2"},  # blank -> not started
    ]}


def _rps():
    return {"review_points": [
        {"number": 1, "comment": "a", "resolved": True, "working_paper": "WP1"},
        {"number": 2, "comment": "b", "resolved": False, "working_paper": "WP2", "reviewed": True},
        {"number": 3, "comment": "c", "resolved": False, "working_paper": "WP3"},
    ]}


def test_wp_buckets_and_unsigned():
    out = _review_signals(_wps(), _rps())
    assert out["wp_total"] == 4
    assert out["wp_signed"] == 1          # WP1 completed
    assert out["wp_in_progress"] == 1     # WP2
    assert out["wp_not_started"] == 2     # WP3 + WP4 (blank)
    names = {w["name"] for w in out["unsigned_working_papers"]}
    assert names == {"WP2", "WP3", "WP4"}  # everything not signed off


def test_review_point_open_cleared_split():
    out = _review_signals(_wps(), _rps())
    assert out["review_points_total"] == 3
    assert out["review_points_open"] == 2
    assert out["review_points_cleared"] == 1
    open_nums = {p["number"] for p in out["open_review_points"]}
    assert open_nums == {2, 3}             # only the unresolved ones


def test_empty():
    out = _review_signals({}, {})
    assert out["wp_total"] == 0 and out["review_points_total"] == 0
    assert out["unsigned_working_papers"] == [] and out["open_review_points"] == []
