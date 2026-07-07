"""
File Review / wrap-up agent (agent_type = "file_review").

Goal: "Review this audit file for gaps before sign-off." Read-only — it surfaces
findings; the auditor acts on them. It runs entirely on existing read tools, so
it needs NO 1audit-be change.

Numbers discipline: period movements are computed in PURE PYTHON
(``_compute_movements``); the LLM only classifies the provided lists and writes
prose. If a figure can't be read, it is reported under ``data_gaps`` — never
guessed.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from agent.types import PlannedStep, RunContext, register_definition
from structured import generate_structured


# ---------------------------------------------------------------------------
# Structured result schema
# ---------------------------------------------------------------------------
class AccountFinding(BaseModel):
    account: str = Field(description="account code/name as given in the data")
    reason: str = Field(description="one line: why it needs attention")


class WorkingPaperFinding(BaseModel):
    working_paper: str
    status: Optional[str] = Field(default=None, description="raw status, e.g. in_progress / not_started")
    reason: str


class RiskFinding(BaseModel):
    risk: str
    reason: str


class MovementFinding(BaseModel):
    statement: str
    line: str
    cy: Optional[float] = None
    py: Optional[float] = None
    delta: Optional[float] = None
    pct: Optional[float] = Field(default=None, description="percent change, already computed")
    reason: str = ""


class NeedsAttentionReport(BaseModel):
    summary: str = Field(description="2-3 sentence overview of the file's readiness")
    unmapped_or_odd_accounts: List[AccountFinding] = Field(default_factory=list)
    working_papers_not_signed_off: List[WorkingPaperFinding] = Field(default_factory=list)
    risks_without_coverage: List[RiskFinding] = Field(default_factory=list)
    large_movements: List[MovementFinding] = Field(default_factory=list)
    data_gaps: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Deterministic helpers (no LLM, no invented numbers)
# ---------------------------------------------------------------------------
_NAME_KEYS = ("name", "label", "account_name", "title", "line", "description", "account", "caption")
_CY_KEYS = ("current_year", "cy", "current", "cy_amount", "current_amount", "current_balance", "amount", "balance", "value")
_PY_KEYS = ("prior_year", "py", "previous", "prior", "py_amount", "previous_amount", "prior_amount", "comparative", "last_year")


def _is_error(value: Any) -> bool:
    return isinstance(value, dict) and "error" in value


def _num(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.replace(",", "").replace("(", "-").replace(")", "").strip()
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _first(node: dict, keys) -> Any:
    for k in keys:
        if k in node:
            return node[k]
    return None


def _extract_lines(data: Any) -> List[Dict[str, Any]]:
    """Walk an arbitrary financial-statement payload and collect leaf rows that
    carry a label plus a current-year and prior-year amount. Tolerant of unknown
    shapes: anything it can't read simply isn't returned (-> reported as a gap)."""
    rows: List[Dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            name = _first(node, _NAME_KEYS)
            cy = _num(_first(node, _CY_KEYS))
            py = _num(_first(node, _PY_KEYS))
            if isinstance(name, str) and name.strip() and cy is not None and py is not None:
                rows.append({"name": name.strip(), "cy": cy, "py": py})
            for v in node.values():
                if isinstance(v, (dict, list)):
                    walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(data)
    return rows


def _truncate(value: Any, limit: int) -> Any:
    if isinstance(value, list) and len(value) > limit:
        return value[:limit]
    return value


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------
class FileReviewAgent:
    agent_type = "file_review"
    allowed_tools = [
        "get_audit_file_summary",
        "get_trial_balance",
        "list_working_papers",
        "get_risks",
        "get_financial_statement",
    ]

    LARGE_MOVE_PCT = 30.0  # flag a line if |% change| >= this (or PY=0, CY!=0)
    MAX_MOVEMENTS = 50

    def default_goal(self, audit_file_id: int) -> str:
        return f"Review audit file {audit_file_id} for gaps before sign-off."

    def build_plan(self, ctx: RunContext) -> List[PlannedStep]:
        return [
            PlannedStep("Read file summary", "read", "get_audit_file_summary"),
            PlannedStep("Read trial balance", "read", "get_trial_balance"),
            PlannedStep("List working papers", "read", "list_working_papers"),
            PlannedStep("Read risks", "read", "get_risks"),
            PlannedStep(
                "Read balance sheet", "read", "get_financial_statement",
                {"statement_type": "balance_sheet"},
            ),
            PlannedStep(
                "Read income statement", "read", "get_financial_statement",
                {"statement_type": "income_statement"},
            ),
            PlannedStep("Compute period movements", "compute", "compute_movements"),
        ]

    def execute_step(self, step: PlannedStep, ctx: RunContext) -> Any:
        if step.tool == "compute_movements":
            return self._compute_movements(ctx)
        raise ValueError(f"file_review: unknown compute step '{step.tool}'")

    def _compute_movements(self, ctx: RunContext) -> Dict[str, Any]:
        movements: List[Dict[str, Any]] = []
        gaps: List[str] = []
        parsed = 0
        for label, key in (("balance_sheet", "balance_sheet"), ("income_statement", "income_statement")):
            data = ctx.find("get_financial_statement", statement_type=key)
            if data is None or _is_error(data):
                gaps.append(f"{label}: not available")
                continue
            rows = _extract_lines(data)
            if not rows:
                gaps.append(f"{label}: could not read comparative line items")
                continue
            for r in rows:
                parsed += 1
                cy, py = r["cy"], r["py"]
                delta = cy - py
                pct = (delta / py * 100.0) if py != 0 else None
                big = (pct is not None and abs(pct) >= self.LARGE_MOVE_PCT) or (py == 0 and cy != 0)
                if big:
                    movements.append({
                        "statement": label,
                        "line": r["name"],
                        "cy": round(cy, 2),
                        "py": round(py, 2),
                        "delta": round(delta, 2),
                        "pct": (round(pct, 1) if pct is not None else None),
                    })
        capped = len(movements) > self.MAX_MOVEMENTS
        if capped:
            gaps.append(f"movements truncated to first {self.MAX_MOVEMENTS}")
        return {
            "movements": movements[: self.MAX_MOVEMENTS],
            "lines_parsed": parsed,
            "threshold_pct": self.LARGE_MOVE_PCT,
            "gaps": gaps,
        }

    def synthesize(self, ctx: RunContext) -> Dict[str, Any]:
        summary = ctx.find("get_audit_file_summary")
        tb = ctx.find("get_trial_balance")
        wps = ctx.find("list_working_papers")
        risks = ctx.find("get_risks")
        movements = ctx.find("compute_movements") or {}

        payload = {
            "file_summary": summary,
            "trial_balance": _truncate(tb, 200),
            "working_papers": _truncate(wps, 120),
            "risks": _truncate(risks, 120),
            "computed_movements": movements,
        }
        prompt = (
            "You are reviewing one audit file for gaps before sign-off. Below is the data "
            "already gathered from the file (read-only tools) and the period movements "
            "ALREADY COMPUTED in code.\n\n"
            f"DATA (JSON):\n{json.dumps(payload, default=str)[:120000]}\n\n"
            "Produce a 'needs attention' report:\n"
            "- unmapped_or_odd_accounts: accounts in the trial balance that look unmapped, "
            "uncategorised, suspense/clearing, or have an odd sign for their type.\n"
            "- working_papers_not_signed_off: working papers whose status indicates they are "
            "not yet reviewed/signed off. For each, set `status` to the working paper's RAW "
            "status value from the data (e.g. in_progress, not_started) and keep `reason` to a "
            "short plain-English note WITHOUT repeating the status.\n"
            "- risks_without_coverage: identified risks with no linked procedure or working paper.\n"
            "- large_movements: copy ONLY from computed_movements.movements (do not recompute or "
            "add lines); write a one-line reason for each.\n"
            "- data_gaps: anything that could not be retrieved (include computed_movements.gaps).\n"
        )
        system = (
            "You are a senior auditor's assistant. Use ONLY the values provided. NEVER invent, "
            "estimate, or recompute a figure — the figures in computed_movements were calculated "
            "in code and are the only numbers you may report. If something is missing, list it "
            "under data_gaps rather than guessing. Be concise and specific."
        )
        report = generate_structured(
            prompt,
            NeedsAttentionReport,
            system=system,
            usage_out=ctx.usage_out,
        )
        return report.model_dump(mode="json")


register_definition(FileReviewAgent())
