"""
Risk Assessment agent (agent_type = "risk_assessment").

Goal: "Identify and assess the risks of material misstatement (ISA 315) on this
file." Read-only — it proposes risks, assertions and responses for the auditor to
accept/edit; it writes nothing, so it needs NO 1audit-be change.

Numbers discipline: the LLM may reason about risk (a qualitative judgement) but
never invents a financial figure. The quantitative SIGNALS it reasons from —
the largest balances, the biggest year-on-year movements, negative balances —
are computed in PURE PYTHON in ``_risk_signals`` from the trial balance, so every
number the assessment cites traces back to read data.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from agent.types import PlannedStep, RunContext, register_definition
from agent.definitions.file_review import _num
from structured import generate_structured


_BALANCE_FRACTION = 0.05    # "significant balance" = >= 5% of the largest balance
_MOVE_PCT = 30.0
_MOVE_FRACTION = 0.005      # a movement matters if |delta| >= 0.5% of the largest balance
_TOP = 12


# ---------------------------------------------------------------------------
# Structured result schema
# ---------------------------------------------------------------------------
class RiskItem(BaseModel):
    area: str = Field(description="account, balance, or cycle the risk attaches to")
    assertions: List[str] = Field(default_factory=list, description="relevant assertions, e.g. Existence, Valuation, Completeness")
    risk_level: str = Field(description="High / Medium / Low")
    significant_risk: bool = Field(default=False, description="true if a significant risk under ISA 315")
    rationale: str = Field(description="why this is a risk, citing the provided figures")
    suggested_response: str = Field(default="", description="the audit response / procedure to address it")


class RiskAssessment(BaseModel):
    summary: str = Field(description="2-4 sentence overview of the entity's risk profile")
    risks: List[RiskItem] = Field(default_factory=list)
    already_documented: List[str] = Field(default_factory=list, description="risks already captured in the file's risk register")


# ---------------------------------------------------------------------------
# Deterministic signals (no LLM, no invented numbers) — module-level for testing
# ---------------------------------------------------------------------------
def _label(r: Dict[str, Any]) -> str:
    return f"{r['name']} ({r['code']})" if r.get("code") else r["name"]


def _tb_accounts(tb: Any) -> List[Dict[str, Any]]:
    """Tolerantly read TB rows {name, code, cy, py} from get_trial_balance."""
    accounts = tb.get("accounts") if isinstance(tb, dict) else (tb if isinstance(tb, list) else [])
    rows: List[Dict[str, Any]] = []
    for a in accounts or []:
        if not isinstance(a, dict):
            continue
        name = a.get("account_name") or a.get("name") or a.get("account_code")
        cy = _num(a.get("cy_amount", a.get("cy")))
        py = _num(a.get("py_amount", a.get("py")))
        if name and cy is not None:
            rows.append({"name": str(name), "code": a.get("account_code"), "cy": cy, "py": py})
    return rows


def _risk_signals(tb: Any, top: int = _TOP) -> Dict[str, Any]:
    """Quantitative risk signals from the trial balance: big balances, big moves,
    negatives. These ground the LLM's risk identification in real figures."""
    rows = _tb_accounts(tb)
    if not rows:
        return {"accounts_considered": 0, "large_balances": [], "large_movements": [], "negative_balances": []}
    scale = max(abs(r["cy"]) for r in rows)
    bal_floor = scale * _BALANCE_FRACTION
    move_floor = scale * _MOVE_FRACTION

    large_balances = sorted((r for r in rows if abs(r["cy"]) >= bal_floor),
                            key=lambda r: abs(r["cy"]), reverse=True)[:top]

    moves: List[Dict[str, Any]] = []
    for r in rows:
        if r["py"] is None:
            continue
        delta = r["cy"] - r["py"]
        pct = (delta / r["py"] * 100.0) if r["py"] != 0 else None
        meaningful = (pct is not None and abs(pct) >= _MOVE_PCT) or (r["py"] == 0 and r["cy"] != 0)
        if meaningful and abs(delta) >= move_floor:
            moves.append({**r, "delta": round(delta, 2), "pct": (round(pct, 1) if pct is not None else None)})
    moves.sort(key=lambda r: abs(r["delta"]), reverse=True)

    negatives = sorted((r for r in rows if r["cy"] < 0), key=lambda r: r["cy"])[:top]

    return {
        "accounts_considered": len(rows),
        "balance_floor": round(bal_floor, 2),
        "movement_floor": round(move_floor, 2),
        "large_balances": [{"account": _label(r), "cy": round(r["cy"], 2)} for r in large_balances],
        "large_movements": [{"account": _label(r), "cy": round(r["cy"], 2), "py": round(r["py"], 2),
                             "delta": r["delta"], "pct": r["pct"]} for r in moves[:top]],
        "negative_balances": [{"account": _label(r), "cy": round(r["cy"], 2)} for r in negatives],
    }


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------
class RiskAssessmentAgent:
    agent_type = "risk_assessment"
    allowed_tools = [
        "get_audit_file_summary",
        "get_trial_balance",
        "get_financial_statement",
        "get_risks",
        "get_materiality",
    ]

    def default_goal(self, audit_file_id: int) -> str:
        return f"Identify and assess the risks of material misstatement (ISA 315) on audit file {audit_file_id}."

    def build_plan(self, ctx: RunContext) -> List[PlannedStep]:
        return [
            PlannedStep("Read file summary", "read", "get_audit_file_summary"),
            PlannedStep("Read trial balance", "read", "get_trial_balance"),
            PlannedStep("Read income statement", "read", "get_financial_statement", {"statement_type": "income_statement"}),
            PlannedStep("Read documented risks", "read", "get_risks"),
            PlannedStep("Read materiality", "read", "get_materiality"),
            PlannedStep("Surface risk signals", "compute", "risk_signals"),
        ]

    def execute_step(self, step: PlannedStep, ctx: RunContext) -> Any:
        if step.tool == "risk_signals":
            return _risk_signals(ctx.find("get_trial_balance"))
        raise ValueError(f"risk_assessment: unknown compute step '{step.tool}'")

    def synthesize(self, ctx: RunContext) -> Dict[str, Any]:
        signals = ctx.find("risk_signals") or {}
        gaps: List[str] = []
        if not signals.get("accounts_considered"):
            gaps.append("trial balance not readable — risk signals are limited")

        payload = {
            "entity": ctx.find("get_audit_file_summary"),
            "materiality": ctx.find("get_materiality"),
            "income_statement": ctx.find("get_financial_statement", statement_type="income_statement"),
            "documented_risks": ctx.find("get_risks"),
            "signals": signals,
        }
        prompt = (
            "You are performing risk assessment (ISA 315) on one audit file. The quantitative signals below "
            "(largest balances, biggest year-on-year movements, negative balances) were ALREADY COMPUTED in code "
            "from the trial balance — treat every number as fixed.\n\n"
            f"DATA (JSON):\n{json.dumps(payload, default=str)[:90000]}\n\n"
            "Produce:\n"
            "- summary: 2-4 sentences on the entity's overall risk profile.\n"
            "- risks: the risks of material misstatement. For EACH, give the `area` (account/balance/cycle), the "
            "relevant `assertions`, a `risk_level` (High/Medium/Low), `significant_risk` (true/false per ISA 315), a "
            "`rationale` that cites the provided figures, and a `suggested_response` (the audit procedure).\n"
            "- already_documented: risks that the file's documented_risks already capture (so the auditor sees coverage).\n"
            "Base risks on the signals and entity facts; cite figures only from the data; never invent a number."
        )
        system = (
            "You are a senior auditor assessing risks of material misstatement. Reason about risk qualitatively, but "
            "cite ONLY the figures provided (computed in code) and never invent or recompute a number. Prioritise "
            "material balances and large movements; be specific and link each risk to assertions and a response."
        )
        report = generate_structured(prompt, RiskAssessment, system=system, usage_out=ctx.usage_out)

        risks_out = []
        for ri in report.risks:
            metrics = "Assertions: " + " · ".join(ri.assertions) if ri.assertions else ""
            if ri.significant_risk:
                metrics = (metrics + " · " if metrics else "") + "Significant risk"
            risks_out.append({
                "title": ri.area,
                "status": ri.risk_level,
                "metrics": metrics,
                "reason": ri.rationale,
                "action": (f"Response: {ri.suggested_response}" if ri.suggested_response else ""),
            })
        return {
            "summary": report.summary,
            "risks_identified": len(report.risks),
            "significant_risks": sum(1 for ri in report.risks if ri.significant_risk),
            "risks": risks_out,
            "already_documented": list(report.already_documented or []),
            "data_gaps": gaps,
        }


register_definition(RiskAssessmentAgent())
