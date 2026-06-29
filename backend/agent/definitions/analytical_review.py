"""
Analytical Review agent (agent_type = "analytical_review").

Goal: "Perform analytical review (ISA 520) on this audit file." Read-only — it
reviews the numbers and drafts the analytical-review commentary for the auditor;
it changes nothing, so it needs NO 1audit-be change.

Numbers discipline (the whole point of this agent): every figure — period
fluctuations AND key ratios — is computed in PURE PYTHON here. The LLM is given
those finished numbers and may only write prose: a business explanation for each
movement, what evidence to corroborate it with, ratio commentary, and the overall
conclusion. The model never sees a blank where a number should be and never
returns a number we keep — its notes are merged back onto the Python figures by
index, so a hallucinated value can't reach the report.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from agent.types import PlannedStep, RunContext, register_definition
from agent.definitions.file_review import _extract_lines, _is_error  # shared, tolerant FS parser
from structured import generate_structured


# ---------------------------------------------------------------------------
# Thresholds (transparent + tunable)
# ---------------------------------------------------------------------------
_PCT_THRESHOLD = 10.0     # a line is "significant" only if |% change| >= this …
_SCALE_FRACTION = 0.005   # … AND |delta| >= 0.5% of the largest balance (size proxy)
_MAX_FLUX = 40            # cap the list the LLM annotates


# ---------------------------------------------------------------------------
# LLM output schema — PROSE ONLY (numbers are merged in afterwards by index)
# ---------------------------------------------------------------------------
class FluctuationNote(BaseModel):
    index: int = Field(description="the `index` of the fluctuation this note explains")
    explanation: str = Field(default="", description="plausible business reason / expectation for the movement")
    corroboration: str = Field(default="", description="specific evidence the auditor should obtain to corroborate it")


class RatioNote(BaseModel):
    index: int = Field(description="the `index` of the ratio this note refers to")
    commentary: str = Field(default="", description="one-line read on the trend")


class AnalyticalDraft(BaseModel):
    summary: str = Field(description="2-4 sentence overall analytical-review narrative")
    conclusion: str = Field(description="1-2 sentences: do results look consistent/expected, or warrant more work?")
    fluctuation_notes: List[FluctuationNote] = Field(default_factory=list)
    ratio_notes: List[RatioNote] = Field(default_factory=list)
    follow_up: List[str] = Field(default_factory=list, description="concrete items for the auditor to corroborate / investigate")


# ---------------------------------------------------------------------------
# Deterministic helpers (no LLM, no invented numbers) — module-level for testing
# ---------------------------------------------------------------------------
def _collect_rows(statements: Dict[str, Any]) -> tuple[List[Dict[str, Any]], List[str]]:
    """Flatten BS + IS into comparative leaf rows {statement,name,cy,py}; anything
    unreadable is reported as a gap rather than guessed."""
    rows: List[Dict[str, Any]] = []
    gaps: List[str] = []
    for label in ("balance_sheet", "income_statement"):
        data = statements.get(label)
        if data is None or _is_error(data):
            gaps.append(f"{label.replace('_', ' ')}: not available")
            continue
        parsed = _extract_lines(data)
        if not parsed:
            gaps.append(f"{label.replace('_', ' ')}: could not read comparative line items")
            continue
        for r in parsed:
            rows.append({"statement": label, "name": r["name"], "cy": r["cy"], "py": r["py"]})
    return rows, gaps


def _compute_fluctuations(
    statements: Dict[str, Any],
    pct_threshold: float = _PCT_THRESHOLD,
    scale_fraction: float = _SCALE_FRACTION,
    max_flux: int = _MAX_FLUX,
) -> Dict[str, Any]:
    """Year-on-year movement per line, with a DUAL significance test: a meaningful
    percentage move AND a value big enough to matter for the entity's size (guards
    against flagging trivial lines that swing wildly in % terms)."""
    rows, gaps = _collect_rows(statements)
    scale = max((abs(r["cy"]) for r in rows), default=0.0)
    abs_floor = round(scale * scale_fraction, 2)

    flux: List[Dict[str, Any]] = []
    for r in rows:
        cy, py = r["cy"], r["py"]
        delta = cy - py
        pct = (delta / py * 100.0) if py != 0 else None
        meaningful_pct = (pct is not None and abs(pct) >= pct_threshold) or (py == 0 and cy != 0)
        if meaningful_pct and abs(delta) >= abs_floor:
            flux.append({
                "statement": r["statement"], "line": r["name"],
                "cy": round(cy, 2), "py": round(py, 2), "delta": round(delta, 2),
                "pct": (round(pct, 1) if pct is not None else None),
                "direction": "increase" if delta > 0 else "decrease",
            })
    flux.sort(key=lambda x: abs(x["delta"]), reverse=True)
    return {
        "fluctuations": flux[:max_flux],
        "lines_analyzed": len(rows),
        "significant_count": len(flux),
        "threshold_pct": pct_threshold,
        "abs_floor": abs_floor,
        "capped": len(flux) > max_flux,
        "gaps": gaps,
    }


def _norm(name: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace — so 'Shareholders' Equity'
    and 'Non-current assets' normalise to comparable tokens."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(name).lower())).strip()


def _find_line(rows: List[Dict[str, Any]], includes: List[str], excludes: tuple = ()) -> Optional[Dict[str, Any]]:
    """First row whose normalised name CONTAINS an include phrase (tried in priority
    order) and NONE of the exclude tokens. Contiguous-substring matching plus the
    exclude list disambiguates close labels (e.g. 'total liabilities' must not match
    'total liabilities and equity')."""
    for inc in includes:
        for r in rows:
            n = _norm(r["name"])
            if inc in n and not any(x in n for x in excludes):
                return r
    return None


def _compute_ratios(statements: Dict[str, Any]) -> Dict[str, Any]:
    """A small, high-value ratio set — each computed ONLY when both component lines
    are confidently identified (else silently skipped, never guessed)."""
    rows, _ = _collect_rows(statements)

    revenue = _find_line(rows, ["total revenue", "net sales", "revenue", "turnover", "sales"],
                         excludes=("cost", "deferred", "unearned"))
    gross_profit = _find_line(rows, ["gross profit", "gross margin"])
    net_profit = _find_line(rows, ["profit for the year", "net profit", "net income",
                                   "profit after tax", "profit attributable", "net earnings"],
                            excludes=("before",))
    total_assets = _find_line(rows, ["total assets"])
    current_assets = _find_line(rows, ["total current assets", "current assets"],
                                excludes=("non current", "liabilit"))
    current_liabilities = _find_line(rows, ["total current liabilities", "current liabilities"],
                                     excludes=("non current", "asset"))
    total_liabilities = _find_line(rows, ["total liabilities"], excludes=("current", "equity"))
    total_equity = _find_line(rows, ["total equity", "total shareholders equity", "shareholders equity",
                                     "stockholders equity", "owners equity"], excludes=("liabilit",))

    ratios: List[Dict[str, Any]] = []

    def add(name: str, num: Optional[dict], den: Optional[dict], pct: bool = False) -> None:
        if not num or not den:
            return
        ndp = 1 if pct else 2
        factor = 100.0 if pct else 1.0

        def one(a: Any, b: Any) -> Optional[float]:
            if a is None or b in (0, None):
                return None
            return round(a / b * factor, ndp)

        cy = one(num["cy"], den["cy"])
        py = one(num["py"], den["py"])
        if cy is None and py is None:
            return
        delta = round(cy - py, ndp) if (cy is not None and py is not None) else None
        ratios.append({"line": name, "cy": cy, "py": py, "delta": delta})

    add("Gross margin %", gross_profit, revenue, pct=True)
    add("Net margin %", net_profit, revenue, pct=True)
    add("Current ratio", current_assets, current_liabilities)
    add("Debt to equity", total_liabilities, total_equity)
    add("Return on assets %", net_profit, total_assets, pct=True)
    return {"ratios": ratios}


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------
class AnalyticalReviewAgent:
    agent_type = "analytical_review"
    allowed_tools = [
        "get_audit_file_summary",
        "get_financial_statement",
    ]

    def default_goal(self, audit_file_id: int) -> str:
        return f"Perform analytical review (ISA 520) on audit file {audit_file_id}."

    def build_plan(self, ctx: RunContext) -> List[PlannedStep]:
        return [
            PlannedStep("Read file summary", "read", "get_audit_file_summary"),
            PlannedStep("Read balance sheet", "read", "get_financial_statement", {"statement_type": "balance_sheet"}),
            PlannedStep("Read income statement", "read", "get_financial_statement", {"statement_type": "income_statement"}),
            PlannedStep("Analyse fluctuations", "compute", "compute_fluctuations"),
            PlannedStep("Compute key ratios", "compute", "compute_ratios"),
        ]

    def _statements(self, ctx: RunContext) -> Dict[str, Any]:
        return {
            "balance_sheet": ctx.find("get_financial_statement", statement_type="balance_sheet"),
            "income_statement": ctx.find("get_financial_statement", statement_type="income_statement"),
        }

    def execute_step(self, step: PlannedStep, ctx: RunContext) -> Any:
        if step.tool == "compute_fluctuations":
            return _compute_fluctuations(self._statements(ctx))
        if step.tool == "compute_ratios":
            return _compute_ratios(self._statements(ctx))
        raise ValueError(f"analytical_review: unknown compute step '{step.tool}'")

    def synthesize(self, ctx: RunContext) -> Dict[str, Any]:
        flux_out = ctx.find("compute_fluctuations") or {}
        ratio_out = ctx.find("compute_ratios") or {}
        fluctuations = flux_out.get("fluctuations", []) if isinstance(flux_out, dict) else []
        ratios = ratio_out.get("ratios", []) if isinstance(ratio_out, dict) else []
        gaps = list(flux_out.get("gaps", []) if isinstance(flux_out, dict) else [])
        if flux_out.get("capped"):
            gaps.append(f"only the {len(fluctuations)} largest fluctuations are shown for commentary")

        # Index-tagged, numbers-final payload. The model annotates by index; it never
        # supplies a figure we keep.
        payload = {
            "entity": ctx.find("get_audit_file_summary"),
            "significant_fluctuations": [
                {"index": i, "statement": f["statement"], "line": f["line"],
                 "cy": f["cy"], "py": f["py"], "delta": f["delta"], "pct": f["pct"], "direction": f["direction"]}
                for i, f in enumerate(fluctuations)
            ],
            "key_ratios": [
                {"index": i, "ratio": r["line"], "cy": r["cy"], "py": r["py"], "delta": r["delta"]}
                for i, r in enumerate(ratios)
            ],
            "thresholds": {"pct": flux_out.get("threshold_pct"), "value_floor": flux_out.get("abs_floor")},
            "lines_analyzed": flux_out.get("lines_analyzed"),
        }
        prompt = (
            "You are performing analytical review (ISA 520) on one audit file. The year-on-year "
            "fluctuations and key ratios below were ALREADY COMPUTED in code from the financial "
            "statements — every number is fixed.\n\n"
            f"DATA (JSON):\n{json.dumps(payload, default=str)[:90000]}\n\n"
            "Produce:\n"
            "- summary: 2-4 sentences on how the entity moved year on year overall.\n"
            "- conclusion: 1-2 sentences on whether results look consistent/expected or warrant further audit work.\n"
            "- fluctuation_notes: for EACH significant fluctuation (reference it by its `index`), a plausible "
            "business explanation/expectation for the movement, and `corroboration` = the specific evidence the "
            "auditor should obtain to confirm it.\n"
            "- ratio_notes: for EACH key ratio (by `index`), a one-line read on the trend.\n"
            "- follow_up: concrete items the auditor should corroborate or investigate.\n"
            "Explain the movements; do NOT restate the raw numbers as if you computed them, and never invent a figure."
        )
        system = (
            "You are a senior auditor performing analytical procedures. Use ONLY the figures provided (they were "
            "computed in code). Never compute, estimate, or invent a number. Give business-sensible explanations "
            "and concrete corroboration steps; be concise and specific."
        )
        draft = generate_structured(prompt, AnalyticalDraft, system=system, usage_out=ctx.usage_out)

        fnotes = {n.index: n for n in draft.fluctuation_notes}
        rnotes = {n.index: n for n in draft.ratio_notes}
        flux_final = [
            {**f, "reason": (fnotes[i].explanation if i in fnotes else ""),
             "corroboration": (fnotes[i].corroboration if i in fnotes else "")}
            for i, f in enumerate(fluctuations)
        ]
        ratio_final = [
            {**r, "reason": (rnotes[i].commentary if i in rnotes else "")}
            for i, r in enumerate(ratios)
        ]
        return {
            "summary": draft.summary,
            "conclusion": draft.conclusion,
            "lines_analyzed": flux_out.get("lines_analyzed", 0),
            "fluctuations_reviewed": flux_out.get("significant_count", 0),
            "significant_fluctuations": flux_final,
            "key_ratios": ratio_final,
            "follow_up": list(draft.follow_up or []),
            "data_gaps": gaps,
        }


register_definition(AnalyticalReviewAgent())
