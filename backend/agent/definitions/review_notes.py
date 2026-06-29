"""
Review Notes / EQR agent (agent_type = "review_notes").

Goal: "Review this file as a manager / engagement quality reviewer and raise
review notes." Read-only — it surfaces review notes for the preparer to clear;
it writes nothing, so it needs NO 1audit-be change. (A v1.1 that WRITES the notes
back as review points would need a new grant-scoped endpoint — deferred.)

How it differs from File Review: File Review is a pre-sign-off gap scan; this is
the reviewer's critical lens — it reads the EXISTING review points (clearing the
ones already resolved, following up the open ones) and raises NEW notes on the
quality / sufficiency of the work, each tied to a working paper with a severity
and a required action.

Numbers discipline: the work here is qualitative (review judgement). The only
figures are counts (working papers signed off, review points open/cleared),
computed in PURE PYTHON in ``_review_signals``; the LLM never invents a number.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from agent.types import PlannedStep, RunContext, register_definition
from structured import generate_structured


_SIGNED = ("completed", "signed_off", "signed off", "reviewed")
_NOT_STARTED = ("not_started", "not started", "")


# ---------------------------------------------------------------------------
# Structured result schema
# ---------------------------------------------------------------------------
class ReviewNote(BaseModel):
    working_paper: str = Field(description="the working paper / area the note is about")
    severity: str = Field(description="High / Medium / Low")
    note: str = Field(description="the reviewer's point — what is missing, unclear, or unsupported")
    action_required: str = Field(default="", description="what the preparer must do to clear it")


class EqrReview(BaseModel):
    summary: str = Field(description="2-4 sentence overall review opinion on the file's readiness")
    conclusion: str = Field(description="1-2 sentences: is the file ready for sign-off, or what must clear first?")
    new_review_notes: List[ReviewNote] = Field(default_factory=list)
    open_points_followup: List[str] = Field(default_factory=list, description="follow-up on existing UNRESOLVED review points")


# ---------------------------------------------------------------------------
# Deterministic signals (no LLM, no invented numbers) — module-level for testing
# ---------------------------------------------------------------------------
def _status(w: Dict[str, Any]) -> str:
    return str(w.get("status") or "").strip().lower()


def _review_signals(working_papers: Any, review_points: Any) -> Dict[str, Any]:
    """WP sign-off stats + existing review-point open/cleared split. Grounds the
    reviewer so it follows up the right open points and doesn't re-raise cleared ones."""
    wlist = working_papers.get("working_papers", []) if isinstance(working_papers, dict) else []
    rlist = review_points.get("review_points", []) if isinstance(review_points, dict) else []
    wlist = [w for w in wlist if isinstance(w, dict)]
    rlist = [r for r in rlist if isinstance(r, dict)]

    signed = sum(1 for w in wlist if _status(w) in _SIGNED)
    not_started = sum(1 for w in wlist if _status(w) in _NOT_STARTED)
    in_progress = len(wlist) - signed - not_started
    unsigned = [
        {"name": w.get("name"), "reference": w.get("reference"), "status": w.get("status"), "section": w.get("section")}
        for w in wlist if _status(w) not in _SIGNED
    ]

    open_points = [r for r in rlist if not r.get("resolved")]
    cleared = len(rlist) - len(open_points)
    open_list = [
        {"number": r.get("number"), "comment": r.get("comment"),
         "working_paper": r.get("working_paper"), "reviewed": bool(r.get("reviewed"))}
        for r in open_points
    ]
    return {
        "wp_total": len(wlist), "wp_signed": signed, "wp_in_progress": in_progress, "wp_not_started": not_started,
        "unsigned_working_papers": unsigned,
        "review_points_total": len(rlist), "review_points_open": len(open_points), "review_points_cleared": cleared,
        "open_review_points": open_list,
    }


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------
class ReviewNotesAgent:
    agent_type = "review_notes"
    allowed_tools = [
        "get_audit_file_summary",
        "list_working_papers",
        "get_review_points",
        "get_risks",
        "get_procedure_results",
    ]

    def default_goal(self, audit_file_id: int) -> str:
        return f"Review audit file {audit_file_id} as an EQR and raise review notes."

    def build_plan(self, ctx: RunContext) -> List[PlannedStep]:
        return [
            PlannedStep("Read file summary", "read", "get_audit_file_summary"),
            PlannedStep("List working papers", "read", "list_working_papers"),
            PlannedStep("Read existing review points", "read", "get_review_points"),
            PlannedStep("Read risks", "read", "get_risks"),
            PlannedStep("Read procedure results", "read", "get_procedure_results"),
            PlannedStep("Assess review coverage", "compute", "review_signals"),
        ]

    def execute_step(self, step: PlannedStep, ctx: RunContext) -> Any:
        if step.tool == "review_signals":
            return _review_signals(ctx.find("list_working_papers"), ctx.find("get_review_points"))
        raise ValueError(f"review_notes: unknown compute step '{step.tool}'")

    def synthesize(self, ctx: RunContext) -> Dict[str, Any]:
        sig = ctx.find("review_signals") or {}
        gaps: List[str] = []
        if not sig.get("wp_total"):
            gaps.append("no working papers found for this file")

        payload = {
            "entity": ctx.find("get_audit_file_summary"),
            "signals": sig,
            "risks": ctx.find("get_risks"),
            "procedure_results": ctx.find("get_procedure_results"),
        }
        prompt = (
            "You are the engagement quality reviewer (EQR) for this audit file. The working-paper sign-off counts, "
            "the list of unsigned working papers, and the EXISTING review points (split into open vs cleared) below "
            "were ALREADY COMPUTED in code — treat the counts as fixed.\n\n"
            f"DATA (JSON):\n{json.dumps(payload, default=str)[:90000]}\n\n"
            "Produce:\n"
            "- summary: 2-4 sentences on the file's overall readiness for sign-off.\n"
            "- conclusion: is the file ready, or what must be cleared first?\n"
            "- new_review_notes: NEW review points you would raise. For EACH, give the `working_paper`/area, a "
            "`severity` (High/Medium/Low), the `note` (what is missing, unclear, or unsupported), and the "
            "`action_required` to clear it. Focus on unsigned working papers, risks without evident procedures, and "
            "weak/unsupported conclusions. Do NOT re-raise points already in the CLEARED existing review points.\n"
            "- open_points_followup: follow-up on each EXISTING UNRESOLVED review point (signals.open_review_points).\n"
            "Be specific and constructive; never invent a figure or a fact not in the data."
        )
        system = (
            "You are a senior engagement quality reviewer. Use ONLY the provided data; the counts were computed in code "
            "— never invent or recompute one. Raise precise, constructive review notes a preparer can act on."
        )
        review = generate_structured(prompt, EqrReview, system=system, usage_out=ctx.usage_out)

        notes_out = [
            {
                "title": n.working_paper,
                "status": n.severity,
                "reason": n.note,
                "action": (f"Action: {n.action_required}" if n.action_required else ""),
            }
            for n in review.new_review_notes
        ]
        open_existing = [
            (f"#{p['number']} ({p['working_paper']}): {p['comment']}" if p.get("working_paper") else f"#{p['number']}: {p['comment']}")
            for p in sig.get("open_review_points", [])
        ]
        return {
            "summary": review.summary,
            "conclusion": review.conclusion,
            "review_notes_raised": len(notes_out),
            "open_points": sig.get("review_points_open", 0),
            "working_papers_signed": sig.get("wp_signed", 0),
            "review_notes": notes_out,
            "open_review_points": (list(review.open_points_followup) or open_existing),
            "data_gaps": gaps,
        }


register_definition(ReviewNotesAgent())
