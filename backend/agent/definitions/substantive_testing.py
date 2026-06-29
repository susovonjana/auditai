"""
Substantive Testing Review agent (agent_type = "substantive_testing").

Goal: "Review the substantive testing on this file and conclude." It evaluates the
sampling/testing that has been done, drafts a per-sample testing conclusion, and —
at an APPROVAL checkpoint — writes those conclusions back into the file (the same
``aud_sample_test_conclusions`` rows the substantive-testing screen shows). It does
NOT select samples or record per-item results (those are a later refinement).

Safe write policy: a conclusion is drafted ONLY for a sample that has been tested
AND whose test does not already have a conclusion — the agent FILLS gaps, it never
silently overwrites an auditor's own conclusion. Nothing is saved until the auditor
approves the checkpoint.

Numbers discipline: every quantitative judgement — coverage %, misstatement vs the
trivial threshold, projected misstatement vs performance materiality, and the
aggregate projected misstatement vs overall materiality (ISA 450) — is computed in
PURE PYTHON in ``_evaluate_testing``. The LLM is handed the finished figures and
only writes prose (per-sample notes + the overall conclusion), merged back by index;
the written conclusion text embeds the CODE-built figures, so a hallucinated number
can never reach the file.
"""
from __future__ import annotations

import html
import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from agent.types import PlannedStep, RunContext, register_definition
from agent.definitions.file_review import _num
from structured import generate_structured


# ---------------------------------------------------------------------------
# LLM output schema — PROSE ONLY (figures are merged in afterwards by index)
# ---------------------------------------------------------------------------
class SampleNote(BaseModel):
    index: int = Field(description="the `index` of the sample this note refers to")
    note: str = Field(default="", description="a concise conclusion of THIS sample's testing and the recommended next action — it is written into the file's conclusion field, so make it self-contained")


class TestingDraft(BaseModel):
    summary: str = Field(description="2-4 sentence overview of the testing performed and where it stands")
    conclusion: str = Field(description="ISA 330/450 conclusion: is the testing sufficient and are misstatements below materiality?")
    sample_notes: List[SampleNote] = Field(default_factory=list)
    needs_attention: List[str] = Field(default_factory=list, description="accounts/samples needing more testing or follow-up")


# ---------------------------------------------------------------------------
# Deterministic evaluation (no LLM, no invented numbers) — module-level for testing
# ---------------------------------------------------------------------------
def _fmt(n: Optional[float]) -> str:
    if n is None:
        return "—"
    return f"{n:,.0f}" if abs(n) >= 1000 else f"{n:,.2f}".rstrip("0").rstrip(".")


def _evaluate_testing(sampling: Any) -> Dict[str, Any]:
    """Turn the sampling design into a per-sample testing evaluation plus the
    file-level ISA 450 roll-up. Robust to missing/zero fields (test files have
    many)."""
    samples = sampling.get("samples", []) if isinstance(sampling, dict) else []
    results: List[Dict[str, Any]] = []
    agg_projected = 0.0
    file_overall_mat = 0.0
    tested = 0
    not_started = 0

    for s in samples:
        if not isinstance(s, dict):
            continue
        tested_items = _num(s.get("no_of_tested_items")) or 0.0
        pop_items = _num(s.get("no_of_items")) or 0.0
        found = _num(s.get("misstatement_found")) or 0.0
        trivial = _num(s.get("misstatement_trivial")) or 0.0
        projected = _num(s.get("projected_misstatement")) or 0.0
        perf_mat = _num(s.get("performance_materiality")) or 0.0
        overall_mat = _num(s.get("overall_materiality")) or 0.0

        agg_projected += projected
        file_overall_mat = max(file_overall_mat, overall_mat)

        if tested_items <= 0:
            status = "not_started"
            not_started += 1
        elif pop_items and tested_items >= pop_items:
            status = "complete"
            tested += 1
        elif pop_items and tested_items < pop_items:
            status = "in_progress"
            tested += 1
        else:  # items recorded but population size unknown
            status = "tested"
            tested += 1

        coverage = round(tested_items / pop_items * 100, 1) if pop_items else None
        flags: List[str] = []
        if status == "not_started":
            flags.append("not yet tested")
        if found > 0 and trivial and found > trivial:
            flags.append("misstatement above the trivial threshold")
        if projected > 0 and perf_mat and projected > perf_mat:
            flags.append("projected misstatement exceeds performance materiality")

        results.append({
            "sample_id": s.get("sample_id"),
            "reference": s.get("reference"),
            "account_name": s.get("account_name") or s.get("account_code") or "(unnamed)",
            "account_code": s.get("account_code"),
            "tests": list(s.get("tests") or []),
            "concluded_tests": list(s.get("concluded_tests") or []),
            "status": status,
            "tested_items": int(tested_items),
            "population_items": int(pop_items),
            "coverage_pct": coverage,
            "misstatement_found": round(found, 2),
            "misstatement_trivial": round(trivial, 2),
            "projected_misstatement": round(projected, 2),
            "performance_materiality": round(perf_mat, 2),
            "flags": flags,
        })

    return {
        "samples": results,
        "total": len(results),
        "tested": tested,
        "not_started": not_started,
        "attention_count": sum(1 for r in results if r["flags"]),
        "aggregate_projected_misstatement": round(agg_projected, 2),
        "file_overall_materiality": round(file_overall_mat, 2),
        "aggregate_within_materiality": (agg_projected <= file_overall_mat) if file_overall_mat else None,
    }


def _metrics_line(r: Dict[str, Any]) -> str:
    """A compact, code-built figure string for the UI (numbers never come from the LLM)."""
    parts: List[str] = []
    if r["tested_items"] or r["population_items"]:
        cover = f"tested {r['tested_items']}"
        if r["population_items"]:
            cover += f"/{r['population_items']}"
        if r["coverage_pct"] is not None:
            cover += f" ({r['coverage_pct']}%)"
        parts.append(cover)
    parts.append(f"misstatement {_fmt(r['misstatement_found'])} vs trivial {_fmt(r['misstatement_trivial'])}")
    if r["performance_materiality"]:
        parts.append(f"projected {_fmt(r['projected_misstatement'])} vs PM {_fmt(r['performance_materiality'])}")
    return " · ".join(parts)


def _conclusion_html(r: Dict[str, Any], note: str) -> str:
    """Build the conclusion text written into the file. The figures and the
    pass/attention verdict are CODE-OWNED; only the auditor-facing ``note`` is LLM
    prose (HTML-escaped). Stored as simple HTML so the TipTap conclusion editor
    renders it cleanly."""
    flags = r.get("flags") or []
    if not flags:
        verdict = "No exceptions above the trivial threshold were noted; substantive testing for this sample is complete."
    else:
        verdict = "Attention required — " + "; ".join(flags) + "."
    note = (note or "").strip()
    parts = [
        f"<p>Substantive testing — {html.escape(_metrics_line(r))}.</p>",
        f"<p>{html.escape(verdict)}</p>",
    ]
    if note:
        parts.append(f"<p>{html.escape(note)}</p>")
    parts.append("<p><em>Drafted with 1audit AI — review before sign-off.</em></p>")
    return "".join(parts)


def _build_conclusion_writes(samples: List[Dict[str, Any]], notes: Dict[int, str]) -> Dict[str, Any]:
    """Turn the per-sample evaluation + LLM notes into the write payload, applying
    the safe policy: only TESTED samples, only tests that have NO existing
    conclusion. Returns {conclusions (machine), preview (human), skipped (human)}."""
    items: List[Dict[str, Any]] = []
    preview: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for i, r in enumerate(samples):
        ref = r.get("reference")
        label = r.get("account_name") or ref or f"sample {i + 1}"
        disp = f"{ref} · {label}" if ref else label

        if r.get("status") == "not_started":
            skipped.append({"sample": disp, "reason": "not yet tested"})
            continue
        sid = r.get("sample_id")
        if sid is None:
            skipped.append({"sample": disp, "reason": "no sample id from source"})
            continue
        tests = r.get("tests") or []
        if not tests:
            skipped.append({"sample": disp, "reason": "no test method set up"})
            continue
        concluded = set(r.get("concluded_tests") or [])
        targets = [t for t in tests if t not in concluded]
        if not targets:
            skipped.append({"sample": disp, "reason": "already concluded"})
            continue

        comment = _conclusion_html(r, notes.get(i, ""))
        for t in targets:
            items.append({"sample_id": sid, "test_type_name": t, "conclusion_comment": comment})
        preview.append({
            "sample": disp,
            "tests": targets,
            "status": r.get("status"),
            "metrics": _metrics_line(r),
        })
    return {"conclusions": items, "preview": preview, "skipped": skipped}


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------
class SubstantiveTestingAgent:
    agent_type = "substantive_testing"
    allowed_tools = [
        "get_audit_file_summary",
        "get_materiality",
        "get_sampling_design",
        "get_procedure_results",
        "get_risks",
        "save_test_conclusions",
    ]

    def default_goal(self, audit_file_id: int) -> str:
        return f"Review substantive testing on audit file {audit_file_id} and conclude."

    def build_plan(self, ctx: RunContext) -> List[PlannedStep]:
        return [
            PlannedStep("Read file summary", "read", "get_audit_file_summary"),
            PlannedStep("Read materiality", "read", "get_materiality"),
            PlannedStep("Read sampling design", "read", "get_sampling_design"),
            PlannedStep("Read procedure results", "read", "get_procedure_results"),
            PlannedStep("Read risks", "read", "get_risks"),
            PlannedStep("Evaluate testing & misstatements", "compute", "evaluate_testing"),
            PlannedStep("Draft testing conclusions (AI)", "analysis", "draft_testing"),
            PlannedStep("Save drafted conclusions to the file", "write", "save_test_conclusions", requires_approval=True),
        ]

    def execute_step(self, step: PlannedStep, ctx: RunContext) -> Any:
        if step.tool == "evaluate_testing":
            return _evaluate_testing(ctx.find("get_sampling_design"))
        if step.tool == "draft_testing":
            return self._draft_testing(ctx)
        raise ValueError(f"substantive_testing: unknown compute step '{step.tool}'")

    # ---- write checkpoint: build the preview the auditor approves -------------
    def prepare_write(self, step: PlannedStep, ctx: RunContext) -> Optional[dict]:
        if step.tool == "save_test_conclusions":
            ev = ctx.find("evaluate_testing") or {}
            samples = ev.get("samples", []) if isinstance(ev, dict) else []
            draft = ctx.find("draft_testing") or {}
            notes_raw = draft.get("notes", {}) if isinstance(draft, dict) else {}
            notes = {int(k): v for k, v in notes_raw.items()}
            return _build_conclusion_writes(samples, notes)
        return None

    # ---- analysis: the single LLM pass (prose only) --------------------------
    def _draft_testing(self, ctx: RunContext) -> Dict[str, Any]:
        ev = ctx.find("evaluate_testing") or {}
        samples = ev.get("samples", []) if isinstance(ev, dict) else []
        payload = {
            "entity": ctx.find("get_audit_file_summary"),
            "materiality": ctx.find("get_materiality"),
            "procedure_results": ctx.find("get_procedure_results"),
            "risks": ctx.find("get_risks"),
            "evaluation": {
                "aggregate_projected_misstatement": ev.get("aggregate_projected_misstatement"),
                "file_overall_materiality": ev.get("file_overall_materiality"),
                "aggregate_within_materiality": ev.get("aggregate_within_materiality"),
                "samples_total": ev.get("total"), "samples_tested": ev.get("tested"),
                "samples_not_started": ev.get("not_started"),
                "samples": [
                    {"index": i, "account": r["account_name"], "status": r["status"],
                     "coverage_pct": r["coverage_pct"], "misstatement_found": r["misstatement_found"],
                     "misstatement_trivial": r["misstatement_trivial"],
                     "projected_misstatement": r["projected_misstatement"],
                     "performance_materiality": r["performance_materiality"], "flags": r["flags"]}
                    for i, r in enumerate(samples)
                ],
            },
        }
        prompt = (
            "You are reviewing the SUBSTANTIVE TESTING on one audit file and forming a conclusion. The per-sample "
            "evaluation below (status, coverage, misstatement vs the trivial threshold, projected misstatement vs "
            "performance materiality) and the aggregate roll-up were ALREADY COMPUTED in code — every number is fixed.\n\n"
            f"DATA (JSON):\n{json.dumps(payload, default=str)[:90000]}\n\n"
            "Produce:\n"
            "- summary: 2-4 sentences on what testing has been done and where it stands.\n"
            "- conclusion: an ISA 330/450 conclusion — is the testing sufficient, and is the aggregate projected "
            "misstatement below overall materiality? If samples are untested or a projected misstatement exceeds "
            "performance materiality, say the testing is not yet complete.\n"
            "- sample_notes: for EACH sample (reference it by its `index`), a CONCISE conclusion of that sample's "
            "testing and the recommended next action. This note is written into the file's conclusion field, so make "
            "it self-contained and professional (one or two sentences).\n"
            "- needs_attention: accounts/samples that still need testing or follow-up.\n"
            "Explain the figures; never invent or recompute one."
        )
        system = (
            "You are a senior auditor evaluating substantive testing. Use ONLY the figures provided (computed in code). "
            "Never compute, estimate, or invent a number. Be concise, specific, and conclusion-oriented."
        )
        draft = generate_structured(prompt, TestingDraft, system=system, usage_out=ctx.usage_out)
        return {
            "summary": draft.summary,
            "conclusion": draft.conclusion,
            "notes": {n.index: n.note for n in draft.sample_notes},
            "needs_attention": list(draft.needs_attention or []),
        }

    # ---- final structured result (no LLM call — reuses the draft) ------------
    def synthesize(self, ctx: RunContext) -> Dict[str, Any]:
        ev = ctx.find("evaluate_testing") or {}
        samples = ev.get("samples", []) if isinstance(ev, dict) else []
        draft = ctx.find("draft_testing") or {}
        notes_raw = draft.get("notes", {}) if isinstance(draft, dict) else {}
        notes = {int(k): v for k, v in notes_raw.items()}
        write_res = ctx.find("save_test_conclusions") or {}
        conclusions_written = write_res.get("written", 0) if isinstance(write_res, dict) else 0

        gaps: List[str] = []
        if not samples:
            gaps.append("no sampling design / samples found for this file")

        sample_results = [
            {
                "line": (f"{r['account_name']} ({r['account_code']})" if r.get("account_code") else r["account_name"]),
                "status": r["status"],
                "metrics": _metrics_line(r),
                "reason": notes.get(i, ""),
            }
            for i, r in enumerate(samples)
        ]
        return {
            "summary": draft.get("summary", "") if isinstance(draft, dict) else "",
            "conclusion": draft.get("conclusion", "") if isinstance(draft, dict) else "",
            "samples_total": ev.get("total", 0),
            "samples_tested": ev.get("tested", 0),
            "untested_samples": ev.get("not_started", 0),
            "conclusions_written": conclusions_written,
            "sample_results": sample_results,
            "needs_attention": draft.get("needs_attention", []) if isinstance(draft, dict) else [],
            "data_gaps": gaps,
        }


register_definition(SubstantiveTestingAgent())
