"""
Engagement Build-Out agent (agent_type = "engagement_buildout").

Goal: after a TB import, run the post-import setup in one supervised pass:
  read TB -> AI-map accounts (Feature C) -> [APPROVE] persist mappings ->
  [APPROVE] create lead schedules -> read financials (on-read) ->
  variance (pure python) -> materiality recommendation -> build-out summary.

Numbers discipline: mappings come from the existing engine (be /trial_balance/ai_map),
variances are computed in code, the materiality FIGURE stays in 1audit-be — the LLM
only RECOMMENDS benchmark options and writes prose. Three write checkpoints; nothing
is saved until the auditor approves.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from agent.types import PlannedStep, RunContext, register_definition
from agent.definitions.file_review import _extract_lines  # shared, tolerant FS parser
from structured import generate_structured


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class MaterialityOption(BaseModel):
    basis: str = Field(description="benchmark, e.g. Revenue / Total assets / Profit before tax")
    percent_low: Optional[float] = None
    percent_high: Optional[float] = None
    rationale: str = ""


class MaterialitySuggestion(BaseModel):
    options: List[MaterialityOption] = Field(default_factory=list)
    note: str = ""


class BuildOutNarrative(BaseModel):
    summary: str = Field(description="2-4 sentences on what was set up and what to check")
    needs_attention: List[str] = Field(default_factory=list)


_LARGE_MOVE_PCT = 30.0


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------
class EngagementBuildOutAgent:
    agent_type = "engagement_buildout"
    allowed_tools = [
        "get_audit_file_summary",
        "get_trial_balance",
        "get_financial_statement",
        "persist_tb_mappings",
        "create_lead_schedules",
    ]

    def default_goal(self, audit_file_id: int) -> str:
        return f"Build out audit file {audit_file_id} after trial-balance import."

    def build_plan(self, ctx: RunContext) -> List[PlannedStep]:
        return [
            PlannedStep("Read file summary", "read", "get_audit_file_summary"),
            PlannedStep("Read trial balance", "read", "get_trial_balance"),
            PlannedStep("Propose account mappings (AI)", "analysis", "propose_mappings"),
            PlannedStep("Persist approved mappings", "write", "persist_tb_mappings", requires_approval=True),
            PlannedStep("Create lead schedules", "write", "create_lead_schedules", requires_approval=True),
            PlannedStep("Read balance sheet", "read", "get_financial_statement", {"statement_type": "balance_sheet"}),
            PlannedStep("Read income statement", "read", "get_financial_statement", {"statement_type": "income_statement"}),
            PlannedStep("Compute period movements", "compute", "compute_variance"),
            PlannedStep("Recommend materiality benchmark (AI)", "analysis", "suggest_materiality"),
        ]

    def execute_step(self, step: PlannedStep, ctx: RunContext) -> Any:
        if step.tool == "propose_mappings":
            return self._propose_mappings(ctx)
        if step.tool == "compute_variance":
            return self._compute_variance(ctx)
        if step.tool == "suggest_materiality":
            return self._suggest_materiality(ctx)
        raise ValueError(f"engagement_buildout: unknown step '{step.tool}'")

    # ---- write checkpoints: build the preview the auditor approves ----------
    def prepare_write(self, step: PlannedStep, ctx: RunContext) -> Optional[dict]:
        # proposed_write carries a human-readable `preview` alongside the machine
        # payload; 1audit-be ignores `preview` (it reads only mappings / tb_account_ids).
        proposed = ctx.find("propose_mappings") or {}
        mappings = proposed.get("mappings", []) if isinstance(proposed, dict) else []
        if step.tool == "persist_tb_mappings":
            return {"mappings": mappings, "preview": proposed.get("preview", [])}
        if step.tool == "create_lead_schedules":
            ids = sorted({int(x) for g in mappings for x in g.get("tb_account_ids", [])})
            names = proposed.get("accounts_by_id", {}) if isinstance(proposed, dict) else {}
            return {"tb_account_ids": ids, "preview": [names.get(str(i), f"#{i}") for i in ids]}
        return None

    # ---- analysis / compute (no LLM numbers) --------------------------------
    def _propose_mappings(self, ctx: RunContext) -> Dict[str, Any]:
        # Reuse Feature C end-to-end (be gathers context + runs the engine).
        res = ctx.copilot.post("trial_balance/ai_map", {"language": ctx.language})
        if not isinstance(res, dict) or "error" in res:
            note = res.get("error") if isinstance(res, dict) else "ai_map unavailable"
            return {"mappings": [], "preview": [], "accounts_by_id": {}, "suggestion_count": 0, "mappable_count": 0, "note": str(note)}
        suggestions = res.get("suggestions", []) or []
        # Name lookups so the approval preview reads in plain English, not ids.
        acct = {int(a["tb_account_id"]): a for a in res.get("accounts_lookup", []) if a.get("tb_account_id") is not None}
        coa = {int(c["coa_original_id"]): (c.get("label") or "") for c in res.get("coa_lookup", []) if c.get("coa_original_id") is not None}

        def label_for(tb: int) -> str:
            a = acct.get(int(tb), {})
            name = a.get("account_name") or f"#{tb}"
            code = a.get("account_code")
            return f"{name} ({code})" if code else name

        groups: Dict[int, Dict[str, Any]] = {}
        for s in suggestions:
            c = s.get("coa_original_id")
            tb = s.get("trial_balance_account_id")
            if c is None or tb is None:
                continue
            g = groups.setdefault(int(c), {"coa_original_id": int(c), "tb_account_ids": [], "confs": []})
            g["tb_account_ids"].append(int(tb))
            if s.get("confidence") is not None:
                g["confs"].append(float(s["confidence"]))
        mappings: List[Dict[str, Any]] = []
        preview: List[Dict[str, Any]] = []
        for g in groups.values():
            conf = round(sum(g["confs"]) / len(g["confs"]), 3) if g["confs"] else None
            mappings.append({
                "tb_account_ids": g["tb_account_ids"], "coa_original_id": g["coa_original_id"],
                "mapping_source": "ai", "mapping_confidence": conf,
            })
            preview.append({
                "category": coa.get(g["coa_original_id"]) or f"COA {g['coa_original_id']}",
                "accounts": [label_for(tb) for tb in g["tb_account_ids"]],
                "confidence": round(conf * 100) if conf is not None else None,
            })
        return {
            "mappings": mappings,
            "preview": preview,
            "accounts_by_id": {str(tb): label_for(tb) for tb in acct},
            "suggestion_count": len(suggestions),
            "mappable_count": sum(len(m["tb_account_ids"]) for m in mappings),
            "tier_counts": res.get("tier_counts", {}),
        }

    def _compute_variance(self, ctx: RunContext) -> Dict[str, Any]:
        movements: List[Dict[str, Any]] = []
        gaps: List[str] = []
        parsed = 0
        for label in ("balance_sheet", "income_statement"):
            data = ctx.find("get_financial_statement", statement_type=label)
            if data is None or (isinstance(data, dict) and "error" in data):
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
                if (pct is not None and abs(pct) >= _LARGE_MOVE_PCT) or (py == 0 and cy != 0):
                    movements.append({
                        "statement": label, "line": r["name"],
                        "cy": round(cy, 2), "py": round(py, 2), "delta": round(delta, 2),
                        "pct": (round(pct, 1) if pct is not None else None),
                    })
        return {"movements": movements[:50], "lines_parsed": parsed, "gaps": gaps}

    def _suggest_materiality(self, ctx: RunContext) -> Dict[str, Any]:
        payload = {
            "file_summary": ctx.find("get_audit_file_summary"),
            "balance_sheet": ctx.find("get_financial_statement", statement_type="balance_sheet"),
            "income_statement": ctx.find("get_financial_statement", statement_type="income_statement"),
        }
        prompt = (
            "Recommend overall materiality BENCHMARK options for this engagement, based on the "
            "entity profile and financials below. For each option give the basis (Revenue / Total "
            "assets / Profit before tax), a typical percentage band, and a one-line rationale.\n\n"
            f"DATA (JSON):\n{json.dumps(payload, default=str)[:60000]}\n\n"
            "Return 2-3 options. RECOMMEND ONLY — do not compute the materiality figure (1audit "
            "calculates it from the chosen basis)."
        )
        system = (
            "You are a senior auditor's assistant. Recommend materiality benchmark options grounded "
            "in the provided figures; never compute or assert a final materiality amount. Use ONLY the "
            "values provided."
        )
        out = generate_structured(prompt, MaterialitySuggestion, system=system, usage_out=ctx.usage_out)
        return out.model_dump(mode="json")

    # ---- final structured result -------------------------------------------
    def synthesize(self, ctx: RunContext) -> Dict[str, Any]:
        persist = ctx.find("persist_tb_mappings") or {}
        leads = ctx.find("create_lead_schedules") or {}
        variance = ctx.find("compute_variance") or {}
        materiality = ctx.find("suggest_materiality") or {}
        propose = ctx.find("propose_mappings") or {}

        accounts_mapped = persist.get("updated_count", 0) if isinstance(persist, dict) else 0
        leads_created = leads.get("created", 0) if isinstance(leads, dict) else 0
        movements = variance.get("movements", []) if isinstance(variance, dict) else []
        options = materiality.get("options", []) if isinstance(materiality, dict) else []
        gaps = variance.get("gaps", []) if isinstance(variance, dict) else []

        facts = {
            "accounts_mapped": accounts_mapped,
            "accounts_proposed": propose.get("mappable_count") if isinstance(propose, dict) else None,
            "lead_schedules_created": leads_created,
            "large_movements": movements,
            "materiality_options": options,
        }
        prose = generate_structured(
            "Write a short build-out summary for the auditor from these FACTS (already computed; "
            "do not change any number):\n" + json.dumps(facts, default=str)[:40000] +
            "\nSay what was set up (mappings, lead schedules), call out the notable period movements, "
            "and note the materiality recommendation. Then list concrete 'needs attention' items.",
            BuildOutNarrative,
            system="You are a senior auditor's assistant. Use ONLY the provided facts; never invent or "
                   "recompute a figure. Be concise and specific.",
            usage_out=ctx.usage_out,
        )
        return {
            "summary": prose.summary,
            "accounts_mapped": accounts_mapped,
            "lead_schedules_created": leads_created,
            "large_movements": movements,
            "materiality_recommendation": options,
            "needs_attention": prose.needs_attention,
            "data_gaps": gaps,
        }


register_definition(EngagementBuildOutAgent())
