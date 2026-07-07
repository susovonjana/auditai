"""
Procedure Build-Out agent (agent_type = "procedure_buildout").

Goal: "Draft the audit program for ONE Program & Checklist working paper." It
reads the working paper's configuration + the file context (risks, audit plan,
area accounts, materiality), makes ONE structured LLM call that plans the whole
program the way an audit senior would — ordered titles, procedures with child
steps, a suggested response set + rationale per item — and pauses at an APPROVAL
checkpoint. Only after the auditor approves does the (possibly edited) tree get
written, by 1audit-be, in one transaction, every section stamped with this run's
id so the whole draft is undoable in one click.

Safe write policy: APPEND-ONLY. The agent never edits or deletes existing
sections; its draft is added below whatever the working paper already contains,
and the prompt shows the existing items so it doesn't duplicate them.

Structure discipline (mirror of substantive_testing's numbers discipline): the
LLM proposes prose and structure ONLY. Everything enforceable is validated or
built in PURE PYTHON — allowed response types, picklist option counts, depth and
size caps, the HTML tag whitelist, temp-id parenting, and the run-id stamp.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from agent.types import PlannedStep, RunContext, register_definition
from prompts.procedure_plan import SYSTEM_PROMPT, TEMPLATE_SYSTEM_SUFFIX, build_plan_prompt
from structured import generate_structured


# ---------------------------------------------------------------------------
# LLM output schema. Depth is bounded STRUCTURALLY (section -> step -> sub-step)
# instead of by a recursive model, so the tool-use JSON schema stays simple and
# the depth<=3 cap can't be violated by construction.
# ---------------------------------------------------------------------------
class ResponseSetProposal(BaseModel):
    type: Literal["text", "text_area", "date", "picklist", "multi_picklist"]
    placeholder: str = Field(default="", description="short label shown in the empty response box (PRIMARY language)")
    placeholder_sl: str = Field(default="", description="the placeholder in the SECONDARY language; empty on single-language files")
    options: List[str] = Field(default_factory=list, description="picklist/multi_picklist choices (2-6, short, PRIMARY language)")
    options_sl: List[str] = Field(default_factory=list, description="the same options in the SECONDARY language, aligned 1:1 with options; empty on single-language files")


class SubStepNode(BaseModel):
    procedure_html: str = Field(default="", description="restricted HTML: <p><ol><ul><li><strong><em> only (PRIMARY language)")
    procedure_html_sl: str = Field(default="", description="the same procedure in the SECONDARY language (same HTML rules); empty on single-language files")
    assertions: List[str] = Field(default_factory=list, description="assertion NAMES this step addresses (only when assertions are enabled)")
    response_sets: List[ResponseSetProposal] = Field(default_factory=list)
    rationale: str = Field(default="", description="one line: why this step / this response type")


class StepNode(SubStepNode):
    children: List[SubStepNode] = Field(default_factory=list, description="genuinely distinct sub-steps only")


class SectionNode(BaseModel):
    section_type: Literal["title", "procedure", "comment"]
    title: str = Field(default="", description="title sections: short plain-text heading (PRIMARY language)")
    title_sl: str = Field(default="", description="the heading in the SECONDARY language; empty on single-language files")
    procedure_html: str = Field(default="", description="procedure sections: restricted HTML; comment sections: the paragraph text (PRIMARY language)")
    procedure_html_sl: str = Field(default="", description="the same content in the SECONDARY language; empty on single-language files")
    assertions: List[str] = Field(default_factory=list)
    response_sets: List[ResponseSetProposal] = Field(default_factory=list)
    rationale: str = Field(default="")
    children: List[StepNode] = Field(
        default_factory=list,
        description="a title's procedures, or a procedure's sub-steps (never on comments)",
    )


class ProcedurePlan(BaseModel):
    config_summary: str = Field(description="2-3 plain-language sentences on the WP's setup, shown to the auditor first")
    nodes: List[SectionNode] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Pure-Python validation + flattening (no LLM) — module level for tests
# ---------------------------------------------------------------------------
_ALLOWED_RESPONSE_TYPES = {"text", "text_area", "date", "picklist", "multi_picklist"}
_ALLOWED_TAGS = {"p", "ol", "ul", "li", "strong", "em", "br"}
_MAX_TOP_LEVEL = 15
_MAX_DEPTH = 3
_MAX_SECTIONS_TOTAL = 120
_MAX_RESPONSE_SETS = 4
_MAX_PICKLIST_OPTIONS = 12
_MAX_HTML_CHARS = 20000

_TAG_RE = re.compile(r"</?([a-zA-Z][a-zA-Z0-9]*)\b[^>]*>")
_SCRIPT_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def sanitize_html(html: Optional[str]) -> str:
    """Keep only the procedure editor's tags, attribute-free (same whitelist the
    be endpoint re-applies — defense in depth on both sides of the wire)."""
    if not html:
        return ""
    s = str(html)[:_MAX_HTML_CHARS]
    s = _SCRIPT_RE.sub("", s)
    s = _COMMENT_RE.sub("", s)

    def _keep(match: re.Match) -> str:
        tag = match.group(1).lower()
        if tag not in _ALLOWED_TAGS:
            return ""
        if tag == "br":
            return "<br/>"
        return f"</{tag}>" if match.group(0).startswith("</") else f"<{tag}>"

    return _TAG_RE.sub(_keep, s).strip()


def _strip_tags(html: Optional[str]) -> str:
    text = re.sub(r"<[^>]*>", " ", str(html or "")).replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def _field(rs: Any, name: str, default: Any) -> Any:
    return getattr(rs, name, None) or (rs.get(name) if isinstance(rs, dict) else None) or default


def validate_response_sets(
    proposals: List[Any], notes: List[str], label: str, *, bilingual: bool = False
) -> List[Dict[str, Any]]:
    """Filter a node's proposed response sets down to what the product accepts.
    Invalid entries are DROPPED with a note (never silently mutated into
    something the model didn't propose). Secondary-language fields (placeholder_sl,
    options_sl) pass through only when the file is bilingual; options_sl pairs
    with options BY INDEX so a dropped option drops its translation."""
    out: List[Dict[str, Any]] = []
    for rs in proposals or []:
        rtype = _field(rs, "type", None)
        placeholder = _field(rs, "placeholder", "")
        placeholder_sl = _field(rs, "placeholder_sl", "") if bilingual else ""
        options = list(_field(rs, "options", []) or [])
        options_sl = list(_field(rs, "options_sl", []) or []) if bilingual else []
        if rtype not in _ALLOWED_RESPONSE_TYPES:
            notes.append(f"{label}: dropped response set with unknown type '{rtype}'")
            continue
        entry: Dict[str, Any] = {"type": rtype, "placeholder": _strip_tags(placeholder)[:255]}
        if bilingual and _strip_tags(placeholder_sl):
            entry["placeholder_sl"] = _strip_tags(placeholder_sl)[:255]
        if rtype in ("picklist", "multi_picklist"):
            pairs = []
            for i, o in enumerate(options):
                primary = _strip_tags(o)[:255]
                if not primary:
                    continue
                secondary = _strip_tags(options_sl[i])[:255] if i < len(options_sl) else ""
                pairs.append((primary, secondary))
            # de-dup on the primary text, keep order
            seen: set = set()
            pairs = [p for p in pairs if not (p[0].lower() in seen or seen.add(p[0].lower()))]
            if len(pairs) < 2:
                notes.append(f"{label}: dropped {rtype} with fewer than 2 options")
                continue
            pairs = pairs[:_MAX_PICKLIST_OPTIONS]
            entry["options"] = [p[0] for p in pairs]
            if bilingual and any(p[1] for p in pairs):
                entry["options_sl"] = [p[1] for p in pairs]
        out.append(entry)
        if len(out) >= _MAX_RESPONSE_SETS:
            break
    return out


def flatten_plan(
    nodes: List[SectionNode],
    *,
    assertions_enabled: bool,
    bilingual: bool = False,
) -> Dict[str, Any]:
    """Turn the LLM's tree into the be bulk_create payload rows (temp-id
    parenting, parents first) + a render-ready preview tree + validation notes.

    Flattening rules (mirror how the product actually stores a program):
      - titles never nest; a title's children become TOP-LEVEL procedures placed
        right after it (visual grouping is by order);
      - only procedure-under-procedure parenting is real (depth <= 3);
      - empty/invalid nodes are dropped with a note, never invented;
      - secondary-language (*_sl) content passes through only when ``bilingual``
        (single-language files strip any _sl the model emitted anyway).
    """
    notes: List[str] = []
    sections: List[Dict[str, Any]] = []
    preview: List[Dict[str, Any]] = []
    counter = {"n": 0}

    def next_id() -> str:
        counter["n"] += 1
        return f"t{counter['n']}"

    def sl_html(node: Any, attr: str) -> str:
        if not bilingual:
            return ""
        return sanitize_html(getattr(node, attr, "") or "")

    def emit_procedure(
        node: Any, parent_temp_id: Optional[str], depth: int, label: str,
    ) -> Optional[Dict[str, Any]]:
        if len(sections) >= _MAX_SECTIONS_TOTAL:
            return None
        html = sanitize_html(getattr(node, "procedure_html", "") or "")
        if not _strip_tags(html):
            notes.append(f"{label}: dropped procedure with empty text")
            return None
        html_sl = sl_html(node, "procedure_html_sl")
        temp_id = next_id()
        row: Dict[str, Any] = {
            "temp_id": temp_id,
            "parent_temp_id": parent_temp_id,
            "section_type": 2,
            "procedure": html,
        }
        if _strip_tags(html_sl):
            row["procedure_sl"] = html_sl
        rsets = validate_response_sets(
            list(getattr(node, "response_sets", []) or []), notes, label, bilingual=bilingual,
        )
        if rsets:
            row["response_sets"] = rsets
        if assertions_enabled:
            asserts = [_strip_tags(a)[:100] for a in (getattr(node, "assertions", []) or []) if _strip_tags(a)]
            if asserts:
                row["assertions"] = asserts[:10]
        sections.append(row)

        item = {
            "temp_id": temp_id,
            "kind": "procedure",
            "text": _strip_tags(html)[:400],
            "html": html,
            "html_sl": row.get("procedure_sl", ""),
            "response_sets": rsets,
            "rationale": _strip_tags(getattr(node, "rationale", ""))[:300],
            "assertions": row.get("assertions", []),
            "depth": depth,
            "children": [],
        }
        if depth < _MAX_DEPTH:
            for j, child in enumerate(list(getattr(node, "children", []) or [])):
                child_item = emit_procedure(child, temp_id, depth + 1, f"{label}.{j + 1}")
                if child_item:
                    item["children"].append(child_item)
        else:
            dropped = len(list(getattr(node, "children", []) or []))
            if dropped:
                notes.append(f"{label}: dropped {dropped} sub-step(s) below the depth-{_MAX_DEPTH} cap")
        return item

    top_nodes = list(nodes or [])
    if len(top_nodes) > _MAX_TOP_LEVEL:
        notes.append(f"kept the first {_MAX_TOP_LEVEL} of {len(top_nodes)} top-level groups")
        top_nodes = top_nodes[:_MAX_TOP_LEVEL]

    for i, node in enumerate(top_nodes):
        label = f"node {i + 1}"
        if node.section_type == "comment":
            # a free-text paragraph (e.g. the "Objective: …" opener) — always
            # top-level, never nests, carries no response sets
            html = sanitize_html(node.procedure_html or node.title)
            if not _strip_tags(html):
                notes.append(f"{label}: dropped comment with empty text")
                continue
            if len(sections) >= _MAX_SECTIONS_TOTAL:
                break
            html_sl = sl_html(node, "procedure_html_sl")
            temp_id = next_id()
            row = {"temp_id": temp_id, "parent_temp_id": None, "section_type": 3, "description": html}
            if _strip_tags(html_sl):
                row["description_sl"] = html_sl
            sections.append(row)
            preview.append({
                "temp_id": temp_id, "kind": "comment", "text": _strip_tags(html)[:400], "html": html,
                "html_sl": row.get("description_sl", ""),
                "rationale": _strip_tags(node.rationale)[:300], "depth": 1, "children": [],
            })
        elif node.section_type == "title":
            title = _strip_tags(node.title or node.procedure_html)[:500]
            if not title:
                notes.append(f"{label}: dropped title with empty text")
                continue
            if len(sections) >= _MAX_SECTIONS_TOTAL:
                break
            title_sl = _strip_tags(getattr(node, "title_sl", ""))[:500] if bilingual else ""
            temp_id = next_id()
            row = {"temp_id": temp_id, "parent_temp_id": None, "section_type": 1, "title": title}
            if title_sl:
                row["title_sl"] = title_sl
            sections.append(row)
            group = {
                "temp_id": temp_id, "kind": "title", "text": title, "text_sl": title_sl,
                "rationale": _strip_tags(node.rationale)[:300], "depth": 1, "children": [],
            }
            preview.append(group)
            # a title's children are that group's procedures — flattened to top
            # level right after it (titles group by ORDER, they never parent)
            for j, child in enumerate(list(node.children or [])):
                item = emit_procedure(child, None, 1, f"{label}.{j + 1}")
                if item:
                    group["children"].append(item)
        else:
            item = emit_procedure(node, None, 1, label)
            if item:
                preview.append(item)

    counts = {
        "total_sections": len(sections),
        "titles": sum(1 for s in sections if s["section_type"] == 1),
        "procedures": sum(1 for s in sections if s["section_type"] == 2),
        "comments": sum(1 for s in sections if s["section_type"] == 3),
        "with_response_sets": sum(1 for s in sections if s.get("response_sets")),
        "response_types": {},
    }
    for s in sections:
        for rs in s.get("response_sets", []):
            counts["response_types"][rs["type"]] = counts["response_types"].get(rs["type"], 0) + 1

    return {"sections": sections, "preview": preview, "notes": notes, "counts": counts}


def summarize_wp_config(wp_content: Any) -> Dict[str, Any]:
    """CODE-derived facts about the working paper's global config + current
    content. These ground both the LLM's config_summary and the validation
    (e.g. assertions stripped when disabled)."""
    wp = (wp_content or {}).get("working_paper", {}) if isinstance(wp_content, dict) else {}
    config = wp.get("config") or {}
    settings = config.get("settings") or {}
    default_sets = [
        {"type": rs.get("type"), "placeholder": rs.get("placeholder")}
        for rs in (settings.get("response_sets") or [])
        if isinstance(rs, dict)
    ]
    items = (wp_content or {}).get("items", []) if isinstance(wp_content, dict) else []
    existing_titles = [i.get("text") for i in items if isinstance(i, dict) and i.get("kind") == "title"][:20]
    # Org language setup (from be): content columns hold the PRIMARY language,
    # *_sl columns the SECONDARY — drafted only when the org uses one.
    langs = wp.get("languages") or {}
    primary = (langs.get("primary") or "en").lower()
    secondary = (langs.get("secondary") or "").lower() or None
    bilingual = bool(langs.get("use_secondary") and secondary and secondary != primary)
    return {
        "working_paper_name": wp.get("name"),
        "working_paper_type": wp.get("type"),
        "assertions_enabled": bool(settings.get("enable_assertions")),
        "signature_enabled": bool(settings.get("enable_signature")),
        "notes_enabled": bool(settings.get("enable_notes")),
        "attachments_enabled": bool(settings.get("enable_attachments")),
        "default_response_sets": default_sets,
        "existing_section_count": len(items),
        "existing_titles": existing_titles,
        "is_empty": len(items) == 0,
        "primary_language": primary,
        "secondary_language": secondary if bilingual else None,
        "bilingual": bilingual,
    }


# Bounds for the prior-program grounding block handed to the LLM (defense in
# depth over the be caps — the prompt has its own char budget on top).
_MAX_PRIOR_ITEMS = 60
_MAX_PRIOR_HTML = 1200


def compact_prior_program(res: Any, *, bilingual: bool) -> Optional[Dict[str, Any]]:
    """The prior_program_sources response -> the compact grounding dict handed to
    the prompt + provenance for the checkpoint. None when there is nothing to
    adapt (error / no source / empty items). Re-caps size, strips secondary-
    language fields when the draft is single-language, and builds the
    human-readable label per kind — prior_year "FY2025 A.1 Inventory (34
    sections)", current_year "… in this year's file '…'", structure "the
    structure of '…' in this file"; template runs add template_peer ("the firm
    template '…' — …") and org_practice ("… from engagement file '…'").
    Templates are never used for real-file runs. Pure."""
    if not isinstance(res, dict) or res.get("error"):
        return None
    source = res.get("source")
    if not isinstance(source, dict) or not source.get("items"):
        return None
    kind = source.get("kind") or "prior_year"
    wp = source.get("working_paper") or {}
    audit_file = source.get("audit_file") or {}
    section_count = int(source.get("section_count") or 0)

    emitted = 0
    truncated = bool(source.get("truncated"))

    def strip_node(node: Any) -> Optional[Dict[str, Any]]:
        nonlocal emitted, truncated
        if not isinstance(node, dict):
            return None
        if emitted >= _MAX_PRIOR_ITEMS:
            truncated = True
            return None
        emitted += 1
        out: Dict[str, Any] = {"kind": node.get("kind")}
        keys = ["title", "description", "procedure_html"]
        if bilingual:
            keys += ["title_sl", "description_sl", "procedure_html_sl"]
        for key in keys:
            value = node.get(key)
            if value:
                out[key] = str(value)[:_MAX_PRIOR_HTML]
        if node.get("assertions"):
            out["assertions"] = [str(a) for a in node["assertions"]][:20]
        if node.get("response_sets"):
            out["response_sets"] = node["response_sets"][:4]
        children = [c for c in (strip_node(c) for c in (node.get("children") or [])) if c]
        if children:
            out["children"] = children
        return out

    items = [n for n in (strip_node(n) for n in source["items"]) if n]
    if not items:
        return None

    end = str(audit_file.get("audit_period_end_date") or "")
    fy = f"FY{end[:4]} " if end[:4].isdigit() else ""
    ref = str(wp.get("reference") or "").strip()
    name = str(wp.get("name") or "").strip()
    base = f"{ref + ' ' if ref else ''}{name}".strip()
    head = f"{fy}{base}".strip() or "the client's working paper"
    file_name = str(audit_file.get("name") or "").strip()
    if kind == "structure":
        where = (
            "this file"
            if audit_file.get("is_current_file")
            else (f"'{file_name}'" if file_name else "another of the client's files")
        )
        label = f"the structure of '{name or ref or 'a sibling program'}' in {where} ({section_count} sections)"
    elif kind == "current_year":
        where = f" in this year's file '{file_name}'" if file_name else ""
        label = f"{head}{where} ({section_count} sections)"
    elif kind == "template_peer":
        label = (
            f"the firm template '{file_name or 'working paper template'}' — "
            f"{base or 'the same working paper'} ({section_count} sections)"
        )
    elif kind == "org_practice":
        where = f" from engagement file '{file_name}'" if file_name else " from a recent engagement file"
        label = f"{head}{where} ({section_count} sections)"
    else:
        label = f"{head} ({section_count} sections)"

    return {
        "kind": kind,
        "label": label,
        "matched_by": source.get("matched_by"),
        "audit_file_id": audit_file.get("id"),
        "working_paper_id": wp.get("id"),
        "section_count": section_count,
        "truncated": truncated,
        "items": items,
    }


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------
class ProcedureBuildOutAgent:
    agent_type = "procedure_buildout"
    # this agent is scoped to ONE working paper — the router enforces the param
    requires_working_paper = True
    allowed_tools = [
        "get_working_paper",
        "get_prior_program_sources",
        "get_audit_file_summary",
        "get_risks",
        "get_audit_plan",
        "get_materiality",
        "get_audit_area",
        "bulk_create_program_sections",
        "undo_program_sections_run",
    ]

    def default_goal(self, audit_file_id: int) -> str:
        return f"Draft the audit program for a working paper on audit file {audit_file_id}."

    def build_plan(self, ctx: RunContext) -> List[PlannedStep]:
        wp_id = ctx.working_paper_id
        if not wp_id:
            raise ValueError("procedure_buildout requires working_paper_id")
        if ctx.is_template:
            # TEMPLATE MODE — the "file" is the firm's skeleton: no client, no
            # risks, no materiality, no TB. Draft the firm's STANDARD program
            # from the firm's own reference programs + house style; skip every
            # engagement-data read (empty on a skeleton and misleading in the UI).
            return [
                PlannedStep("Read working paper & configuration", "read", "get_working_paper", {"working_paper_id": int(wp_id)}),
                PlannedStep("Find reference programs across the firm", "read", "get_prior_program_sources", {"working_paper_id": int(wp_id)}),
                PlannedStep("Read file summary", "read", "get_audit_file_summary"),
                PlannedStep("Summarize working-paper setup", "compute", "summarize_config"),
                PlannedStep("Draft the firm's standard program (AI)", "analysis", "propose_procedures"),
                PlannedStep(
                    "Create the approved sections", "write", "bulk_create_program_sections",
                    {"working_paper_id": int(wp_id)}, requires_approval=True,
                ),
            ]
        return [
            PlannedStep("Read working paper & configuration", "read", "get_working_paper", {"working_paper_id": int(wp_id)}),
            PlannedStep("Find this client's earlier programs", "read", "get_prior_program_sources", {"working_paper_id": int(wp_id)}),
            PlannedStep("Read file summary", "read", "get_audit_file_summary"),
            PlannedStep("Read risks", "read", "get_risks"),
            PlannedStep("Read audit plan", "read", "get_audit_plan"),
            PlannedStep("Read materiality", "read", "get_materiality"),
            PlannedStep("Read area accounts & testing", "compute", "gather_area_context"),
            PlannedStep("Summarize working-paper setup", "compute", "summarize_config"),
            PlannedStep("Draft the audit program (AI)", "analysis", "propose_procedures"),
            PlannedStep(
                "Create the approved sections", "write", "bulk_create_program_sections",
                {"working_paper_id": int(wp_id)}, requires_approval=True,
            ),
        ]

    def execute_step(self, step: PlannedStep, ctx: RunContext) -> Any:
        if step.tool == "gather_area_context":
            return self._gather_area_context(ctx)
        if step.tool == "summarize_config":
            return summarize_wp_config(ctx.find("get_working_paper"))
        if step.tool == "propose_procedures":
            return self._propose_procedures(ctx)
        raise ValueError(f"procedure_buildout: unknown compute step '{step.tool}'")

    # ---- write checkpoint: the exact payload the auditor approves -------------
    def prepare_write(self, step: PlannedStep, ctx: RunContext) -> Optional[dict]:
        if step.tool != "bulk_create_program_sections":
            return None
        draft = ctx.find("propose_procedures")
        if not isinstance(draft, dict) or draft.get("error"):
            raise ValueError("drafting failed — nothing to write")
        sections = draft.get("sections") or []
        if not sections:
            raise ValueError("the draft contains no valid sections to create")
        if not ctx.run_id:
            raise ValueError("run id missing — cannot stamp the write for undo")
        return {
            # popped into the URL path by the registry tool (never model-chosen)
            "working_paper_id": int(step.args.get("working_paper_id") or ctx.working_paper_id),
            # stamps every created section's config -> one-click undo_run
            "ai_run_id": str(ctx.run_id),
            "sections": sections,
            # human-readable half of the checkpoint; be ignores these keys
            "preview": draft.get("preview", []),
            "config_summary": draft.get("config_summary", ""),
            "validation_notes": draft.get("validation_notes", []),
            "counts": draft.get("counts", {}),
            # what the draft was grounded on (client's earlier program + style examples)
            "grounding": draft.get("grounding"),
        }

    # ---- compute steps ---------------------------------------------------------
    def _gather_area_context(self, ctx: RunContext) -> Dict[str, Any]:
        """The accounts (and any testing done) behind this WP's area, keyed on the
        working paper's name. Read via the existing audit_area endpoint; tolerant —
        a program WP without TB-matching accounts is normal (e.g. 'Going concern')."""
        wp = ctx.find("get_working_paper") or {}
        name = ((wp.get("working_paper") or {}).get("name") or "").strip() if isinstance(wp, dict) else ""
        if not name:
            return {"accounts": [], "note": "working paper name unavailable"}
        res = ctx.copilot.get("audit_area", {"area": name})
        if not isinstance(res, dict) or "error" in res:
            return {"accounts": [], "note": str(res.get("error") if isinstance(res, dict) else "area lookup failed")}
        accounts = []
        for a in (res.get("accounts") or [])[:25]:
            if not isinstance(a, dict):
                continue
            row = {
                "account": a.get("account_name"),
                "code": a.get("account_code"),
                "cy_amount": a.get("cy_amount"),
                "py_amount": a.get("py_amount"),
            }
            if a.get("samples"):
                row["samples_tested"] = len(a["samples"])
            accounts.append(row)
        return {"accounts": accounts, "count": len(accounts), "testing_performed": bool(res.get("testing_performed"))}

    # ---- analysis: the ONE structured LLM call ---------------------------------
    def _propose_procedures(self, ctx: RunContext) -> Dict[str, Any]:
        config_facts = ctx.find("summarize_config") or {}
        wp_content = ctx.find("get_working_paper") or {}
        existing_items = wp_content.get("items", []) if isinstance(wp_content, dict) else []

        bilingual = bool(config_facts.get("bilingual"))
        # Grounding: this client's own earlier program to ADAPT (or a sibling
        # program as a structure reference — never a template), plus a count of
        # house-style examples — surfaced to the auditor as provenance.
        prior = compact_prior_program(ctx.find("get_prior_program_sources"), bilingual=bilingual)
        style_count = len([e for e in (ctx.style_examples or []) if (e or "").strip()])
        prompt = build_plan_prompt(
            working_paper_name=config_facts.get("working_paper_name"),
            config_facts=config_facts,
            existing_items=existing_items,
            file_summary=ctx.find("get_audit_file_summary"),
            risks=ctx.find("get_risks"),
            audit_plan=ctx.find("get_audit_plan"),
            materiality=ctx.find("get_materiality"),
            area_context=ctx.find("gather_area_context"),
            style_examples=ctx.style_examples,
            language=config_facts.get("primary_language") or ctx.language,
            secondary_language=config_facts.get("secondary_language"),
            prior_program=prior,
            template_mode=bool(ctx.is_template),
        )
        plan = generate_structured(
            prompt, ProcedurePlan,
            system=SYSTEM_PROMPT + (TEMPLATE_SYSTEM_SUFFIX if ctx.is_template else ""),
            # a bilingual draft carries every field twice — give it head-room
            max_output_tokens=16000 if bilingual else 8000,
            usage_out=ctx.usage_out,
        )
        flat = flatten_plan(
            plan.nodes,
            assertions_enabled=bool(config_facts.get("assertions_enabled")),
            bilingual=bilingual,
        )
        return {
            "config_summary": plan.config_summary,
            "sections": flat["sections"],
            "preview": flat["preview"],
            "validation_notes": flat["notes"],
            "counts": flat["counts"],
            "grounding": {
                "source": (
                    {k: prior[k] for k in ("kind", "label", "matched_by", "audit_file_id", "working_paper_id", "section_count")}
                    if prior else None
                ),
                "style_examples": style_count,
            },
        }

    # ---- final structured result (no extra LLM call) ---------------------------
    def synthesize(self, ctx: RunContext) -> Dict[str, Any]:
        draft = ctx.find("propose_procedures") or {}
        write_res = ctx.find("bulk_create_program_sections") or {}
        counts = draft.get("counts", {}) if isinstance(draft, dict) else {}
        created = write_res.get("count", 0) if isinstance(write_res, dict) else 0
        rejected = isinstance(write_res, dict) and write_res.get("rejected") is True

        needs_attention: List[str] = list(draft.get("validation_notes", []) if isinstance(draft, dict) else [])
        unmatched = write_res.get("unmatched_assertions") if isinstance(write_res, dict) else None
        if unmatched:
            needs_attention.append(f"assertion names not found in this organization: {', '.join(unmatched)}")
        if rejected:
            needs_attention.append("the draft was rejected at the checkpoint — nothing was written")

        grounding = draft.get("grounding") if isinstance(draft, dict) else None
        source = (grounding or {}).get("source") if isinstance(grounding, dict) else None
        if isinstance(grounding, dict) and not source:
            needs_attention.append(
                "no reference program of the firm was found to learn from — "
                "the draft was built from professional standards and firm style only"
                if ctx.is_template
                else "no earlier program of this client was found to learn from — "
                "the draft was built from this file's risk data and firm style only"
            )

        if created:
            grounded_on = f"Grounded on {source['label']}. " if source and source.get("label") else ""
            summary = (
                f"{grounded_on}Created {created} section(s) in the working paper: {counts.get('titles', 0)} title(s) and "
                f"{counts.get('procedures', 0)} procedure(s), {counts.get('with_response_sets', 0)} with a tailored "
                f"response set. Undo is available for this run."
            )
        elif rejected:
            summary = "The proposed program was rejected at the approval checkpoint; the working paper is unchanged."
        else:
            summary = "The run finished without writing to the working paper."

        return {
            "summary": summary,
            "config_summary": draft.get("config_summary", "") if isinstance(draft, dict) else "",
            "sections_proposed": counts.get("total_sections", 0),
            "sections_created": created,
            "response_type_breakdown": counts.get("response_types", {}),
            "needs_attention": needs_attention,
            "data_gaps": [],
        }


register_definition(ProcedureBuildOutAgent())
