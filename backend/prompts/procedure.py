"""
Prompt builder for AI procedure generation (ticket A-1).

A procedure says WHAT THE AUDITOR WILL DO. It must never contain client figures
or conclusions — those belong in findings/response (ticket A-6). Output is clean
semantic HTML so it survives the TipTap editor pipeline in 1audit unchanged.
"""
from __future__ import annotations

from typing import List, Optional

# Bound prompt size so a pathological payload can't blow up token usage.
_MAX_RISKS = 8
_MAX_ASSERTIONS = 20
_MAX_CHUNKS = 8
_MAX_CHUNK_CHARS = 1200

_LANG_NAME = {"en": "English", "ar": "Arabic"}


SYSTEM_PROMPT = (
    "You are an experienced audit assistant that drafts audit PROCEDURES — the "
    "step-by-step work an auditor will perform in response to assessed risks.\n\n"
    "STRICT OUTPUT RULES:\n"
    "1. Output ONLY clean semantic HTML. Allowed tags: <p>, <ol>, <ul>, <li>, "
    "<strong>, <em>. Nothing else.\n"
    "2. No markdown, no code fences (```), no headings, no inline styles, and no "
    "<html>/<head>/<body> wrappers. Do not add any preamble or closing remark — "
    "emit the procedure HTML and nothing else.\n"
    "3. Structure the procedure as an ordered list (<ol> of <li> steps). Each step "
    "is one concrete, actionable instruction.\n"
    "4. NEVER state client-specific figures, balances, amounts, or conclusions. A "
    "procedure describes work to PERFORM, not results obtained. Do not invent "
    "numbers, dates, or names.\n"
    "5. Tailor the steps to the audit area, the assessed risk(s), the assertions to "
    "address, and the client sector. Keep steps consistent with the standard "
    "guidance provided. Be specific and practical (e.g. how to select items, what "
    "to inspect, what to corroborate), referencing the relevant assertion where it "
    "adds clarity.\n"
)


def _clean(value: Optional[str]) -> str:
    return (value or "").strip()


def build_user_prompt(
    *,
    section_title: Optional[str],
    audit_area: Optional[str],
    client_sector: Optional[str],
    assertions: List[str],
    risks: List[dict],
    retrieved_chunks: List[str],
    language: str = "en",
) -> str:
    """Assemble the USER prompt from the section context + retrieved KB guidance.

    ``risks`` is a list of dicts with optional ``title`` / ``description`` /
    ``assessment_level`` keys. ``retrieved_chunks`` are raw standard-guidance
    passages pulled from the knowledge base.
    """
    area = _clean(audit_area) or _clean(section_title) or "the audit area"
    sector = _clean(client_sector) or "not specified"
    lang_name = _LANG_NAME.get(language, "English")

    assertion_list = [_clean(a) for a in assertions if _clean(a)][:_MAX_ASSERTIONS]
    assertions_text = ", ".join(assertion_list) if assertion_list else "not specified"

    risk_lines: List[str] = []
    for r in risks[:_MAX_RISKS]:
        title = _clean(r.get("title"))
        desc = _clean(r.get("description"))
        level = _clean(r.get("assessment_level"))
        if not (title or desc):
            continue
        prefix = f"({level}) " if level else ""
        body = title if title else ""
        if desc:
            body = f"{body}: {desc}" if body else desc
        risk_lines.append(f"- {prefix}{body}")
    risks_text = "\n".join(risk_lines) if risk_lines else "- (no specific risk provided)"

    guidance_parts: List[str] = []
    for chunk in retrieved_chunks[:_MAX_CHUNKS]:
        c = _clean(chunk)
        if c:
            guidance_parts.append(c[:_MAX_CHUNK_CHARS])
    guidance_text = (
        "\n\n---\n\n".join(guidance_parts)
        if guidance_parts
        else "(no specific standard guidance retrieved — rely on general audit best practice)"
    )

    section_line = f"Section / work area: {_clean(section_title)}\n" if _clean(section_title) else ""

    return (
        f"{section_line}"
        f"Audit area: {area}\n"
        f"Client sector: {sector}\n"
        f"Assertions to address: {assertions_text}\n\n"
        f"Risk(s) to respond to:\n{risks_text}\n\n"
        f"Relevant standard guidance (from our knowledge base):\n{guidance_text}\n\n"
        f"Write a tailored, step-by-step audit procedure that responds to the "
        f"risk(s) above and covers the listed assertions, consistent with the "
        f"guidance. Be specific and practical. Remember: describe the work to "
        f"perform — do NOT state any client figures or conclusions.\n"
        f"Write the procedure in {lang_name}."
    )


def build_retrieval_query(
    *,
    section_title: Optional[str],
    audit_area: Optional[str],
    assertions: List[str],
    risks: List[dict],
) -> str:
    """Build the KB retrieval query from the section context."""
    parts: List[str] = []
    if _clean(section_title):
        parts.append(_clean(section_title))
    if _clean(audit_area):
        parts.append(_clean(audit_area))
    for r in risks[:_MAX_RISKS]:
        parts.append(_clean(r.get("description")) or _clean(r.get("title")))
    parts.extend(_clean(a) for a in assertions[:_MAX_ASSERTIONS])
    parts.append("audit procedure steps")
    return " ".join(p for p in parts if p)
