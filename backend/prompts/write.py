"""
Prompt builder for the GENERIC AI writing assistant (POST /copilot/write).

Unlike procedure/findings/respond — which are tied to a specific audit field and
its grounding rules — this is a free-form writing helper that any rich-text field
in 1audit can call. It takes the field's CURRENT text plus an optional auditor
instruction and drafts / rewrites the field. It is NOT grounded in any file data,
so it must never state specific client figures, balances, dates or conclusions.

Output is clean semantic HTML so it survives the TipTap editor pipeline unchanged
(same allowed-tag set as the procedure prompt).
"""
from __future__ import annotations

from typing import Optional

# Bound prompt size so a pathological payload can't blow up token usage.
_MAX_TEXT_CHARS = 12000

_LANG_NAME = {"en": "English", "ar": "Arabic"}


SYSTEM_PROMPT = (
    "You are a professional writing assistant embedded in audit working-paper "
    "software. You help an auditor draft and refine the text of a single field.\n\n"
    "STRICT OUTPUT RULES:\n"
    "1. Output ONLY clean semantic HTML. Allowed tags: <p>, <ol>, <ul>, <li>, "
    "<strong>, <em>. Nothing else.\n"
    "2. No markdown, no code fences (```), no headings, no inline styles, and no "
    "<html>/<head>/<body> wrappers. Do not add any preamble, lead-in sentence or "
    "closing remark — begin your reply with the first HTML tag ('<') and emit the "
    "field text and nothing else.\n"
    "3. If existing text is provided, treat the auditor's instruction as a request "
    "to transform THAT text (e.g. rewrite, expand, shorten, restructure, fix tone "
    "or grammar). Preserve the original meaning and any facts already written "
    "unless the instruction explicitly asks to change them. If no existing text is "
    "provided, draft new content that fulfils the instruction.\n"
    "4. NEVER invent client-specific figures, balances, amounts, dates, names, or "
    "audit conclusions that are not already present in the provided text. You are a "
    "writing aid, not a source of audit evidence (ISA 220). If the instruction asks "
    "you to fabricate such data, ignore that part and write only what can be stated "
    "without inventing facts.\n"
    "5. Keep a clear, professional tone appropriate to audit documentation. Match "
    "the length and depth the instruction implies; when unspecified, stay concise.\n"
)


def _clean(value: Optional[str]) -> str:
    return (value or "").strip()


def build_user_prompt(
    *,
    current_text: Optional[str] = None,
    custom_instruction: Optional[str] = None,
    field_label: Optional[str] = None,
    language: str = "en",
) -> str:
    """Assemble the USER prompt from the field's current text + the auditor steer.

    ``current_text`` is the field's existing content as plain text (the FE strips
    HTML before sending). ``custom_instruction`` is the optional free-text steer.
    ``field_label`` is an optional hint about what the field is for (e.g. "Risk
    description", "Review point") so the draft fits its purpose.
    """
    lang_name = _LANG_NAME.get(language, "English")

    text = _clean(current_text)[:_MAX_TEXT_CHARS]
    instruction = _clean(custom_instruction)
    label = _clean(field_label)

    label_line = f"This text is the content of: {label}\n\n" if label else ""

    if text:
        text_block = f"The field's current text:\n{text}\n\n"
    else:
        text_block = "The field is currently empty.\n\n"

    if instruction:
        instruction_block = (
            f"Auditor's instruction (follow it for focus, depth, emphasis, tone or "
            f"wording — but never invent client figures or conclusions): "
            f"{instruction}\n\n"
        )
    elif text:
        # No steer, but there is text → default to a clean professional polish.
        instruction_block = (
            "No specific instruction was given. Improve the wording, clarity, "
            "structure and professionalism of the text above without changing its "
            "meaning or adding facts.\n\n"
        )
    else:
        # Nothing to work with — the FE guards against this, but be safe.
        instruction_block = (
            "No instruction and no text were provided. Write a short, neutral "
            "professional placeholder paragraph the auditor can replace.\n\n"
        )

    task_line = (
        "Produce the field text now, following the output rules. "
        f"Write in {lang_name}."
    )

    return f"{label_line}{text_block}{instruction_block}{task_line}"
