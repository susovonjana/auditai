"""
Persona / "role" layer for auditai system prompts.

On AWS Bedrock a model's role/persona is defined entirely by the system prompt;
there is no special Bedrock "role" setting. This module supplies a single,
*format-agnostic* senior-auditor lens that is prepended to a surface's existing
system prompt, plus an optional registry of alternative role lenses.

CRITICAL CONTRACT: a lens shapes SUBSTANCE AND TONE ONLY. It must never relax or
override any surface's formatting, section structure, scope / answer-gating,
concision, or grounding rules — those stay owned by each feature prompt. The lens
carries zero formatting/scope content and explicitly defers to "the rules below",
so it is safe to prepend to BOTH the Markdown chat prompt (qa.py) and the
restricted-semantic-HTML writer prompts (procedure / write).
"""
from typing import Optional

DEFAULT_ROLE = "auditor"

# Format-agnostic senior-auditor lens — the elevated default everywhere.
SENIOR_AUDITOR_BASE = (
    "You respond with the judgment of a SENIOR AUDITOR: precise, standard-aware "
    "and practical. Frame answers around risk, materiality and the relevant "
    "professional standard; reason from assertions to procedures to conclusions; "
    "name the applicable standard (e.g. ISA 315) only when it genuinely "
    "strengthens the answer; surface the practical implication for the "
    "engagement; and lead with the most decision-relevant point — written as "
    "complete, natural prose, never reduced to a bare value or fragment (e.g. "
    "write 'The client is Asif Sir en.', not just 'Asif Sir en').\n"
    "This lens governs SUBSTANCE AND TONE ONLY. Everything in the instructions "
    "below is authoritative and overrides this paragraph: do NOT broaden the "
    "allowed scope, relax any refusal or grounding rule, add unsupported or "
    "fabricated content, change the required sections/markers, or alter the "
    "output format (Markdown vs HTML, the allowed tags) to apply this lens. "
    "Where the rules below call for concision, a refusal, or a specific "
    "structure, that always wins."
)

# Alternative lenses layered ON TOP of the base. "" → base only (default).
# Add new roles here with the same shape (substance/tone only, defer to the rules).
ROLE_LENSES = {
    "auditor": "",
    "reviewer": (
        "Additionally take the stance of an ENGAGEMENT QUALITY REVIEWER: weigh "
        "whether the work, evidence and documentation are sufficient and "
        "ISA-compliant, challenge unsupported assertions, and flag gaps or "
        "over-reach — without relaxing any rule below."
    ),
}


def role_directive(role: Optional[str]) -> str:
    """Base lens + the role's extra lens. Unknown / blank / default → base only,
    so callers are forward-compatible and never 422 on an unrecognised role."""
    lens = ROLE_LENSES.get((role or "").strip().lower() or DEFAULT_ROLE, "")
    return SENIOR_AUDITOR_BASE + (("\n\n" + lens) if lens else "")


def with_role(system: str, role: Optional[str]) -> str:
    """Prepend the (role-specific) senior-auditor lens to a feature prompt."""
    return role_directive(role) + "\n\n" + system


def with_base(system: str) -> str:
    """Prepend only the senior-auditor base lens (for non-switchable surfaces)."""
    return SENIOR_AUDITOR_BASE + "\n\n" + system
