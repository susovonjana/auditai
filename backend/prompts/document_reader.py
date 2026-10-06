"""
Prompts for the DOCUMENT READER — the per-document "read, understand, summarise"
engine behind the All-documents AI summary and the ``document_extraction`` agent.

The reader behaves like dropping a file into Claude/ChatGPT and asking "what is
this?": it classifies the document, pulls out the facts an auditor cares about
(parties, dates, amounts, VAT, line items, reference numbers), writes a short and
a full summary, and says which audit area / working paper the evidence supports.

Grounding rules (ISA 220 / ISA 500 evidence discipline):
  * Extract ONLY what is visible in the document. Never invent a figure, party,
    date or number. Unreadable → say so (``data_gaps``), lower ``confidence``.
  * Keep identifiers, codes, names and amounts exactly as printed (no
    translation, no rounding, no currency conversion).
  * Arithmetic checks are NOT done by the model — ``document_reader.run_checks``
    recomputes them in Python from the extracted figures.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

_LANG_NAME = {"en": "English", "ar": "Arabic"}

# Closed-ish taxonomy the model picks from (free text allowed as a fallback so a
# genuinely new kind of document is never forced into the wrong bucket).
DOC_TYPES: List[str] = [
    "invoice", "credit_note", "receipt", "purchase_order", "quotation", "delivery_note",
    "bank_statement", "bank_confirmation", "bank_letter", "cheque",
    "contract", "lease_agreement", "loan_agreement", "insurance_policy",
    "payroll_register", "payslip", "gosi_statement",
    "vat_return", "zakat_tax_return", "tax_certificate", "customs_declaration",
    "financial_statements", "trial_balance", "general_ledger", "journal_voucher",
    "fixed_asset_register", "inventory_count_sheet", "ageing_report", "reconciliation",
    "confirmation_letter", "legal_letter", "board_minutes", "shareholder_resolution",
    "commercial_registration", "articles_of_association", "licence_permit",
    "identity_document", "correspondence", "memo", "report", "spreadsheet", "other",
]

SYSTEM_PROMPT = (
    "You are the DOCUMENT READER inside 1audit, an audit working-paper system used by "
    "external auditors (Saudi Arabia / IFRS / ISAs as adopted by SOCPA; VAT 15%; ZATCA; "
    "GOSI). An auditor has opened ONE uploaded document and wants to understand it fast: "
    "what it is, who it involves, the key figures and dates, and what it is evidence for.\n\n"
    "READ the document content you are given (page images and/or extracted text — read the "
    "images carefully, they may be scans or photos in Arabic, English or both, including "
    "handwriting, stamps and signatures) and fill the structured result.\n\n"
    "RULES:\n"
    "1. Extract ONLY what is actually in the document. Never invent, infer or 'complete' a "
    "figure, name, date, VAT number or IBAN. If something is unreadable or absent, leave it "
    "out and mention it in data_gaps. Lower `confidence` when the document is partially "
    "legible.\n"
    "2. Copy identifiers, amounts, codes and names EXACTLY as printed (same digits, same "
    "spelling, same currency). Do not translate names or references. Amount values must be "
    "plain numbers (no thousands separators); put the currency code/symbol in `currency`.\n"
    "3. Classify `doc_type` from the taxonomy; use 'other' with a clear `title` when none fits.\n"
    "4. `summary_short` is ONE or TWO sentences an auditor can read in the document list. "
    "`summary` is the full picture in 4–10 bullet-style sentences: purpose, parties, period, "
    "key figures, terms/conditions, signatures/approvals, anything unusual. For a long report "
    "or contract, summarise the substance (obligations, amounts, dates, termination, "
    "penalties) — not the layout.\n"
    "5. `audit_relevance`: say which audit area(s) and financial-statement assertions this "
    "document can support as evidence (e.g. invoice → revenue/receivables: occurrence, "
    "accuracy, cut-off), its evidence quality (original/copy/scan/unsigned/system-generated), "
    "and — ONLY from the working papers list you are given — the working paper(s) it most "
    "likely belongs to. Never invent a working paper name.\n"
    "6. `red_flags`: concrete observations an auditor should look at — missing VAT "
    "registration number on a tax invoice, unsigned/undated contract, altered or "
    "inconsistent figures, dates outside the audit period you were told, related-party "
    "names, round-sum amounts, missing approval. Only what the document shows; do not "
    "speculate. Do NOT perform arithmetic yourself — just extract the figures; the system "
    "re-adds them.\n"
    "7. `language_detected`: the document's own language(s) (e.g. 'ar', 'en', 'ar+en').\n"
    "8. Be concise and specific. No preamble.\n"
)


def _lang(language: Optional[str]) -> str:
    return _LANG_NAME.get((language or "en").lower(), "English")


def build_intro_text(
    *,
    name: Optional[str],
    mime_type: Optional[str],
    read_method: str,
    pages: int,
    truncated: bool,
    file_context: Optional[Dict[str, Any]],
    working_papers: Optional[List[Dict[str, Any]]],
    language: str = "en",
) -> str:
    """The opening text block: what the auditor opened + the engagement context the
    model may use to judge relevance (period, client, currency) + the WP list."""
    ctx_lines: List[str] = []
    fc = file_context or {}
    if fc:
        for key, label in (
            ("client", "Client"), ("sector", "Sector"), ("currency", "File currency"),
            ("period_start", "Audit period start"), ("period_end", "Audit period end"),
            ("name", "Audit file"),
        ):
            v = fc.get(key)
            if v:
                ctx_lines.append(f"- {label}: {v}")
    wp_lines: List[str] = []
    for wp in (working_papers or [])[:120]:
        if not isinstance(wp, dict):
            continue
        ref = wp.get("reference") or ""
        nm = wp.get("name") or ""
        if ref or nm:
            wp_lines.append(f"- {ref}{' · ' if ref and nm else ''}{nm}".strip())
    parts = [
        f"DOCUMENT: {name or 'unnamed'} (type: {mime_type or 'unknown'}; read as: {read_method}; "
        f"pages/sheets considered: {pages}{'; CONTENT TRUNCATED — only part of the document is shown' if truncated else ''})",
    ]
    if ctx_lines:
        parts.append("ENGAGEMENT CONTEXT (for relevance judgement only — not document content):\n" + "\n".join(ctx_lines))
    if wp_lines:
        parts.append(
            "WORKING PAPERS ON THIS FILE (choose suggested_working_papers ONLY from these, by reference/name):\n"
            + "\n".join(wp_lines)
        )
    parts.append(f"DOCUMENT TYPE TAXONOMY: {', '.join(DOC_TYPES)}")
    parts.append("The document content follows.")
    return "\n\n".join(parts)


def build_closing_text(language: str = "en") -> str:
    lang = _lang(language)
    return (
        "Now read everything above and return the structured result. Write every free-text field "
        f"(title, summaries, key_facts, red_flags, audit_relevance notes, data_gaps) in {lang}; keep "
        "quoted identifiers, names, codes and amounts exactly as printed in the document."
    )


# ---------------------------------------------------------------------------
# Follow-up Q&A over ONE document ("Ask about this document")
# ---------------------------------------------------------------------------
ASK_SYSTEM_PROMPT = (
    "You answer an auditor's questions about ONE uploaded document, using ONLY the document "
    "content provided (its extracted text and the structured reading of it). Quote figures, "
    "names, dates and references exactly as they appear. If the answer is not in the document, "
    "say so plainly — never guess or use outside knowledge about the client. Keep answers short "
    "and specific; use Markdown bullets or a small table when listing several values. Do not add "
    "disclaimers."
)


def build_ask_prompt(
    *,
    question: str,
    name: Optional[str],
    insight: Optional[Dict[str, Any]],
    text_excerpt: str,
    language: str = "en",
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    lang = _lang(language)
    hist_lines: List[str] = []
    for turn in (history or [])[-6:]:
        role = (turn.get("role") or "").strip().lower()
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            hist_lines.append(f"{'Auditor' if role == 'user' else 'Assistant'}: {content[:1500]}")
    hist = ("\nEARLIER TURNS:\n" + "\n".join(hist_lines) + "\n") if hist_lines else ""
    insight_json = json.dumps(insight or {}, ensure_ascii=False, default=str)[:12000]
    return (
        f"DOCUMENT: {name or 'document'}\n\n"
        f"STRUCTURED READING (JSON):\n{insight_json}\n\n"
        f"DOCUMENT TEXT (may be partial):\n<document_text>\n{text_excerpt}\n</document_text>\n"
        f"{hist}\n"
        f"QUESTION: {question.strip()}\n\n"
        f"Answer in {lang}, grounded only in the document above."
    )
