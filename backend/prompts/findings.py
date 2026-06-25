"""
Prompt builder for AI draft findings (ticket A-6).

Findings record WHAT WAS DONE and WHAT WAS FOUND — this is audit evidence
(ISA 220), so the model must draft ONLY from the file's real results and never
invent figures or finalise a conclusion. The output is a reviewable DRAFT.
"""
from __future__ import annotations

import json
from typing import Any, List, Optional

_LANG_NAME = {"en": "English", "ar": "Arabic"}

_MAX_PROC_CHARS = 6000
_MAX_DATA_CHARS = 8000
_MAX_ASSERTIONS = 20


SYSTEM_PROMPT = (
    "You are an audit assistant drafting working-paper FINDINGS / NOTES in "
    "response to ONE audit procedure, using ONLY the file's real data in the "
    "DATA section. Findings are audit evidence (ISA 220): accuracy is critical "
    "and fabrication or overstatement is unacceptable.\n\n"
    "FIRST decide what the procedure actually asks for, then respond accordingly:\n"
    "A. DIRECT QUESTION (asks for a figure, balance, account, list or "
    "comparison): ANSWER IT DIRECTLY using the figures in DATA — state the "
    "value(s) with the account name + code and the period. A factual lookup "
    "needs no testing, so do NOT add any 'substantive testing not performed' "
    "wording in this case.\n"
    "B. SUBSTANTIVE TEST (sample / verify / recompute / confirm) WITH results in "
    "DATA: summarise what was found, citing the real figures and any exceptions.\n"
    "C. SUBSTANTIVE TEST with NO results in DATA: briefly state the work still to "
    "be performed; do NOT write that work 'was performed / verified / confirmed / "
    "recomputed', do NOT write 'no exceptions were noted', and do NOT state any "
    "tested outcome — absence of testing is not absence of exceptions.\n\n"
    "STRICT RULES (always):\n"
    "1. Every figure you state must appear in DATA. NEVER invent, estimate or "
    "infer numbers, balances, dates, names, or conclusions.\n"
    "2. Distinguish a recorded balance from tested evidence: a trial-balance "
    "amount is only the recorded balance, not proof the procedure was performed; "
    "only sampling/test results are evidence of testing. When you quote a TB "
    "figure for a substantive test that has no results, label it 'per the trial "
    "balance'.\n"
    "3. Do NOT finalise an audit conclusion or opinion — this is a DRAFT for the "
    "auditor to review, complete, and sign off.\n"
    "4. Output ONLY clean semantic HTML using these tags: <p>, <ul>, <ol>, <li>, "
    "<strong>, <em>. No markdown, no code fences, no <html>/<body>, and no "
    "preamble or lead-in sentence — begin your reply with the first HTML tag. "
    "Bold the key figures with <strong>. Be concise and factual. Do NOT add any "
    "'AI-generated' or 'draft' disclaimer line — the app marks AI content itself."
)


def _clean(value: Optional[str]) -> str:
    return (value or "").strip()


def build_user_prompt(
    *,
    procedure_text: Optional[str],
    results: Any,
    summary: Any,
    assertions: List[str],
    language: str = "en",
    testing_performed: bool = False,
) -> str:
    lang_name = _LANG_NAME.get(language, "English")
    assertion_list = [_clean(a) for a in (assertions or []) if _clean(a)][:_MAX_ASSERTIONS]
    assertions_text = ", ".join(assertion_list) if assertion_list else "not specified"

    if testing_performed:
        testing_directive = (
            "TESTING STATUS: tested samples are present in DATA — describe the "
            "results actually found, citing real figures and any exceptions."
        )
    else:
        testing_directive = (
            "TESTING STATUS: no tested samples are present in DATA. If the "
            "procedure is a DIRECT QUESTION, just answer it from the figures in "
            "DATA (no testing wording needed). If it requires SUBSTANTIVE TESTING, "
            "state the work still to be performed — do NOT claim anything was "
            "verified/confirmed/recomputed, do NOT write 'no exceptions', and do "
            "NOT fabricate. You may note a recorded balance 'per the trial "
            "balance' for context only."
        )

    file_ctx = {}
    if isinstance(summary, dict) and not summary.get("error"):
        file_ctx = {
            "currency": summary.get("currency"),
            "period_start": summary.get("period_start"),
            "period_end": summary.get("period_end"),
        }

    data = {"audit_file": file_ctx, "results": results}
    data_json = json.dumps(data, ensure_ascii=False, default=str)[:_MAX_DATA_CHARS]

    proc = _clean(procedure_text)[:_MAX_PROC_CHARS] or "(no procedure text provided)"

    return (
        f"PROCEDURE (the work to perform / responded to):\n{proc}\n\n"
        f"Assertions addressed: {assertions_text}\n\n"
        f"{testing_directive}\n\n"
        f"DATA — the audit file's REAL results. This is the ONLY source of figures "
        f"you may state:\n{data_json}\n\n"
        f"Draft the findings for this procedure in {lang_name}, following every "
        f"rule above. Quote only figures that appear in DATA."
    )
