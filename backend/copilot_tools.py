"""
Copilot tools — the bridge from the LLM to a single audit file's live data in
1audit-be.

Each tool is an HTTP GET to 1audit-be's internal copilot API, presenting the
short-lived ``X-Copilot-Grant`` header. The ``audit_file_id`` and grant are bound
per request via ``CopilotContext`` (a closure) and are NEVER exposed as
model-visible tool parameters — the model only ever supplies harmless filters
like an account name. This enforces least privilege: the LLM can read only the
one file the grant authorises.

Used by:
  - answer_about_file(...)        — the file-mode chat (tool-calling loop)
  - write_with_file(...)          — the ONE grounded in-file writer (note / response
                                    / comment), via /copilot/write
  - the /copilot/procedure route  — grounds a procedure draft when the auditor's
                                    instruction needs file data (shares the tools)
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import requests
from pydantic import BaseModel

from config import (
    COPILOT_DATA_CACHE_TTL_SEC,
    ONEAUDIT_BASE_URL,
    ONEAUDIT_HTTP_TIMEOUT,
)
from copilot_cache import TTLCache
from structured import ToolSpec, ToolLoopResult, generate_structured, run_tool_loop

logger = logging.getLogger(__name__)

# Process-wide short-TTL cache for file-data fetches (keyed by file+endpoint+args).
# Shared across requests so a burst of questions reuses one fetch. File-scoped, so
# different audit files never collide. See copilot_cache for the staleness model.
_data_cache = TTLCache(COPILOT_DATA_CACHE_TTL_SEC)


class CopilotGrantError(Exception):
    """Raised when 1audit rejects the copilot grant (expired / wrong file). The
    chat router maps this to a 401 so the UI can re-mint a grant."""


class CopilotContext:
    """Per-request binding of audit_file_id + grant. Tool impls close over this
    so the model can never see or change which file is being read."""

    def __init__(
        self,
        audit_file_id: int,
        grant: str,
        base_url: Optional[str] = None,
        kb_search: Optional[Callable[[str], Any]] = None,
        use_cache: bool = False,
    ):
        self.audit_file_id = int(audit_file_id)
        self.grant = grant
        self.base_url = (base_url or ONEAUDIT_BASE_URL).rstrip("/")
        # Optional bridge to the standards/help knowledge base (RAG). When set,
        # the model also gets a search_standards tool for GENERAL questions so it
        # never sweeps this file's data to answer "what is a branch?". Injected by
        # the router because retrieval is async and the tool loop is synchronous.
        self.kb_search = kb_search
        # Serve repeated identical fetches from the short-TTL cache. Enabled only
        # for the chat path (repeated questions), and only AFTER validate_grant()
        # has confirmed this request's grant — so a cache hit never bypasses auth.
        self.use_cache = bool(use_cache)

    def _cache_key(self, endpoint: str, params: Optional[dict]) -> str:
        items = sorted((params or {}).items())
        return f"{self.audit_file_id}:{endpoint}:{items}"

    def _request(self, endpoint: str, params: Optional[dict]):
        """Raw GET to 1audit-be. Returns (status_code, payload). status_code is
        None on a network error; payload is the `data` body or an {"error": …}
        dict the LLM (or caller) can reason about."""
        url = f"{self.base_url}/copilot/audit_files/{self.audit_file_id}/{endpoint}"
        try:
            resp = requests.get(
                url,
                params={k: v for k, v in (params or {}).items() if v is not None},
                headers={"X-Copilot-Grant": self.grant},
                timeout=ONEAUDIT_HTTP_TIMEOUT,
            )
        except requests.RequestException as exc:
            logger.warning("copilot tool HTTP error (%s): %s", endpoint, exc)
            return None, {"error": f"Could not reach 1audit for '{endpoint}'."}
        if resp.status_code != 200:
            return resp.status_code, {
                "error": f"1audit returned HTTP {resp.status_code} for '{endpoint}'.",
                "detail": _safe_json(resp),
            }
        body = _safe_json(resp)
        # 1audit-be wraps successful responses as { message, success, data }.
        if isinstance(body, dict) and "data" in body:
            return 200, body["data"]
        return 200, body

    def get(self, endpoint: str, params: Optional[dict] = None) -> Any:
        """GET {base}/copilot/audit_files/{id}/{endpoint}. Returns the response
        `data` payload, or an {"error": ...} dict the LLM can reason about. When
        caching is enabled, a fresh identical fetch is served from memory; only
        successful (non-error) responses are cached."""
        if self.use_cache:
            key = self._cache_key(endpoint, params)
            cached = _data_cache.get(key)
            if cached is not None:
                return cached
        status, payload = self._request(endpoint, params)
        if self.use_cache and status == 200:
            _data_cache.set(self._cache_key(endpoint, params), payload)
        return payload

    def validate_grant(self) -> None:
        """Confirm this request's grant is valid for this file via ONE real
        (uncached) summary fetch. Raises CopilotGrantError on an auth rejection
        (401/403) so the chat router can return a clean 401. A network/5xx error
        does NOT block — the tool loop then surfaces it gracefully. On success the
        summary is warmed into the cache, so the model's get_audit_file_summary
        tool reuses it."""
        status, payload = self._request("summary", None)
        if status in (401, 403):
            detail = payload.get("error") if isinstance(payload, dict) else None
            raise CopilotGrantError(detail or "Copilot grant rejected for this file.")
        if status == 200 and self.use_cache:
            _data_cache.set(self._cache_key("summary", None), payload)


def _safe_json(resp) -> Any:
    try:
        return resp.json()
    except Exception:
        return {"raw": (resp.text or "")[:500]}


# ---------------------------------------------------------------------------
# Tool implementations (bound to a context)
# ---------------------------------------------------------------------------
def build_tool_impls(ctx: CopilotContext) -> Dict[str, Callable[..., Any]]:
    def get_audit_file_summary() -> Any:
        return ctx.get("summary")

    def get_trial_balance(contains: Optional[str] = None) -> Any:
        return ctx.get("trial_balance", {"contains": contains})

    def get_risks() -> Any:
        return ctx.get("risks")

    def get_procedure_results(
        coa_original_id: Optional[int] = None, account: Optional[str] = None
    ) -> Any:
        return ctx.get(
            "procedure_results",
            {"coa_original_id": coa_original_id, "account": account},
        )

    def list_working_papers() -> Any:
        return ctx.get("working_papers")

    def get_financial_statement(statement_type: str) -> Any:
        return ctx.get("financial_statement", {"type": statement_type})

    def get_working_paper(working_paper_id: int) -> Any:
        return ctx.get(f"working_papers/{int(working_paper_id)}/content")

    def get_audit_area(
        area: Optional[str] = None, coa_original_id: Optional[int] = None
    ) -> Any:
        return ctx.get("audit_area", {"area": area, "coa_original_id": coa_original_id})

    def search_standards(query: str = "", **_) -> Any:
        # NOT a file tool — searches the shared knowledge base (auditing
        # standards + 1audit product help). Returns relevant passages so the
        # model can answer general/how-to/concept questions without touching
        # this file's data.
        if not ctx.kb_search:
            return {"error": "standards search is unavailable here"}
        return {"results": ctx.kb_search(query or "")}

    return {
        "get_audit_file_summary": get_audit_file_summary,
        "get_trial_balance": get_trial_balance,
        "get_risks": get_risks,
        "get_procedure_results": get_procedure_results,
        "list_working_papers": list_working_papers,
        "get_financial_statement": get_financial_statement,
        "get_working_paper": get_working_paper,
        "get_audit_area": get_audit_area,
        "search_standards": search_standards,
    }


# Tool specifications exposed to the model. NOTE: audit_file_id / grant are NOT
# here — they are bound server-side per request.
TOOL_SPECS: List[ToolSpec] = [
    ToolSpec(
        name="get_audit_file_summary",
        description=(
            "THIS audit file's profile and key dates: file name, client, sector, "
            "reporting currency, status, the audit period and its start/end dates "
            "(plus the prior-year period), field-work start date, engagement date "
            "and the DUE DATE. Use this for ANY question about this file's "
            "metadata, deadlines or dates (e.g. 'period end date of this file', "
            "'what is the due date of this file')."
        ),
        parameters={"type": "object", "properties": {}},
    ),
    ToolSpec(
        name="get_trial_balance",
        description="Mapped trial-balance accounts with current/prior-year amounts.",
        parameters={
            "type": "object",
            "properties": {
                "contains": {
                    "type": "string",
                    "description": "optional case-insensitive account name/code filter",
                }
            },
        },
    ),
    ToolSpec(
        name="get_risks",
        description="Assessed risks for this file (title, description, level).",
        parameters={"type": "object", "properties": {}},
    ),
    ToolSpec(
        name="get_procedure_results",
        description=(
            "The REAL results for a procedure's account: trial-balance figures "
            "plus sampling outcomes and exceptions. Use to answer what was found."
        ),
        parameters={
            "type": "object",
            "properties": {
                "coa_original_id": {
                    "type": "integer",
                    "description": "the linked chart-of-accounts account id, if known",
                },
                "account": {
                    "type": "string",
                    "description": "optional account name or code to look up",
                },
            },
        },
    ),
    ToolSpec(
        name="list_working_papers",
        description=(
            "List every working paper in THIS audit file — reference, name, "
            "status, section, and its working_paper_id. Call this first to "
            "discover what exists, then get_working_paper for the details."
        ),
        parameters={"type": "object", "properties": {}},
    ),
    ToolSpec(
        name="get_financial_statement",
        description=(
            "The computed financial statement for THIS file, with line items and "
            "current-year vs prior-year amounts. Use for balance-sheet / income-"
            "statement / cash-flow / equity questions."
        ),
        parameters={
            "type": "object",
            "properties": {
                "statement_type": {
                    "type": "string",
                    "enum": [
                        "balance_sheet",
                        "income_statement",
                        "cash_flow",
                        "changes_in_equity",
                    ],
                    "description": "which financial statement to fetch",
                }
            },
            "required": ["statement_type"],
        },
    ),
    ToolSpec(
        name="get_working_paper",
        description=(
            "The full content of ONE working paper by its working_paper_id (from "
            "list_working_papers): each procedure's question, the auditor's "
            "responses, notes and sign-off. Use for qualitative working papers "
            "(understanding the entity, going concern, risk assessment, etc.)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "working_paper_id": {
                    "type": "integer",
                    "description": "the working_paper_id from list_working_papers",
                }
            },
            "required": ["working_paper_id"],
        },
    ),
    ToolSpec(
        name="get_audit_area",
        description=(
            "The accounts in an audit area (by name/code, e.g. 'receivables', "
            "'PPE', 'revenue') with their real trial-balance figures plus sampling "
            "outcomes and exceptions."
        ),
        parameters={
            "type": "object",
            "properties": {
                "area": {
                    "type": "string",
                    "description": "audit area or account name/code to look up",
                },
                "coa_original_id": {
                    "type": "integer",
                    "description": "optional chart-of-accounts id to narrow to one account",
                },
            },
        },
    ),
]


# Knowledge-base tool — added to the file-mode loop ONLY when a kb_search bridge
# is supplied. It is NOT a file tool: it answers general/help/standards questions
# so the model doesn't sweep this file's data for "what is a branch?".
STANDARDS_TOOL_SPEC = ToolSpec(
    name="search_standards",
    description=(
        "Search the 1audit knowledge base — auditing standards (ISA, IFRS, "
        "Saudi SOCPA/ZATCA) AND 1audit product help. Use this for GENERAL "
        "questions: definitions and concepts ('what is a branch?'), what a "
        "standard requires ('what does ISA 315 say?'), or how to use 1audit "
        "('how do I add a working paper?'). Do NOT use the file tools for these. "
        "Returns relevant passages, each with its source document."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "what to look up in the knowledge base",
            }
        },
        "required": ["query"],
    },
)


SYSTEM_FILE_ANSWER = (
    "You are the 1audit assistant. You answer two kinds of questions: (a) about "
    "THIS specific audit file's own data, and (b) general auditing / standards / "
    "1audit product-help questions.\n\n"
    "DECIDE THE SOURCE BEFORE CALLING ANY TOOL:\n"
    "1. ABOUT THIS FILE — its figures, balances, accounts, trial balance, "
    "financial statements, working papers, risks, sampling or audit areas, OR its "
    "profile / metadata (client, sector, currency, status, the audit period and "
    "ANY date such as the period start/end, field-work start, engagement date or "
    "DUE DATE) — use the FILE tools: get_audit_file_summary (for the file's "
    "profile, dates and due date), get_trial_balance, get_financial_statement, "
    "list_working_papers + get_working_paper, get_risks, get_audit_area, "
    "get_procedure_results. Any question phrased as '… of this file' or \"this "
    "file's …\" is about THIS file — answer it from these tools, never from "
    "search_standards. (If unsure which working paper holds something, call "
    "list_working_papers first.)\n"
    "2. GENERAL — a definition or concept (e.g. 'what is a branch?'), what a "
    "standard requires (e.g. 'what does ISA 315 say?'), or how to use 1audit "
    "(e.g. 'how do I add a working paper?') — call search_standards. NEVER call "
    "the file tools for these; they hold only this file's data and would waste "
    "effort and find nothing. A general question is NOT a reason to read the "
    "file.\n\n"
    "GROUNDING — STRICT: Before you state ANY fact about this file (a figure, "
    "balance, account, name, date, currency, status or conclusion) you MUST have "
    "called a file tool that returned it. If you have not called the right tool "
    "yet, call it now — do NOT answer a file question from memory, assumption or "
    "a typical value. If the tool returns an error or has no value for what was "
    "asked, say plainly you could not retrieve it (and, if useful, where it is "
    "set) — NEVER guess or invent a value. Every file-specific number, name and "
    "date in your answer must trace to a tool result. For GENERAL "
    "auditing / accounting / standards questions, prefer the search_standards "
    "passages. If they do not cover it but it is a general professional "
    "concept, answer from your own professional knowledge and open that answer "
    "with the italic line '*General guidance — verify against the standard.*' — "
    "but NEVER use general knowledge for this file's figures or "
    "for 1audit product behaviour, and never fabricate file-specific facts.\n\n"
    "OUTPUT FORMAT — your FINAL reply must be Markdown with these two sections, "
    "and keep BOTH headers exactly in English even when you answer in another "
    "language:\n"
    "## Answer\n"
    "Lead with the direct answer, formatted for easy reading:\n"
    "- Do NOT narrate your process or internal steps (no 'let me check…', 'I "
    "need to access the summary…', 'based on the documentation') and do not "
    "mention tools — just give the answer.\n"
    "- **Bold the key figures, dates and values.** For money, include the "
    "currency and clearly label current year vs prior year.\n"
    "- Use a short '- ' bullet list when there are several values, accounts or "
    "rows; keep a single value inline in a sentence.\n"
    "- ADAPT THE DEPTH to the question: for a simple lookup (one date, name, "
    "status or figure) keep it tight — just the value and its source, no "
    "padding. For an ANALYTICAL question (a variance, trend, comparison, "
    "materiality, or 'is this significant') add ONE short line of grounded "
    "interpretation drawn from the figures — the size and direction of a change "
    "and why it may matter — but never speculate beyond what the numbers show.\n"
    "- End by briefly citing what you used (e.g. 'from the trial balance', 'per "
    "the income statement', or 'per ISA 315').\n\n"
    "## Follow-up Questions\n"
    "Up to three short questions the user is likely to ask next, as a '- ' "
    "bullet list. EACH must be answerable from THIS file's tools or the standards "
    "knowledge base: for a data answer suggest drill-downs (the prior-year "
    "figure, that account's samples/exceptions, the working paper that tests it); "
    "for a general/standards answer suggest closely related concepts. Never "
    "suggest something the tools cannot answer. If you cannot form good grounded "
    "follow-ups, OMIT this whole section rather than invent.\n"
    "Write the Answer text and the questions in the user's language; the two "
    "'##' headers always stay in English."
)


def answer_about_file(
    question: str,
    audit_file_id: int,
    grant: str,
    language: str = "en",
    base_url: Optional[str] = None,
    kb_search: Optional[Callable[[str], Any]] = None,
    usage_out: Optional[dict] = None,
) -> ToolLoopResult:
    """Answer a question about the given audit file via the tool-calling loop.
    When ``kb_search`` is supplied the model also gets search_standards, so it
    can route general/help/standards questions to the knowledge base instead of
    sweeping this file's data. Returns ToolLoopResult(answer, tools_used)."""
    # use_cache=True: across a burst of questions, repeated identical fetches
    # (summary, full trial balance, risks…) are served from the short-TTL cache
    # instead of re-querying 1audit-be. validate_grant() does one real summary
    # fetch first, so every chat request re-checks authorization before any cache
    # hit and warms the summary entry. (Raises CopilotGrantError on a bad grant.)
    ctx = CopilotContext(
        audit_file_id, grant, base_url, kb_search=kb_search, use_cache=True
    )
    ctx.validate_grant()
    impls = build_tool_impls(ctx)
    specs = TOOL_SPECS + [STANDARDS_TOOL_SPEC] if kb_search else TOOL_SPECS
    lang_name = "Arabic" if language == "ar" else "English"
    user = f"Question: {question}\n\nAnswer in {lang_name}."
    # force_first_call: a question only reaches this loop when it needs THIS
    # file's data, so make the model fetch with a tool before it may answer —
    # never let it guess a figure/date or just say it will look it up.
    return run_tool_loop(
        SYSTEM_FILE_ANSWER, user, specs, impls, max_steps=6,
        force_first_call=True, usage_out=usage_out,
    )


# ---------------------------------------------------------------------------
# Intent classification (file-data question vs general/KB question)
# ---------------------------------------------------------------------------
# A general/KB question inside a file should get the SAME polished answer as the
# standalone /ask path (rich, bolded, no inline citation) — not the tool-loop's
# plainer wording. So we classify first and let the router send general questions
# through qa.answer_question. Only questions that genuinely need THIS file's
# numbers stay on the tool loop.
class _IntentResult(BaseModel):
    needs_file_data: bool


_INTENT_SYSTEM = (
    "You route a user's question asked inside an audit file's chat. Decide "
    "whether answering it REQUIRES this specific audit file's own data.\n"
    "Set needs_file_data=TRUE when the question asks for anything that belongs to "
    "THIS file, including:\n"
    "- its FIGURES: account balances, trial balance, financial-statement amounts, "
    "a working paper's recorded answers, assessed risks, sampling/exception "
    "results; and\n"
    "- its METADATA / PROFILE: the client or entity name, sector, reporting "
    "currency, the audit period, ANY date (period start/end, prior-year period, "
    "field-work start, engagement date, DUE DATE / deadline), the file's status "
    "or progress, or whether it is consolidated.\n"
    "A strong signal is wording that points at the current file — 'this file', "
    "'this audit', 'this engagement', 'of this file', \"the file's …\" — or asking "
    "for the VALUE of a property (what IS the period end date / due date / client "
    "/ currency of this file). Those are needs_file_data=TRUE.\n"
    "Set needs_file_data=FALSE only for GENERAL questions that are not about this "
    "file's own value: a definition or concept ('what is a balance sheet?', 'what "
    "does a due date mean?'), what an auditing standard requires, or how to USE "
    "the 1audit product ('how do I set the due date?', 'how do I add a working "
    "paper?'). When in doubt, prefer TRUE."
)


def needs_file_data(question: str) -> bool:
    """True if the question needs THIS file's real data (→ tool loop); False for a
    general/KB question (→ the /ask generator). On any failure defaults to True so
    the robust tool loop (which still has search_standards) handles it."""
    try:
        res = generate_structured(
            f"Question: {question}",
            _IntentResult,
            system=_INTENT_SYSTEM,
            temperature=0.0,
            max_output_tokens=256,
            tier="fast",  # cheap routing decision — use the Haiku-class model
        )
        return bool(res.needs_file_data)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("intent classify failed (%s); defaulting to tool loop", exc)
        return True


# ---------------------------------------------------------------------------
# Grounded in-file writer (the ✨ "Generate with AI" button inside an audit file)
# ---------------------------------------------------------------------------
# The SINGLE grounded engine for every non-procedure field inside an audit file —
# the note/findings field, the response field, AND free-form comment/other fields
# all run through here so they behave identically (the user explicitly asked for
# one consistent grounded assistant, 2026-06-25). Bound to ONE audit file via a
# grant; it reads the file's real data through the tools when the field needs it.
#
# It adapts to what it's given:
#   - a <procedure> present  → the field documents/answers that procedure (note or
#     response): draft grounded in the procedure's REAL results (force a data fetch
#     first, so the answer is always grounded).
#   - no procedure, just text → free-form field (comment/title/etc.): polish or
#     draft per the instruction; fetch a file fact only when the instruction needs
#     one (e.g. "the client name") — otherwise no tool call.
WRITE_SYSTEM_PROMPT = (
    "You are an audit assistant helping an auditor write the text of a SINGLE "
    "field of a working paper. You are bound to ONE audit file and read its real "
    "data through the tools. Accuracy is critical (ISA 220): never invent a "
    "figure, name, date or conclusion, and NEVER emit a bracketed placeholder "
    "like '[Client Name]' or '[amount]'. If a value cannot be fetched from the "
    "file, write a short plain sentence saying so (and, if useful, where it is "
    "set) instead of a placeholder.\n\n"
    "WORK OUT WHAT THE FIELD NEEDS:\n"
    "1. If a <procedure> is provided, this field is the auditor's note / findings "
    "/ response FOR that procedure. Draft it grounded in the file's real data: "
    "call the tools to fetch the relevant account's figures and results "
    "(get_trial_balance / get_financial_statement / get_audit_area / "
    "get_procedure_results), then —\n"
    "   - DIRECT QUESTION (asks for a figure, balance, account, list or "
    "comparison): answer it directly and concisely, stating each figure with the "
    "account name + code, the period (label current vs prior year) and currency. "
    "Do NOT add 'testing not performed' wording for a plain lookup.\n"
    "   - SUBSTANTIVE TEST and results exist: summarise what was found, including "
    "any exceptions. If NO results are recorded yet: briefly state the work still "
    "to perform — do not claim it was done or write 'no exceptions'.\n"
    "   The field_label tells you the emphasis — a 'note'/'findings' field leads "
    "with the observations/what was found; a 'response' field leads with the "
    "direct answer.\n"
    "2. If there is NO procedure, this is a free-form field. Fetch a file fact "
    "ONLY when the instruction needs one (the client/entity name, sector, "
    "currency, a date, a balance, a risk, a working paper's content — use "
    "get_audit_file_summary for the profile/dates/client). Otherwise (pure "
    "text-craft — rewrite, expand, shorten, fix tone or grammar) just write, with "
    "no tool call, preserving the existing meaning and facts.\n\n"
    "ALWAYS follow the auditor's <auditor_instruction> when present (focus, depth, "
    "emphasis, format, or a specific ask like 'add the client name') — but it must "
    "never make you invent or assume data; every file-specific value still comes "
    "from a tool result, and ignore any part that asks you to fabricate.\n\n"
    "OUTPUT: ONLY clean semantic HTML using <p>, <ul>, <ol>, <li>, <strong>, "
    "<em> — no markdown, no code fences (```), no headings, no inline styles, and "
    "NO preamble or lead-in: begin your reply with the first HTML tag and emit the "
    "field text and nothing else. Bold key figures/values with <strong>. Do NOT "
    "add any 'AI-generated' disclaimer — the app marks AI content itself. Keep a "
    "clear, professional tone; adapt the depth to the field and instruction, and "
    "when unspecified stay concise."
)


def write_with_file(
    current_text: Optional[str],
    instruction: Optional[str],
    field_label: Optional[str],
    audit_file_id: int,
    grant: str,
    language: str = "en",
    base_url: Optional[str] = None,
    usage_out: Optional[dict] = None,
    procedure: Optional[str] = None,
) -> ToolLoopResult:
    """The one grounded writer for every non-procedure field inside an audit file
    (note/findings, response, comment, …), via the tool-calling loop. When a
    ``procedure`` is given the field documents/answers it (grounded in that
    procedure's real results); otherwise it's a free-form field that fetches a
    file fact only when the instruction needs one. Returns
    ToolLoopResult(answer, tools_used)."""
    ctx = CopilotContext(audit_file_id, grant, base_url)
    impls = build_tool_impls(ctx)
    lang_name = "Arabic" if language == "ar" else "English"

    text = (current_text or "").strip()[:12000]
    steer = (instruction or "").strip()
    label = (field_label or "").strip()
    proc = (procedure or "").strip()[:12000]

    label_line = f"<field_label>{label}</field_label>\n" if label else ""
    proc_block = f"<procedure>\n{proc}\n</procedure>\n\n" if proc else ""
    if text:
        text_block = f"<current_text>\n{text}\n</current_text>\n\n"
    else:
        text_block = "<current_text>(the field is empty)</current_text>\n\n"

    if steer:
        instr_block = (
            f"<auditor_instruction>\n{steer}\n</auditor_instruction>\n\n"
        )
    elif proc:
        instr_block = (
            "No extra instruction was given. Draft this field for the procedure "
            "above, grounded in the file's real data.\n\n"
        )
    elif text:
        instr_block = (
            "No instruction was given. Improve the wording, clarity, structure and "
            "professionalism of the current text without changing its meaning or "
            "adding facts. Do not call any tool.\n\n"
        )
    else:
        instr_block = (
            "No instruction and no text were provided. Write a short, neutral "
            "professional placeholder paragraph the auditor can replace. Do not "
            "call any tool.\n\n"
        )

    user = (
        f"{label_line}{proc_block}{text_block}{instr_block}"
        f"Produce the field text now, following the output rules. Write in {lang_name}."
    )
    # Force a first tool call only when documenting/answering a procedure (the
    # note/response case must be grounded in real results, like respond did). A
    # free-form field decides for itself whether it needs the tools.
    return run_tool_loop(
        WRITE_SYSTEM_PROMPT, user, TOOL_SPECS, impls, max_steps=6,
        force_first_call=bool(proc), usage_out=usage_out,
    )
