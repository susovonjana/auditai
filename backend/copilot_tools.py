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
import time
from typing import Any, Callable, Dict, List, Optional

import jwt
import requests
from pydantic import BaseModel

from config import (
    COPILOT_CHAT_MAX_TOKENS,
    COPILOT_DATA_CACHE_TTL_SEC,
    COPILOT_GRANT_SECRET,
    ONEAUDIT_BASE_URL,
    ONEAUDIT_HTTP_CONNECT_TIMEOUT,
    ONEAUDIT_HTTP_TIMEOUT,
)
from copilot_cache import TTLCache
from prompts import personas
from structured import (
    ToolSpec,
    ToolLoopResult,
    astream_tool_loop,
    generate_structured,
    run_tool_loop,
)

logger = logging.getLogger(__name__)

# Process-wide short-TTL cache for file-data fetches (keyed by file+endpoint+args).
# Shared across requests so a burst of questions reuses one fetch. File-scoped, so
# different audit files never collide. See copilot_cache for the staleness model.
_data_cache = TTLCache(COPILOT_DATA_CACHE_TTL_SEC)


def clear_file_cache(audit_file_id: int) -> int:
    """Invalidate every cached fetch for ONE audit file (all endpoints/params).
    Called when 1audit reports the file changed, so the next question re-fetches
    fresh data instead of serving the (now longer-lived) cache. Returns count."""
    return _data_cache.clear_prefix(f"{int(audit_file_id)}:")


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
        # Optional bridge to the per-file semantic index (phase-2 RAG). When set,
        # the model gets a search_file tool to find qualitative working-paper
        # narrative ("which WP discusses going concern?"). Injected by the chat
        # router (retrieval is async; the tool loop is synchronous).
        self.file_search: Optional[Callable[[str], Any]] = None
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
        clean_params = {k: v for k, v in (params or {}).items() if v is not None}
        headers = {"X-Copilot-Grant": self.grant}
        # (connect, read) timeouts: connect is short so a down be is detected at
        # once; read caps a stalled call so it can't freeze the whole answer.
        timeout = (ONEAUDIT_HTTP_CONNECT_TIMEOUT, ONEAUDIT_HTTP_TIMEOUT)
        last_exc = None
        # Retry ONCE on a transient connection error only — NOT on a read timeout
        # (retrying a genuine stall would just double the wait).
        for attempt in range(2):
            try:
                resp = requests.get(url, params=clean_params, headers=headers, timeout=timeout)
                break
            except requests.ConnectionError as exc:
                last_exc = exc
                if attempt == 0:
                    continue
                logger.warning("copilot tool connection error (%s): %s", endpoint, exc)
                return None, {"error": f"Could not reach 1audit for '{endpoint}'."}
            except requests.RequestException as exc:
                logger.warning("copilot tool HTTP error (%s): %s", endpoint, exc)
                return None, {"error": f"Could not reach 1audit for '{endpoint}'."}
        else:  # pragma: no cover - loop always breaks or returns
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

    def post(self, endpoint: str, json_body: Optional[dict] = None) -> Any:
        """POST {base}/copilot/audit_files/{id}/{endpoint} with the grant header.
        Used ONLY by approval-gated agent write tools. Returns the response `data`
        payload, or an {"error": ...} dict (which the runtime treats as a failed
        write and stops the run — never a half-written file)."""
        url = f"{self.base_url}/copilot/audit_files/{self.audit_file_id}/{endpoint}"
        headers = {"X-Copilot-Grant": self.grant}
        timeout = (ONEAUDIT_HTTP_CONNECT_TIMEOUT, ONEAUDIT_HTTP_TIMEOUT)
        try:
            resp = requests.post(url, json=json_body or {}, headers=headers, timeout=timeout)
        except requests.RequestException as exc:
            logger.warning("copilot write error (%s): %s", endpoint, exc)
            return {"error": f"Could not reach 1audit for '{endpoint}'."}
        if resp.status_code != 200:
            return {
                "error": f"1audit returned HTTP {resp.status_code} for '{endpoint}'.",
                "detail": _safe_json(resp),
            }
        body = _safe_json(resp)
        if isinstance(body, dict) and "data" in body:
            return body["data"]
        return body

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

    def validate_grant_local(self) -> None:
        """Validate the grant WITHOUT a be round-trip — checks scope, expiry and
        that it's for THIS file. If COPILOT_GRANT_SECRET is configured (matching
        be's), the HS256 signature is verified too; otherwise we decode-only and
        rely on be enforcing the grant on every tool call (the hard boundary).
        Raises CopilotGrantError on any failure so the chat route returns 401."""
        token = self.grant
        if not token:
            raise CopilotGrantError("Copilot grant is missing.")
        try:
            if COPILOT_GRANT_SECRET:
                decoded = jwt.decode(token, COPILOT_GRANT_SECRET, algorithms=["HS256"])
            else:
                decoded = jwt.decode(token, options={"verify_signature": False})
                exp = decoded.get("exp")
                if exp is not None and float(exp) < time.time():
                    raise CopilotGrantError("Copilot grant has expired.")
        except jwt.ExpiredSignatureError:
            raise CopilotGrantError("Copilot grant has expired.")
        except jwt.PyJWTError as exc:
            raise CopilotGrantError(f"Invalid copilot grant: {exc}")
        if decoded.get("scope") != "copilot":
            raise CopilotGrantError("Invalid copilot grant scope.")
        try:
            if int(decoded.get("audit_file_id")) != self.audit_file_id:
                raise CopilotGrantError("Copilot grant is for a different audit file.")
        except (TypeError, ValueError):
            raise CopilotGrantError("Copilot grant is missing a valid audit_file_id.")


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

    def get_materiality() -> Any:
        return ctx.get("materiality")

    def get_review_points() -> Any:
        return ctx.get("review_points")

    def get_analytical_review() -> Any:
        return ctx.get("analytical_review")

    def get_engagement_team() -> Any:
        return ctx.get("engagement_team")

    def get_sampling_design(account: Optional[str] = None) -> Any:
        return ctx.get("sampling", {"account": account})

    def list_documents() -> Any:
        return ctx.get("documents")

    def search_standards(query: str = "", **_) -> Any:
        # NOT a file tool — searches the shared knowledge base (auditing
        # standards + 1audit product help). Returns relevant passages so the
        # model can answer general/how-to/concept questions without touching
        # this file's data.
        if not ctx.kb_search:
            return {"error": "standards search is unavailable here"}
        return {"results": ctx.kb_search(query or "")}

    def search_file(query: str = "", **_) -> Any:
        # Semantic search over THIS file's working-paper NARRATIVE (procedure
        # questions, the auditor's notes/answers, titles, comments). Use to FIND
        # qualitative content across working papers; then get_working_paper for
        # full detail. Returns passages, each with its working-paper source.
        if not ctx.file_search:
            return {"error": "file search is unavailable here"}
        return {"results": ctx.file_search(query or "")}

    return {
        "get_audit_file_summary": get_audit_file_summary,
        "get_trial_balance": get_trial_balance,
        "get_risks": get_risks,
        "get_procedure_results": get_procedure_results,
        "list_working_papers": list_working_papers,
        "get_financial_statement": get_financial_statement,
        "get_working_paper": get_working_paper,
        "get_audit_area": get_audit_area,
        "get_materiality": get_materiality,
        "get_review_points": get_review_points,
        "get_analytical_review": get_analytical_review,
        "get_engagement_team": get_engagement_team,
        "get_sampling_design": get_sampling_design,
        "list_documents": list_documents,
        "search_standards": search_standards,
        "search_file": search_file,
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
            "and the DUE DATE. ALSO returns the file ADMINISTRATOR / file manager "
            "(name + role) and who CREATED the file. Use this for ANY question "
            "about this file's metadata, deadlines or dates (e.g. 'period end date "
            "of this file', 'what is the due date of this file'), or about who "
            "administers / manages / created / owns this file."
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
    ToolSpec(
        name="get_materiality",
        description=(
            "THIS file's materiality: overall (initial & final), performance "
            "materiality and trivial-misstatement threshold — each with current-"
            "year & prior-year amounts and the relevant percentages — plus the "
            "benchmark bases used. Use for ANY materiality question about this "
            "file (e.g. 'what is the materiality of this file', 'performance "
            "materiality', 'trivial threshold')."
        ),
        parameters={"type": "object", "properties": {}},
    ),
    ToolSpec(
        name="get_review_points",
        description=(
            "THIS file's review points / completion sign-off: each point's text, "
            "whether it's reviewed and resolved (and by whom/when), and the "
            "working paper it sits on. Use for review status / completion / "
            "outstanding-points questions."
        ),
        parameters={"type": "object", "properties": {}},
    ),
    ToolSpec(
        name="get_analytical_review",
        description=(
            "THIS file's analytical review: line items flagged with a significant "
            "variance and the auditor's comments, across the balance sheet and "
            "income statement. Use for analytical-review / variance-explanation "
            "questions."
        ),
        parameters={"type": "object", "properties": {}},
    ),
    ToolSpec(
        name="get_engagement_team",
        description=(
            "THIS file's engagement team: member names, their roles, and whether "
            "they accepted the independence declaration. Use for 'who is on the "
            "team / engagement partner / independence' questions."
        ),
        parameters={"type": "object", "properties": {}},
    ),
    ToolSpec(
        name="get_sampling_design",
        description=(
            "THIS file's samples: each sample's ACCOUNT (name + code), population "
            "amount, planned vs tested item counts (the sample size = no_of_items), "
            "the materiality parameters used, and any misstatement found / "
            "projected. Use for sampling design / sample-size questions. For a "
            "specific account (e.g. 'sample size of the Land account') pass that "
            "account name/code as `account` to narrow the samples."
        ),
        parameters={
            "type": "object",
            "properties": {
                "account": {
                    "type": "string",
                    "description": "optional account name/code to narrow samples to one account",
                }
            },
        },
    ),
    ToolSpec(
        name="list_documents",
        description=(
            "THIS file's documents / supporting evidence (metadata only — name, "
            "reference, type, and linked working papers; not the file contents). "
            "Use for 'what documents are attached / supporting evidence' questions."
        ),
        parameters={"type": "object", "properties": {}},
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


# Per-file semantic search — added to the file-mode loop ONLY when a file_search
# bridge is supplied (the chat path). It searches THIS file's working-paper
# narrative; it does NOT replace the structured tools (TB, materiality, samples…).
FILE_SEARCH_TOOL_SPEC = ToolSpec(
    name="search_file",
    description=(
        "Semantic search over THIS audit file's working-paper NARRATIVE — the "
        "qualitative text the auditor wrote: procedure questions, notes, free-text "
        "answers/conclusions, section titles and comments, across ALL working "
        "papers at once. Use this to FIND where something is discussed or what was "
        "concluded when you don't know which working paper holds it — e.g. 'what "
        "did we conclude on going concern?', 'where are related parties addressed?', "
        "'revenue recognition approach', 'management override response'. Returns "
        "passages, each tagged with its working paper, so you can then call "
        "get_working_paper for the full detail. For STRUCTURED data (figures, "
        "balances, materiality, samples, dates) use the dedicated tools instead — "
        "they are exact and always current."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "what to find in this file's working-paper narrative",
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
    "financial statements, working papers, risks, materiality, review points, "
    "analytical review, engagement team, samples, documents, sampling or audit areas, OR its "
    "profile / metadata (client, sector, currency, status, the file administrator "
    "/ manager or who created it, the audit period and "
    "ANY date such as the period start/end, field-work start, engagement date or "
    "DUE DATE) — use the FILE tools: get_audit_file_summary (for the file's "
    "profile, dates, due date, AND the file administrator / who created it), "
    "get_trial_balance, get_financial_statement, "
    "list_working_papers + get_working_paper, get_risks, get_audit_area, "
    "get_procedure_results, get_materiality (overall / performance / trivial "
    "materiality, cy & py), get_review_points (review / completion sign-off), "
    "get_analytical_review (variances + comments), get_engagement_team (members / "
    "roles / independence), get_sampling_design (samples per ACCOUNT + parameters; "
    "pass an `account` to narrow to one account's sample size), or "
    "list_documents (attached evidence). Any question phrased as '… of this file' "
    "or \"this "
    "file's …\" is about THIS file — answer it from these tools, never from "
    "search_standards. To FIND qualitative working-paper content — what was "
    "concluded or discussed and WHERE (e.g. going concern, related parties, "
    "revenue recognition, management override) — use search_file (semantic search "
    "over all working papers' narrative), then get_working_paper for the full "
    "detail; if unsure which working paper holds something, search_file or "
    "list_working_papers first. (search_file is for qualitative text only — for "
    "figures/balances/materiality/samples/dates use the structured tools above.)\n"
    "BE EFFICIENT WITH TOOLS — latency matters: call ONLY the tools needed to "
    "answer (don't gather extra context 'just in case'); for a specific value "
    "(materiality, a date, the client, an account balance) call that ONE tool "
    "directly. When you need several INDEPENDENT pieces of data, request them "
    "TOGETHER in a single step (parallel tool calls), not one after another. After "
    "search_file, go straight to get_working_paper for the working paper it "
    "surfaced — do NOT also call list_working_papers (search_file already names "
    "the source).\n"
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

# Elevated default: the senior-auditor lens prepended to the file-chat prompt.
# Format/scope/grounding rules above stay authoritative (the lens defers to them).
_SYSTEM_FILE_ANSWER_SENIOR = personas.with_base(SYSTEM_FILE_ANSWER)


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
    specs = list(TOOL_SPECS)
    if kb_search:
        specs.append(STANDARDS_TOOL_SPEC)
    if ctx.file_search:
        specs.append(FILE_SEARCH_TOOL_SPEC)
    lang_name = "Arabic" if language == "ar" else "English"
    user = f"Question: {question}\n\nAnswer in {lang_name}."
    # force_first_call: a question only reaches this loop when it needs THIS
    # file's data, so make the model fetch with a tool before it may answer —
    # never let it guess a figure/date or just say it will look it up.
    return run_tool_loop(
        _SYSTEM_FILE_ANSWER_SENIOR, user, specs, impls, max_steps=6,
        force_first_call=True, usage_out=usage_out,
    )


async def astream_about_file(
    question: str,
    ctx: CopilotContext,
    language: str = "en",
    usage_out: Optional[dict] = None,
    tools_used_out: Optional[list] = None,
):
    """Streaming sibling of ``answer_about_file``: yields the grounded answer's
    text deltas as they generate (the final tool-loop turn streams token by
    token). Expects a PRE-VALIDATED ``ctx`` — the chat route validates the grant
    and warms the file summary up front, so this does not re-validate. If
    ``ctx.kb_search`` is set the model also gets search_standards for general
    questions. Output is capped at COPILOT_CHAT_MAX_TOKENS (file answers are
    short). Fills ``usage_out`` / ``tools_used_out`` once the stream completes."""
    impls = build_tool_impls(ctx)
    specs = list(TOOL_SPECS)
    if ctx.kb_search:
        specs.append(STANDARDS_TOOL_SPEC)
    if ctx.file_search:
        specs.append(FILE_SEARCH_TOOL_SPEC)
    lang_name = "Arabic" if language == "ar" else "English"
    user = f"Question: {question}\n\nAnswer in {lang_name}."
    async for piece in astream_tool_loop(
        _SYSTEM_FILE_ANSWER_SENIOR, user, specs, impls, 6,
        force_first_call=True, max_output_tokens=COPILOT_CHAT_MAX_TOKENS,
        usage_out=usage_out, tools_used_out=tools_used_out,
    ):
        yield piece


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
    "materiality (overall / performance / trivial), review points / sign-off "
    "status, analytical-review variances, the engagement team, samples, attached "
    "documents, a working paper's recorded answers, assessed risks, sampling/exception "
    "results; and\n"
    "- its METADATA / PROFILE: the client or entity name, sector, reporting "
    "currency, the audit period, ANY date (period start/end, prior-year period, "
    "field-work start, engagement date, DUE DATE / deadline), the file's status "
    "or progress, whether it is consolidated, or WHO administers / manages / "
    "created / owns this file (the file administrator).\n"
    "- about a NAMED ACCOUNT in this file: its balance, samples or the SAMPLE SIZE "
    "of an account (e.g. 'sample size of the Land account'), exceptions, or risk.\n"
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
    "like '[Client Name]' or '[amount]'. Only if a value genuinely cannot be "
    "fetched (you CALLED the right tool and it returned no value) write a short "
    "plain sentence saying so (and, if useful, where it is set) instead of a "
    "placeholder — never claim a value is unavailable without having called its "
    "tool first.\n\n"
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
    "2. If there is NO procedure, this is a free-form field. If the field or "
    "instruction asks for a specific FILE VALUE, you MUST call the matching tool "
    "to fetch it before writing — pick the right one: get_materiality (overall / "
    "performance / TRIVIAL materiality, cy & py), get_audit_file_summary (client, "
    "sector, currency, dates, due date, file administrator), get_trial_balance / "
    "get_financial_statement / get_audit_area (balances & account figures), "
    "get_risks, get_review_points, get_analytical_review, get_engagement_team, "
    "get_sampling_design (samples / sample size), list_documents, or "
    "list_working_papers + get_working_paper for qualitative content. Only when "
    "the field is pure text-craft (rewrite, expand, shorten, fix tone or grammar, "
    "with no data value needed) write directly with no tool call, preserving the "
    "existing meaning and facts.\n\n"
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
    role: Optional[str] = None,
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
        personas.with_role(WRITE_SYSTEM_PROMPT, role), user, TOOL_SPECS, impls, max_steps=6,
        force_first_call=bool(proc), usage_out=usage_out,
    )
