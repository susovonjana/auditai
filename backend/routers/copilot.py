"""
Copilot router — AI features that live inside a 1audit audit file.

Endpoints:
  POST /copilot/procedure   stream an AI-drafted audit procedure (NDJSON)

The event format mirrors /ask/stream exactly: newline-delimited JSON objects
with a leading {"type":"meta"}, a run of {"type":"delta","text":…}, an optional
{"type":"error","message":…}, and a final {"type":"done", …}.

NOTE: No `from __future__ import annotations` here — slowapi's decorator
interferes with FastAPI's resolution of stringified forward references (same
reason as routers/user.py).
"""
import asyncio
import json
import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from config import RATE_LIMIT_ASK, TOP_K_CHUNKS, COPILOT_PRIME_MEMORY_ORG_ID
from database import get_db, AsyncSessionLocal
from embeddings import embed_query
from rate_limit import limiter
from routers.user import _load_session, _persist_ask_history
import copilot_tools
import proc_memory
import qa
import schemas
import structured
import tb_mapping_engine
import tb_mapping_memory
from prompts import procedure as procedure_prompt
from prompts import findings as findings_prompt

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/copilot", tags=["copilot"])


# Triple-backticks never legitimately appear in semantic HTML, so we can safely
# strip any code-fence markers the model emits despite the prompt forbidding them.
def _strip_fences(text: str) -> str:
    if not text:
        return text
    return text.replace("```html", "").replace("```HTML", "").replace("```", "")


# How long the synchronous tool loop will block waiting for one KB search
# (embedding + multi-query retrieval + rerank). Generous: retrieval can take a
# few seconds, and a slow lookup must not kill the whole chat answer.
_KB_SEARCH_TIMEOUT = 45


async def _kb_retrieve(query: str, language: str, k: int) -> list:
    """Embed + retrieve the top-k knowledge-base passages (auditing standards +
    1audit product help) for the file-mode chat's search_standards tool. Runs on
    the main event loop (scheduled via run_coroutine_threadsafe), so it opens its
    own short-lived session bound to that loop's engine."""
    embedding = await embed_query(query)
    async with AsyncSessionLocal() as db:
        chunks = await qa.retrieve_chunks(db, query, embedding, k, language=language)
    return [
        {
            "document": c.document_filename,
            "page": c.page_number,
            "section": c.section_heading,
            "content": (c.content or "")[:1200],
        }
        for c in chunks
    ]


@router.post("/procedure")
@limiter.limit(RATE_LIMIT_ASK)
async def generate_procedure(
    request: Request,
    payload: schemas.ProcedureRequest,
    db: AsyncSession = Depends(get_db),
):
    """Stream an AI-drafted audit procedure (clean semantic HTML) grounded in
    KB-retrieved standard guidance + the section context the FE supplies."""
    # Auth/session consistency with /ask (anonymous auditai session token).
    await _load_session(db, payload.session_token)

    risks = [r.model_dump() for r in payload.risks]
    query = procedure_prompt.build_retrieval_query(
        section_title=payload.section_title,
        audit_area=payload.audit_area,
        assertions=payload.assertions,
        risks=risks,
    )

    # KB grounding is best-effort: a retrieval/DB hiccup must not block drafting.
    chunks = []
    try:
        embedding = await embed_query(query)
        chunks = await qa.retrieve_chunks(
            db, query, embedding, TOP_K_CHUNKS, language=payload.language
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("procedure KB retrieval failed (continuing ungrounded): %s", exc)

    # House-style few-shot examples (ticket A-1b): this org's closest accepted
    # procedures. Best-effort — an empty/failed lookup just drafts without them.
    risk_summary = proc_memory.summarize_risks(risks)
    examples = []
    try:
        mems = await proc_memory.search_examples(
            db,
            organization_id=payload.organization_id,
            client_sector=payload.client_sector,
            audit_area=payload.audit_area or payload.section_title,
            risk_summary=risk_summary,
            k=3,
        )
        examples = [m.procedure_html for m in mems]
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("proc_memory retrieval failed (continuing without examples): %s", exc)

    system = procedure_prompt.SYSTEM_PROMPT
    user_prompt = procedure_prompt.build_user_prompt(
        section_title=payload.section_title,
        audit_area=payload.audit_area,
        client_sector=payload.client_sector,
        assertions=payload.assertions,
        risks=risks,
        retrieved_chunks=[c.content for c in chunks],
        examples=examples,
        language=payload.language,
    )

    started = time.perf_counter()
    document_filenames = list({c.document_filename for c in chunks})

    async def event_stream():
        yield json.dumps(
            {
                "type": "meta",
                "chunks_found": len(chunks),
                "documents": document_filenames,
            }
        ) + "\n"

        try:
            async for piece in structured.astream_text(
                system, user_prompt, temperature=0.3, max_output_tokens=4096
            ):
                cleaned = _strip_fences(piece)
                if cleaned:
                    yield json.dumps({"type": "delta", "text": cleaned}) + "\n"
        except Exception as exc:
            logger.exception("Procedure streaming error: %s", exc)
            message = (
                qa.friendly_llm_error(exc)
                if qa.is_quota_error(exc)
                else "Generation interrupted; please retry."
            )
            yield json.dumps({"type": "error", "message": message}) + "\n"

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        yield json.dumps(
            {
                "type": "done",
                "response_time_ms": elapsed_ms,
                "documents": document_filenames,
            }
        ) + "\n"

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@router.post("/procedure/confirm")
async def confirm_procedure(
    payload: schemas.ProcedureConfirmRequest,
    db: AsyncSession = Depends(get_db),
):
    """Append an accepted AI-assisted procedure to this org's procedure memory
    (ticket A-1b), so future drafts learn its house style. Additive only — no
    1audit data is touched. Returns {stored, id}. Storing is best-effort: a
    failure here must never break the auditor's save, so errors are swallowed."""
    await _load_session(db, payload.session_token)
    risks = [r.model_dump() for r in payload.risks]
    try:
        mem_id = await proc_memory.add_memory(
            db,
            organization_id=payload.organization_id,
            client_sector=payload.client_sector,
            audit_area=payload.audit_area or payload.section_title,
            risk_summary=proc_memory.summarize_risks(risks),
            assertions=payload.assertions,
            procedure_html=payload.procedure_html,
            confirmed_by=payload.user_id,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("proc_memory store failed: %s", exc)
        return {"stored": False, "id": None}
    return {"stored": bool(mem_id), "id": str(mem_id) if mem_id else None}


@router.post("/findings")
@limiter.limit(RATE_LIMIT_ASK)
async def generate_findings(
    request: Request,
    payload: schemas.ProcedureFindingsRequest,
    db: AsyncSession = Depends(get_db),
):
    """Stream an AI-drafted FINDINGS narrative for a procedure, grounded ONLY in
    the audit file's real results (TB figures + sampling) fetched live from
    1audit-be via the copilot grant. Never invents figures (ISA 220)."""
    await _load_session(db, payload.session_token)

    ctx = copilot_tools.CopilotContext(payload.audit_file_id, payload.copilot_grant)
    # Fetch the linked account's real results + the file summary (blocking HTTP
    # → run in threads). Best-effort: a failure becomes an {"error": …} dict.
    results = await asyncio.to_thread(
        ctx.get,
        "procedure_results",
        {"coa_original_id": payload.coa_original_id, "account": payload.account},
    )
    summary = await asyncio.to_thread(ctx.get, "summary")

    # If we couldn't read ANY file data, the grant is likely invalid/expired.
    results_failed = isinstance(results, dict) and results.get("error")
    summary_failed = isinstance(summary, dict) and summary.get("error")
    grant_broken = bool(results_failed and summary_failed)

    testing_performed = bool(
        isinstance(results, dict) and results.get("testing_performed")
    )
    system = findings_prompt.SYSTEM_PROMPT
    user_prompt = findings_prompt.build_user_prompt(
        procedure_text=payload.procedure,
        results=results,
        summary=summary,
        assertions=payload.assertions,
        language=payload.language,
        testing_performed=testing_performed,
    )

    started = time.perf_counter()

    async def event_stream():
        yield json.dumps(
            {
                "type": "meta",
                "testing_performed": testing_performed,
                "data_available": not grant_broken,
            }
        ) + "\n"

        if grant_broken:
            yield json.dumps(
                {
                    "type": "error",
                    "message": "Could not read this file's data (the copilot grant may have expired). Please try again.",
                }
            ) + "\n"
            yield json.dumps(
                {"type": "done", "response_time_ms": int((time.perf_counter() - started) * 1000)}
            ) + "\n"
            return

        try:
            async for piece in structured.astream_text(
                system, user_prompt, temperature=0.2, max_output_tokens=4096
            ):
                cleaned = _strip_fences(piece)
                if cleaned:
                    yield json.dumps({"type": "delta", "text": cleaned}) + "\n"
        except Exception as exc:
            logger.exception("Findings streaming error: %s", exc)
            message = (
                qa.friendly_llm_error(exc)
                if qa.is_quota_error(exc)
                else "Generation interrupted; please retry."
            )
            yield json.dumps({"type": "error", "message": message}) + "\n"

        yield json.dumps(
            {
                "type": "done",
                "response_time_ms": int((time.perf_counter() - started) * 1000),
                "testing_performed": testing_performed,
            }
        ) + "\n"

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@router.post("/respond")
@limiter.limit(RATE_LIMIT_ASK)
async def generate_response(
    request: Request,
    payload: schemas.ProcedureFindingsRequest,
    db: AsyncSession = Depends(get_db),
):
    """Answer/respond to an audit procedure using ONLY the file's real data, via
    the copilot tool-calling loop (precise — it can filter the trial balance by
    account name to fetch the exact figure a question asks for). Non-streaming:
    returns the drafted response as clean HTML. Never invents figures (ISA 220)."""
    await _load_session(db, payload.session_token)

    if not (payload.procedure and payload.procedure.strip()):
        raise HTTPException(status_code=422, detail="procedure text is required")

    try:
        result = await asyncio.to_thread(
            copilot_tools.respond_to_procedure,
            payload.procedure,
            payload.audit_file_id,
            payload.copilot_grant,
            payload.language,
        )
    except Exception as exc:
        logger.exception("Procedure response error: %s", exc)
        if qa.is_quota_error(exc):
            raise HTTPException(status_code=429, detail=qa.friendly_llm_error(exc))
        raise HTTPException(
            status_code=502, detail="Could not draft a response. Please retry."
        )

    answer = _strip_fences(result.answer or "").strip()
    if not answer:
        raise HTTPException(
            status_code=502,
            detail="Could not read this file's data (the copilot grant may have expired). Please try again.",
        )
    return {"answer": answer, "tools_used": result.tools_used}


@router.post("/chat")
@limiter.limit(RATE_LIMIT_ASK)
async def chat_about_file(
    request: Request,
    payload: schemas.CopilotChatRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Answer a free-form question about ONE audit file via the copilot tool-loop
    — it can read the working-paper index, any working paper's content, the trial
    balance, the financial statements, risks and audit-area results. Grounded only
    in the file's real data; never invents figures (ISA 220). Non-streaming; the
    answer is Markdown (## Answer + ## Follow-up Questions) for the chat bubble."""
    session = await _load_session(db, payload.session_token)
    started = time.perf_counter()

    # A general / knowledge-base question gets the SAME rich answer as the
    # standalone /ask path (qa.answer_question — bold, multi-paragraph, no inline
    # citation). Only questions that genuinely need THIS file's numbers go to the
    # tool loop. Classify first (cheap structured call, run off-thread).
    needs_data = await asyncio.to_thread(
        copilot_tools.needs_file_data, payload.question
    )
    if not needs_data:
        try:
            qres = await qa.answer_question(
                db, session.id, payload.question, language=payload.language
            )
        except Exception as exc:
            logger.exception("Copilot chat KB path error: %s", exc)
            if qa.is_quota_error(exc):
                raise HTTPException(status_code=429, detail=qa.friendly_llm_error(exc))
            raise HTTPException(status_code=502, detail="Could not answer that. Please retry.")
        kb_answer = (qres.answer or "").strip()
        if not kb_answer:
            raise HTTPException(status_code=502, detail="Could not answer that. Please retry.")
        history_id = uuid.uuid4()
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        background_tasks.add_task(
            _persist_ask_history,
            history_id,
            session.id,
            payload.question,
            qres,
            elapsed_ms,
            payload.user_id,
            payload.organization_id,
        )
        return {
            "answer": kb_answer,
            "sources": ["search_standards"],
            "history_id": str(history_id),
        }

    # Otherwise the question needs the file's real data → the copilot tool loop.
    # Bridge async KB retrieval into the synchronous tool loop: the loop runs in a
    # worker thread and, when the model calls search_standards, schedules
    # _kb_retrieve back onto this (running) event loop and blocks for the result.
    loop = asyncio.get_running_loop()

    def kb_search(query: str):
        if not (query and query.strip()):
            return []
        fut = asyncio.run_coroutine_threadsafe(
            _kb_retrieve(query.strip(), payload.language, TOP_K_CHUNKS), loop
        )
        # Raises on timeout/error → run_tool_loop surfaces it to the model as a
        # tool {"error": ...} rather than crashing the request.
        return fut.result(timeout=_KB_SEARCH_TIMEOUT)

    try:
        result = await asyncio.to_thread(
            copilot_tools.answer_about_file,
            payload.question,
            payload.audit_file_id,
            payload.copilot_grant,
            payload.language,
            kb_search=kb_search,
        )
    except copilot_tools.CopilotGrantError as exc:
        logger.info("Copilot chat grant rejected: %s", exc)
        raise HTTPException(
            status_code=401,
            detail="Could not authorize AI access to this file's data (the grant may have expired). Please retry.",
        )
    except Exception as exc:
        logger.exception("Copilot chat error: %s", exc)
        if qa.is_quota_error(exc):
            raise HTTPException(status_code=429, detail=qa.friendly_llm_error(exc))
        raise HTTPException(status_code=502, detail="Could not answer that. Please retry.")

    answer = (result.answer or "").strip()
    if not answer:
        raise HTTPException(
            status_code=502,
            detail="Could not read this file's data (the copilot grant may have expired). Please try again.",
        )
    sources = []
    seen = set()
    for tc in result.tools_used or []:
        name = getattr(tc, "name", None) or (tc.get("name") if isinstance(tc, dict) else None)
        if name and name not in seen:
            seen.add(name)
            sources.append(name)

    # Persist a history row (same table as /ask) so the chat bubble shows the
    # feedback (👍/👎) and translate controls, which key off a history_id. The
    # tool names stand in for documents_referenced. Written in the background.
    history_id = uuid.uuid4()
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    background_tasks.add_task(
        _persist_ask_history,
        history_id,
        session.id,
        payload.question,
        qa.QAResult(answer=answer, was_answered=True, documents_referenced=sources),
        elapsed_ms,
        payload.user_id,
        payload.organization_id,
    )
    return {"answer": answer, "sources": sources, "history_id": str(history_id)}


@router.post("/tb-mapping")
@limiter.limit(RATE_LIMIT_ASK)
async def map_trial_balance(
    request: Request,
    payload: schemas.TbMappingRequest,
    db: AsyncSession = Depends(get_db),
):
    """Suggest a chart-of-account for each unmapped trial-balance account via the
    3-tier cascade (ticket C-3): Tier-1 previous-data exact/fuzzy, Tier-2 org-trend
    semantics + frequency prior (both deterministic, no Gemini), and an optional
    Tier-3 LLM tail. Returns suggestions only — 1audit-be persists after the
    auditor confirms. Never suggests a coa_original_id outside the candidate set."""
    # Trusted server-to-server call from 1audit-be. A session token, when present,
    # is validated for rate-limit scoping; absent is allowed (be already authorized).
    if payload.session_token:
        await _load_session(db, payload.session_token)

    prime_org_id = payload.prime_org_id or COPILOT_PRIME_MEMORY_ORG_ID
    suggestions = await tb_mapping_engine.map_accounts(
        db,
        organization_id=payload.organization_id,
        client_sector=payload.client_sector,
        accounts=payload.accounts,
        coa=payload.coa,
        prior_mappings=payload.prior_mappings,
        prime_org_id=prime_org_id,
        use_llm_tail=payload.use_llm_tail,
        language=payload.language,
    )
    counts: dict = {}
    for s in suggestions:
        counts[s["tier"]] = counts.get(s["tier"], 0) + 1
    return {"suggestions": suggestions, "tier_counts": counts}


@router.post("/tb-mapping/feedback")
@limiter.limit(RATE_LIMIT_ASK)
async def tb_mapping_feedback(
    request: Request,
    payload: schemas.TbMappingFeedbackRequest,
    db: AsyncSession = Depends(get_db),
):
    """Append confirmed TB mappings (AI-accepted OR manual) to tb_mapping_memory so
    the same account auto-resolves via Tier-1 next time (ticket C-6). Best-effort,
    append-only — a storage failure is swallowed and never disrupts the caller."""
    if payload.session_token:
        await _load_session(db, payload.session_token)

    stored = 0
    for item in payload.items:
        try:
            mem_id = await tb_mapping_memory.add_mapping(
                db,
                organization_id=payload.organization_id,
                client_sector=payload.client_sector,
                account_name=item.account_name,
                account_name_sl=item.account_name_sl,
                account_code=item.account_code,
                coa_original_id=item.coa_original_id,
                coa_label=item.coa_label,
                confirmed_by=payload.confirmed_by,
            )
            if mem_id:
                stored += 1
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("tb-mapping feedback store failed: %s", exc)
    return {"stored": stored}
