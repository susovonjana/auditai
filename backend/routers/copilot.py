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

from config import RATE_LIMIT_ASK, TOP_K_CHUNKS
from database import get_db, AsyncSessionLocal
from embeddings import embed_query
from rate_limit import limiter
from routers.user import _load_session, _persist_ask_history
import copilot_tools
import qa
import schemas
import structured
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

    system = procedure_prompt.SYSTEM_PROMPT
    user_prompt = procedure_prompt.build_user_prompt(
        section_title=payload.section_title,
        audit_area=payload.audit_area,
        client_sector=payload.client_sector,
        assertions=payload.assertions,
        risks=risks,
        retrieved_chunks=[c.content for c in chunks],
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
