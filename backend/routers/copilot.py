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
import usage_meter
from prompts import procedure as procedure_prompt
from prompts import write as write_prompt

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/copilot", tags=["copilot"])


# Triple-backticks never legitimately appear in semantic HTML, so we can safely
# strip any code-fence markers the model emits despite the prompt forbidding them.
def _strip_fences(text: str) -> str:
    if not text:
        return text
    return text.replace("```html", "").replace("```HTML", "").replace("```", "")


# Appended to the procedure SYSTEM prompt when the auditor's instruction may need
# this file's real data (grounded path). Gives the model the file tools so an
# instruction like "use the client name" reads the real value instead of a
# placeholder, without changing the normal procedure-drafting behaviour.
PROCEDURE_GROUNDING_ADDENDUM = (
    "\n\nFILE DATA (GROUNDING): You also have tools to read THIS audit file's real "
    "data. When the auditor's instruction asks for a file-specific fact — the "
    "client/entity name, sector, reporting currency, a date, an account balance or "
    "a figure — CALL THE RIGHT TOOL to fetch it (get_audit_file_summary for the "
    "client/profile/dates, get_trial_balance / get_audit_area / "
    "get_financial_statement for figures) and use the REAL value. Never invent a "
    "file-specific value and never write a bracketed placeholder like "
    "'[Client Name]'; if a tool cannot provide it, say so plainly. If the "
    "instruction needs no file data, draft the procedure normally without calling "
    "any tool."
)


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
    # Per-org AI budget gate (HTTP 402 when over). No-op for unmetered/anon orgs.
    await usage_meter.ensure_credits(db, payload.organization_id)

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
        custom_instruction=payload.custom_instruction,
    )

    started = time.perf_counter()
    document_filenames = list({c.document_filename for c in chunks})
    req_id = uuid.uuid4().hex
    usage: dict = {}

    # Ground the draft ONLY when the auditor typed an instruction AND we have a
    # grant — that's the case where it might reference file data ("use the client
    # name"). The default (no instruction) keeps streaming the house-style draft,
    # which never needs file data. The tool loop is synchronous → one NDJSON delta.
    instruction = (payload.custom_instruction or "").strip()
    if payload.audit_file_id and payload.copilot_grant and instruction:
        ctx = copilot_tools.CopilotContext(payload.audit_file_id, payload.copilot_grant)
        impls = copilot_tools.build_tool_impls(ctx)
        system_grounded = system + PROCEDURE_GROUNDING_ADDENDUM

        async def grounded_stream():
            yield json.dumps(
                {
                    "type": "meta",
                    "chunks_found": len(chunks),
                    "documents": document_filenames,
                    "grounded": True,
                }
            ) + "\n"
            try:
                result = await asyncio.to_thread(
                    structured.run_tool_loop,
                    system_grounded, user_prompt, copilot_tools.TOOL_SPECS, impls,
                    max_steps=6, force_first_call=False, usage_out=usage,
                )
                answer = _strip_fences(result.answer or "").strip()
                if answer:
                    yield json.dumps({"type": "delta", "text": answer}) + "\n"
                else:
                    yield json.dumps(
                        {"type": "error", "message": "Generation interrupted; please retry."}
                    ) + "\n"
            except Exception as exc:
                logger.exception("Grounded procedure error: %s", exc)
                message = (
                    qa.friendly_llm_error(exc)
                    if qa.is_quota_error(exc)
                    else "Generation interrupted; please retry."
                )
                yield json.dumps({"type": "error", "message": message}) + "\n"

            await usage_meter.record_usage(
                db, organization_id=payload.organization_id, user_id=payload.user_id,
                feature="procedure", tier="smart", usage=usage, request_id=req_id,
            )
            yield json.dumps(
                {
                    "type": "done",
                    "response_time_ms": int((time.perf_counter() - started) * 1000),
                    "documents": document_filenames,
                }
            ) + "\n"

        return StreamingResponse(
            grounded_stream(),
            media_type="application/x-ndjson",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )

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
                system, user_prompt, temperature=0.3, max_output_tokens=4096, usage_out=usage
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

        await usage_meter.record_usage(
            db, organization_id=payload.organization_id, user_id=payload.user_id,
            feature="procedure", tier="smart", usage=usage, request_id=req_id,
        )
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


@router.post("/write")
@limiter.limit(RATE_LIMIT_ASK)
async def write_assist(
    request: Request,
    payload: schemas.WriteAssistRequest,
    db: AsyncSession = Depends(get_db),
):
    """Generic AI writing assistant for ANY rich-text field. Streams clean semantic
    HTML drafted/rewritten from the field's current text + the auditor's optional
    instruction.

    Two modes, chosen by whether the FE sent file grounding:
      - GROUNDED (audit_file_id + copilot_grant present): runs the copilot
        tool-loop so the draft can read THIS file's real data when the instruction
        needs a file fact (client name, a figure, a date). Never invents — if a
        value can't be fetched it says so rather than emitting a placeholder.
      - UNGROUNDED (no grant): a pure text-craft writer, never states client
        figures (ISA 220).
    Event format mirrors /procedure exactly (NDJSON)."""
    # Same anonymous-session + per-org budget gate as the other copilot writers.
    await _load_session(db, payload.session_token)
    await usage_meter.ensure_credits(db, payload.organization_id)

    started = time.perf_counter()
    req_id = uuid.uuid4().hex
    usage: dict = {}

    grounded = bool(payload.audit_file_id and payload.copilot_grant)

    if grounded:
        # File-grounded: the tool loop is synchronous (returns the full answer),
        # so we run it off-thread and emit it as one NDJSON delta. The meta event
        # is sent first so the FE clears the editor and the stall-timer is armed.
        async def grounded_stream():
            yield json.dumps({"type": "meta", "grounded": True}) + "\n"
            try:
                result = await asyncio.to_thread(
                    copilot_tools.write_with_file,
                    payload.current_text,
                    payload.custom_instruction,
                    payload.field_label,
                    payload.audit_file_id,
                    payload.copilot_grant,
                    payload.language,
                    usage_out=usage,
                    procedure=payload.procedure,
                )
                answer = _strip_fences(result.answer or "").strip()
                if answer:
                    yield json.dumps({"type": "delta", "text": answer}) + "\n"
                else:
                    yield json.dumps(
                        {
                            "type": "error",
                            "message": "Could not read this file's data (the copilot grant may have expired). Please try again.",
                        }
                    ) + "\n"
            except Exception as exc:
                logger.exception("Grounded write-assist error: %s", exc)
                message = (
                    qa.friendly_llm_error(exc)
                    if qa.is_quota_error(exc)
                    else "Generation interrupted; please retry."
                )
                yield json.dumps({"type": "error", "message": message}) + "\n"

            await usage_meter.record_usage(
                db, organization_id=payload.organization_id, user_id=payload.user_id,
                feature="write", tier="smart", usage=usage, request_id=req_id,
            )
            yield json.dumps(
                {"type": "done", "response_time_ms": int((time.perf_counter() - started) * 1000)}
            ) + "\n"

        return StreamingResponse(
            grounded_stream(),
            media_type="application/x-ndjson",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )

    system = write_prompt.SYSTEM_PROMPT
    user_prompt = write_prompt.build_user_prompt(
        current_text=payload.current_text,
        custom_instruction=payload.custom_instruction,
        field_label=payload.field_label,
        language=payload.language,
    )

    async def event_stream():
        yield json.dumps({"type": "meta"}) + "\n"

        try:
            async for piece in structured.astream_text(
                system, user_prompt, temperature=0.4, max_output_tokens=4096, usage_out=usage
            ):
                cleaned = _strip_fences(piece)
                if cleaned:
                    yield json.dumps({"type": "delta", "text": cleaned}) + "\n"
        except Exception as exc:
            logger.exception("Write-assist streaming error: %s", exc)
            message = (
                qa.friendly_llm_error(exc)
                if qa.is_quota_error(exc)
                else "Generation interrupted; please retry."
            )
            yield json.dumps({"type": "error", "message": message}) + "\n"

        await usage_meter.record_usage(
            db, organization_id=payload.organization_id, user_id=payload.user_id,
            feature="write", tier="smart", usage=usage, request_id=req_id,
        )
        yield json.dumps(
            {"type": "done", "response_time_ms": int((time.perf_counter() - started) * 1000)}
        ) + "\n"

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


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
    await usage_meter.ensure_credits(db, payload.organization_id)

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
        await usage_meter.record(
            db, organization_id=payload.organization_id, user_id=payload.user_id,
            feature="chat", tier="smart", model="",
            input_tokens=qres.prompt_tokens, output_tokens=qres.completion_tokens,
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

    usage: dict = {}
    try:
        result = await asyncio.to_thread(
            copilot_tools.answer_about_file,
            payload.question,
            payload.audit_file_id,
            payload.copilot_grant,
            payload.language,
            kb_search=kb_search,
            usage_out=usage,
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
    await usage_meter.record_usage(
        db, organization_id=payload.organization_id, user_id=payload.user_id,
        feature="chat", tier="smart", usage=usage,
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
