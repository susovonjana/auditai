"""
Public user-facing router.

Endpoints:
  POST /session/start            create a new chat session
  POST /ask                      ask a question, get a Markdown answer (single-shot)
  POST /ask/stream               ask a question, stream the answer (NDJSON)
  POST /feedback                 submit thumbs up / down
  GET  /session/{token}/history  this session's Q&A history
  GET  /health                   server health probe

NOTE: No `from __future__ import annotations` — slowapi's decorator
interferes with FastAPI's resolution of stringified forward references.
"""
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from config import (
    MAX_QUESTION_CHARS,
    MAX_QUESTIONS_PER_SESSION_DAY,
    RATE_LIMIT_ASK,
    RATE_LIMIT_SESSION_START,
    RATE_LIMIT_TRANSLATE,
)
from cache import answer_cache
from database import AsyncSessionLocal, get_db
from models import SearchHistory, UserSession
from rate_limit import limiter
import qa
import schemas
import usage_meter

logger = logging.getLogger(__name__)
router = APIRouter(tags=["user"])


# ---------------------------------------------------------------------------
# Session start
# ---------------------------------------------------------------------------
@router.post("/session/start", response_model=schemas.SessionStartResponse)
@limiter.limit(RATE_LIMIT_SESSION_START)
async def start_session(
    request: Request,
    body: Optional[schemas.SessionStartRequest] = Body(None),
    db: AsyncSession = Depends(get_db),
):
    token = str(uuid.uuid4())
    client_host = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    session = UserSession(
        session_token=token,
        user_identifier=(body.user_identifier if body and body.user_identifier else "anonymous"),
        ip_address=client_host,
        user_agent=user_agent,
        total_questions=0,
        user_id=body.user_id if body else None,
        organization_id=body.organization_id if body else None,
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)

    return schemas.SessionStartResponse(
        session_token=session.session_token,
        started_at=session.started_at,
    )


async def _load_session(db: AsyncSession, token: str) -> UserSession:
    res = await db.execute(
        select(UserSession).where(UserSession.session_token == token)
    )
    session = res.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session


async def _check_session_quota(db: AsyncSession, session_id) -> None:
    """
    Enforce a per-session 24-hour cap on /ask calls.

    Per-IP rate limits protect against single attackers; per-session caps
    protect against token-theft attacks where a real user's token is reused
    from many IPs.
    """
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    count = (
        await db.execute(
            select(func.count(SearchHistory.id)).where(
                SearchHistory.session_id == session_id,
                SearchHistory.asked_at >= since,
            )
        )
    ).scalar() or 0
    if count >= MAX_QUESTIONS_PER_SESSION_DAY:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"This session has reached its daily limit of "
                f"{MAX_QUESTIONS_PER_SESSION_DAY} questions. "
                "Please try again in 24 hours."
            ),
        )


def _validate_question_payload(question: str) -> None:
    """Defence-in-depth checks before paying for any compute."""
    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Empty question.")
    if len(question) > MAX_QUESTION_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"Question too long (max {MAX_QUESTION_CHARS} characters).",
        )


# ---------------------------------------------------------------------------
# Ask (non-streaming) — kept for compatibility
# ---------------------------------------------------------------------------
async def _persist_ask_history(
    history_id: uuid.UUID,
    session_id: uuid.UUID,
    question: str,
    result: "qa.QAResult",
    elapsed_ms: int,
    user_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    audit_file_id: Optional[int] = None,
) -> None:
    """Write the SearchHistory row + bump session counters AFTER the response
    has been sent. Uses a fresh AsyncSession because the request-scoped one
    is closed by the time this runs."""
    try:
        async with AsyncSessionLocal() as bg:
            history = SearchHistory(
                id=history_id,
                session_id=session_id,
                question=question,
                question_embedding=result.question_embedding or None,
                ai_answer=result.answer,
                chunks_used=result.chunks_used,
                documents_referenced=result.documents_referenced,
                similarity_scores=result.similarity_scores,
                response_time_ms=elapsed_ms,
                was_answered=result.was_answered,
                user_feedback=None,
                user_id=user_id,
                organization_id=organization_id,
                audit_file_id=audit_file_id,
                prompt_tokens=result.prompt_tokens or None,
                completion_tokens=result.completion_tokens or None,
                total_tokens=result.total_tokens or None,
            )
            bg.add(history)
            sess = await bg.get(UserSession, session_id)
            if sess is not None:
                sess.total_questions = (sess.total_questions or 0) + 1
                sess.last_active_at = datetime.now(timezone.utc)
            await bg.commit()
    except Exception as exc:
        logger.exception("Background persist of /ask history failed: %s", exc)


@router.post("/ask", response_model=schemas.AskResponse)
@limiter.limit(RATE_LIMIT_ASK)
async def ask(
    request: Request,
    payload: schemas.AskRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    _validate_question_payload(payload.question)
    session = await _load_session(db, payload.session_token)
    await _check_session_quota(db, session.id)
    # Per-org AI budget gate (only enforced when an org id is present; the
    # anonymous standalone KB still relies on the per-session cap above).
    await usage_meter.ensure_credits(db, payload.organization_id)

    started = time.perf_counter()
    try:
        result = await qa.answer_question(
            db, session.id, payload.question, language=payload.language,
        )
    except RuntimeError as exc:
        logger.exception("Q&A engine config error: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="There was a problem generating a response. Please try again.",
        )
    except Exception as exc:
        logger.exception("Q&A engine failure: %s", exc)
        if qa.is_quota_error(exc):
            raise HTTPException(
                status_code=429,
                detail=(
                    "The AI assistant is temporarily at capacity. "
                    "Please try again in a moment."
                ),
            )
        raise HTTPException(
            status_code=502,
            detail="There was a problem generating a response. Please try again.",
        )
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    # Pre-allocate the id so the client can submit feedback on this answer
    # before the row physically exists. The background write below uses
    # ON-COMMIT semantics, so feedback submitted in the first ~50ms after
    # response may briefly 404 — clients should treat that as expected and
    # retry-on-404 once.
    history_id = uuid.uuid4()
    background_tasks.add_task(
        _persist_ask_history,
        history_id,
        session.id,
        payload.question,
        result,
        elapsed_ms,
        payload.user_id,
        payload.organization_id,
    )

    if payload.organization_id:
        await usage_meter.record(
            db, organization_id=payload.organization_id, user_id=payload.user_id,
            feature="ask", tier="smart", model="",
            input_tokens=result.prompt_tokens, output_tokens=result.completion_tokens,
        )

    return schemas.AskResponse(
        answer=result.answer,
        history_id=history_id,
        was_answered=result.was_answered,
        response_time_ms=elapsed_ms,
        documents_referenced=result.document_filenames,
        sources=[
            schemas.Source(document=s.document, page=s.page, section=s.section)
            for s in result.sources
        ],
        confidence=result.confidence,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_tokens=result.total_tokens,
    )


# ---------------------------------------------------------------------------
# Ask (streaming)
#
# Emits newline-delimited JSON over a single HTTP response. Each line is one
# JSON object:
#
#   {"type": "meta",  "documents": ["ISA_315.pdf", ...]}
#   {"type": "delta", "text": "## From your..."}
#   {"type": "delta", "text": "knowledge base..."}
#   ...
#   {"type": "done",  "history_id": "...", "was_answered": true,
#    "response_time_ms": 1234}
# ---------------------------------------------------------------------------
@router.post("/ask/stream")
@limiter.limit(RATE_LIMIT_ASK)
async def ask_stream(
    request: Request,
    payload: schemas.AskRequest,
    db: AsyncSession = Depends(get_db),
):
    _validate_question_payload(payload.question)
    session = await _load_session(db, payload.session_token)
    await _check_session_quota(db, session.id)
    await usage_meter.ensure_credits(db, payload.organization_id)

    started = time.perf_counter()
    try:
        preamble = await qa.prepare_stream(
            db, session.id, payload.question, language=payload.language,
        )
    except RuntimeError as exc:
        logger.exception("Stream preamble config error: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="There was a problem generating a response. Please try again.",
        )

    session_id = session.id
    question = payload.question
    caller_user_id = payload.user_id
    caller_organization_id = payload.organization_id

    async def event_stream():
        accumulated: list[str] = []
        document_filenames = list({c.document_filename for c in preamble.chunks})
        sources_payload = [
            {"document": s.document, "page": s.page, "section": s.section}
            for s in qa._build_sources(preamble.chunks)
        ]
        confidence_value, _ = qa._confidence_score(preamble.chunks)

        # Emit metadata first
        meta = {
            "type": "meta",
            "documents": document_filenames,
            "sources": sources_payload,
            "confidence": round(confidence_value, 4),
            "chunks_found": len(preamble.chunks),
        }
        yield json.dumps(meta) + "\n"

        # Stream tokens
        try:
            async for piece in qa.stream_answer(preamble, question):
                accumulated.append(piece)
                yield json.dumps({"type": "delta", "text": piece}) + "\n"
        except Exception as exc:
            logger.exception("Streaming error: %s", exc)
            yield json.dumps(
                {"type": "error", "message": "Stream interrupted; please retry."}
            ) + "\n"

        full_answer = "".join(accumulated).strip() or qa.EMPTY_KB_TEXT
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        was_answered = preamble.was_answered_initial and not qa.looks_like_no_answer(
            full_answer, preamble.language,
        )

        # Cache positive answers for ~1 hour so repeats are instant.
        # Skip if this WAS a cache hit (cached_reply set) — already in cache.
        # Skip small-talk — they're already instant.
        if (
            was_answered
            and not preamble.smalltalk_reply
            and not preamble.cached_reply
            and full_answer
        ):
            answer_cache.set(
                preamble.question or question,
                preamble.language,
                {
                    "answer": full_answer,
                    "was_answered": was_answered,
                    "chunks_used": [str(c.chunk_id) for c in preamble.chunks],
                    "documents_referenced": list(
                        {str(c.document_id) for c in preamble.chunks}
                    ),
                    "document_filenames": document_filenames,
                    "similarity_scores": [
                        round(c.similarity, 4) for c in preamble.chunks
                    ],
                    "sources": sources_payload,
                    "confidence": round(confidence_value, 4),
                },
            )

        # Persist to search_history in a fresh session — the original `db`
        # session is bound to the HTTP request and may be closing.
        history_id: Optional[str] = None
        try:
            async with AsyncSessionLocal() as bg:
                history = SearchHistory(
                    session_id=session_id,
                    question=question,
                    question_embedding=preamble.question_embedding or None,
                    ai_answer=full_answer,
                    chunks_used=[str(c.chunk_id) for c in preamble.chunks],
                    documents_referenced=list({str(c.document_id) for c in preamble.chunks}),
                    similarity_scores=[round(c.similarity, 4) for c in preamble.chunks],
                    response_time_ms=elapsed_ms,
                    was_answered=was_answered,
                    user_feedback=None,
                    user_id=caller_user_id,
                    organization_id=caller_organization_id,
                    prompt_tokens=preamble.usage.get("prompt") or None,
                    completion_tokens=preamble.usage.get("completion") or None,
                    total_tokens=preamble.usage.get("total") or None,
                )
                bg.add(history)

                # Bump session counters
                sess = await bg.get(UserSession, session_id)
                if sess is not None:
                    sess.total_questions = (sess.total_questions or 0) + 1
                    sess.last_active_at = datetime.now(timezone.utc)

                await bg.commit()
                await bg.refresh(history)
                history_id = str(history.id)
        except Exception as exc:
            logger.exception("Failed to persist streamed answer: %s", exc)

        # Meter org-scoped streamed answers against the monthly budget (fresh
        # session — the request db is closing). No-op for anonymous traffic.
        if caller_organization_id:
            try:
                async with AsyncSessionLocal() as mb:
                    await usage_meter.record(
                        mb, organization_id=caller_organization_id, user_id=caller_user_id,
                        feature="ask", tier="smart", model="",
                        input_tokens=int(preamble.usage.get("prompt", 0) or 0),
                        output_tokens=int(preamble.usage.get("completion", 0) or 0),
                    )
            except Exception:
                pass

        yield json.dumps(
            {
                "type": "done",
                "history_id": history_id,
                "was_answered": was_answered,
                "response_time_ms": elapsed_ms,
                "documents": document_filenames,
                "sources": sources_payload,
                "confidence": round(confidence_value, 4),
                "prompt_tokens": int(preamble.usage.get("prompt", 0) or 0),
                "completion_tokens": int(preamble.usage.get("completion", 0) or 0),
                "total_tokens": int(preamble.usage.get("total", 0) or 0),
            }
        ) + "\n"

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------
@router.post("/feedback", response_model=schemas.Message)
async def feedback(
    payload: schemas.FeedbackRequest,
    db: AsyncSession = Depends(get_db),
):
    history = await db.get(SearchHistory, payload.history_id)
    if not history:
        raise HTTPException(status_code=404, detail="History entry not found.")
    history.user_feedback = payload.feedback
    await db.commit()
    return schemas.Message(message="Thanks for the feedback.")


# ---------------------------------------------------------------------------
# Translate (EN ↔ AR) — translates an arbitrary chunk of text. Used by the
# chat UI's "Translate" button under each AI response.
# ---------------------------------------------------------------------------
@router.post("/translate", response_model=schemas.TranslateResponse)
@limiter.limit(RATE_LIMIT_TRANSLATE)
async def translate(
    request: Request,
    payload: schemas.TranslateRequest,
    db: AsyncSession = Depends(get_db),
):
    await _load_session(db, payload.session_token)
    started = time.perf_counter()
    try:
        translated = await qa.translate_text(payload.text, payload.target_language)
    except RuntimeError as exc:
        logger.exception("Translate config error: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Translation is temporarily unavailable. Please try again.",
        )
    except Exception as exc:
        logger.exception("Translate failure: %s", exc)
        if qa.is_quota_error(exc):
            raise HTTPException(
                status_code=429,
                detail="Translation is temporarily at capacity. Please try again in a minute.",
            )
        raise HTTPException(
            status_code=502,
            detail="There was a problem translating that response. Please try again.",
        )
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return schemas.TranslateResponse(
        translated_text=translated,
        target_language=payload.target_language,
        response_time_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# Session history
# ---------------------------------------------------------------------------
@router.get(
    "/session/{token}/history",
    response_model=list[schemas.SearchHistoryItem],
)
async def session_history(
    token: str,
    db: AsyncSession = Depends(get_db),
):
    session = await _load_session(db, token)
    # Strict per-user privacy: return ONLY the session owner's turns, so a shared
    # browser/token can never surface another user's conversation.
    conditions = [SearchHistory.session_id == session.id]
    if session.user_id:
        conditions.append(SearchHistory.user_id == session.user_id)
    rows = (
        await db.execute(
            select(SearchHistory)
            .where(*conditions)
            .order_by(SearchHistory.asked_at.asc())
        )
    ).scalars().all()
    return [schemas.SearchHistoryItem.model_validate(row) for row in rows]


# ---------------------------------------------------------------------------
# Session token usage (today, UTC)
# Used by the chat widget to render "🔢 You used N tokens today" under the
# header. Scoped to the user_id when known so the count carries across
# sessions for the same logged-in user; otherwise falls back to this session.
# ---------------------------------------------------------------------------
@router.get(
    "/session/{token}/usage",
    response_model=schemas.UserUsageResponse,
)
async def session_usage(
    token: str,
    db: AsyncSession = Depends(get_db),
):
    session = await _load_session(db, token)
    midnight = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0,
    )
    if session.user_id:
        stmt = select(func.coalesce(func.sum(SearchHistory.total_tokens), 0)).where(
            SearchHistory.user_id == session.user_id,
            SearchHistory.asked_at >= midnight,
        )
    else:
        stmt = select(func.coalesce(func.sum(SearchHistory.total_tokens), 0)).where(
            SearchHistory.session_id == session.id,
            SearchHistory.asked_at >= midnight,
        )
    total = (await db.execute(stmt)).scalar() or 0
    # Org AI-credit standing for the meter shown before generating. Keyed off the
    # session's organization_id; returns metered=False when metering is off.
    org_credit = await usage_meter.get_org_credit_summary(db, session.organization_id)
    return schemas.UserUsageResponse(
        total_tokens_today=int(total),
        org_credit=schemas.OrgCreditSummary(**org_credit),
    )
