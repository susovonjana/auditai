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

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from config import (
    MAX_QUESTION_CHARS,
    MAX_QUESTIONS_PER_SESSION_DAY,
    RATE_LIMIT_ASK,
    RATE_LIMIT_SESSION_START,
)
from database import AsyncSessionLocal, get_db
from models import SearchHistory, UserSession
from rate_limit import limiter
import qa
import schemas

logger = logging.getLogger(__name__)
router = APIRouter(tags=["user"])


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@router.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "AuditAI",
        "time": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Session start
# ---------------------------------------------------------------------------
@router.post("/session/start", response_model=schemas.SessionStartResponse)
@limiter.limit(RATE_LIMIT_SESSION_START)
async def start_session(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    token = str(uuid.uuid4())
    client_host = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    session = UserSession(
        session_token=token,
        user_identifier="anonymous",
        ip_address=client_host,
        user_agent=user_agent,
        total_questions=0,
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
@router.post("/ask", response_model=schemas.AskResponse)
@limiter.limit(RATE_LIMIT_ASK)
async def ask(
    request: Request,
    payload: schemas.AskRequest,
    db: AsyncSession = Depends(get_db),
):
    _validate_question_payload(payload.question)
    session = await _load_session(db, payload.session_token)
    await _check_session_quota(db, session.id)

    started = time.perf_counter()
    try:
        result = await qa.answer_question(db, session.id, payload.question)
    except RuntimeError as exc:
        logger.exception("Q&A engine config error: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="There was a problem generating a response. Please try again.",
        )
    except Exception as exc:
        logger.exception("Q&A engine failure: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="There was a problem generating a response. Please try again.",
        )
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    history = SearchHistory(
        session_id=session.id,
        question=payload.question,
        question_embedding=result.question_embedding or None,
        ai_answer=result.answer,
        chunks_used=result.chunks_used,
        documents_referenced=result.documents_referenced,
        similarity_scores=result.similarity_scores,
        response_time_ms=elapsed_ms,
        was_answered=result.was_answered,
        user_feedback=None,
    )
    db.add(history)

    session.total_questions = (session.total_questions or 0) + 1
    session.last_active_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(history)

    return schemas.AskResponse(
        answer=result.answer,
        history_id=history.id,
        was_answered=result.was_answered,
        response_time_ms=elapsed_ms,
        documents_referenced=result.document_filenames,
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

    started = time.perf_counter()
    try:
        preamble = await qa.prepare_stream(db, session.id, payload.question)
    except RuntimeError as exc:
        logger.exception("Stream preamble config error: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="There was a problem generating a response. Please try again.",
        )

    session_id = session.id
    question = payload.question

    async def event_stream():
        accumulated: list[str] = []
        document_filenames = list({c.document_filename for c in preamble.chunks})

        # Emit metadata first
        meta = {
            "type": "meta",
            "documents": document_filenames,
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
        was_answered = preamble.was_answered_initial and (
            "knowledge base does not contain" not in full_answer.lower()
        )

        # Persist to search_history in a fresh session — the original `db`
        # session is bound to the HTTP request and may be closing.
        history_id: str | None = None
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

        yield json.dumps(
            {
                "type": "done",
                "history_id": history_id,
                "was_answered": was_answered,
                "response_time_ms": elapsed_ms,
                "documents": document_filenames,
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
    rows = (
        await db.execute(
            select(SearchHistory)
            .where(SearchHistory.session_id == session.id)
            .order_by(SearchHistory.asked_at.asc())
        )
    ).scalars().all()
    return list(rows)
