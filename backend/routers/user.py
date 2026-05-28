"""
Public user-facing router.

Endpoints:
  POST /session/start            create a new chat session
  POST /ask                      ask a question, get a Markdown answer
  POST /feedback                 submit thumbs up / down
  GET  /session/{token}/history  this session's Q&A history
  GET  /health                   server health probe
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models import SearchHistory, UserSession
import qa
import schemas

logger = logging.getLogger(__name__)
router = APIRouter(tags=["user"])


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@router.get("/health")
async def health():
    return {"status": "ok", "service": "AuditAI", "time": datetime.now(timezone.utc).isoformat()}


# ---------------------------------------------------------------------------
# Session start
# ---------------------------------------------------------------------------
@router.post("/session/start", response_model=schemas.SessionStartResponse)
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


# ---------------------------------------------------------------------------
# Ask
# ---------------------------------------------------------------------------
@router.post("/ask", response_model=schemas.AskResponse)
async def ask(
    payload: schemas.AskRequest,
    db: AsyncSession = Depends(get_db),
):
    session = await _load_session(db, payload.session_token)

    started = time.perf_counter()
    try:
        result = await qa.answer_question(db, payload.question)
    except RuntimeError as exc:
        # Missing API keys etc.
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

    # Bump session counters
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
