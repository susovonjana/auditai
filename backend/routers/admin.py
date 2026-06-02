"""
Admin router — all /admin/* endpoints.
All routes (except /admin/login) require a valid JWT token.

NOTE: We deliberately do NOT use `from __future__ import annotations` here
because slowapi's `@limiter.limit(...)` decorator wraps endpoint functions,
and FastAPI's dependency analyzer cannot resolve stringified forward
references (like `schemas.AdminLoginRequest`) once the function lives in
the decorator's wrapping module scope.
"""
import csv
import io
import logging
import re
import shutil
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse
from sqlalchemy import select, func, desc, delete
from sqlalchemy.ext.asyncio import AsyncSession

from auth import (
    create_access_token,
    get_current_admin,
    verify_password,
)
from cache import answer_cache
from chunker import chunk_blocks
from config import (
    ALLOWED_EXTENSIONS,
    JWT_EXPIRE_HOURS,
    MAX_FILE_SIZE_BYTES,
    RATE_LIMIT_ADMIN_LOGIN,
    RATE_LIMIT_ADMIN_UPLOAD,
    UPLOAD_DIR,
)
from file_crypto import (
    encrypt_and_write,
    is_encryption_enabled,
    open_decrypted,
    remove_encrypted,
)
from file_validation import FileValidationError, validate_file
from rate_limit import limiter
from database import AsyncSessionLocal, get_db
from embeddings import embed_texts
from models import (
    AdminUser,
    Document,
    DocumentChunk,
    SearchHistory,
    UserSession,
)
from parser import (
    TextExtractionError,
    UnsupportedFileTypeError,
    extract_blocks,
)
import schemas

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])


# ===========================================================================
# Auth
# ===========================================================================
@router.post("/login", response_model=schemas.AdminLoginResponse)
@limiter.limit(RATE_LIMIT_ADMIN_LOGIN)
async def admin_login(
    request: Request,
    payload: schemas.AdminLoginRequest,
    db: AsyncSession = Depends(get_db),
):
    """Authenticate an admin and return a JWT."""
    result = await db.execute(
        select(AdminUser).where(AdminUser.username == payload.username)
    )
    admin = result.scalar_one_or_none()
    if not admin or not verify_password(payload.password, admin.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
        )

    admin.last_login_at = datetime.now(timezone.utc)
    await db.commit()

    token = create_access_token(subject=admin.username, role=admin.role)
    return schemas.AdminLoginResponse(
        access_token=token,
        token_type="bearer",
        expires_in_hours=JWT_EXPIRE_HOURS,
        username=admin.username,
        role=admin.role,
    )


@router.post("/logout", response_model=schemas.Message)
async def admin_logout(_: AdminUser = Depends(get_current_admin)):
    """Stateless JWT logout — client drops the token."""
    return schemas.Message(message="Logged out.")


# ===========================================================================
# Knowledge Base status / health
# ===========================================================================
@router.get("/status", response_model=schemas.KnowledgeBaseStatus)
async def kb_status(
    db: AsyncSession = Depends(get_db),
    _: AdminUser = Depends(get_current_admin),
):
    total_documents = (
        await db.execute(select(func.count(Document.id)))
    ).scalar() or 0
    total_chunks = (
        await db.execute(select(func.count(DocumentChunk.id)))
    ).scalar() or 0

    has_processing = (
        await db.execute(
            select(func.count(Document.id)).where(Document.status == "processing")
        )
    ).scalar() or 0
    has_error = (
        await db.execute(
            select(func.count(Document.id)).where(Document.status == "error")
        )
    ).scalar() or 0

    last_doc = (
        await db.execute(
            select(Document).order_by(desc(Document.uploaded_at)).limit(1)
        )
    ).scalar_one_or_none()

    if has_error and total_documents > 0:
        health = "red"
    elif has_processing:
        health = "yellow"
    elif total_documents == 0:
        health = "yellow"
    else:
        health = "green"

    return schemas.KnowledgeBaseStatus(
        total_documents=total_documents,
        total_chunks=total_chunks,
        total_embeddings=total_chunks,  # one embedding per chunk
        last_uploaded_filename=last_doc.filename if last_doc else None,
        last_uploaded_at=last_doc.uploaded_at if last_doc else None,
        health_status=health,
    )


# ===========================================================================
# Document upload
# ===========================================================================
@router.post("/upload", response_model=schemas.DocumentUploadResponse)
@limiter.limit(RATE_LIMIT_ADMIN_UPLOAD)
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    category: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
    admin: AdminUser = Depends(get_current_admin),
):
    """
    Upload a document. Returns IMMEDIATELY after the file is validated and
    safely stored. The heavy work (parse + chunk + embed) runs as a
    background task so the HTTP request can't time out on large PDFs.

    Frontend should poll GET /admin/documents/{id} to track progress:
        queued → parsing → embedding → active   (success)
        queued → parsing → ... → error          (failure)
    """
    filename = file.filename or "uploaded"
    filename = Path(filename).name
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="This file type is not supported. Please upload PDF, Word, Excel, or image files.",
        )

    body = bytearray()
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        body.extend(chunk)
        if len(body) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail="This file exceeds the maximum allowed size. Please split it into smaller files.",
            )
    if not body:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    doc_id = uuid.uuid4()
    stored_name = f"{doc_id}{ext}"
    plain_target: Path = UPLOAD_DIR / stored_name
    plain_target.write_bytes(bytes(body))

    try:
        validate_file(plain_target, ext)
    except FileValidationError as exc:
        plain_target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        encrypted_target = encrypt_and_write(bytes(body), plain_target)
    except Exception as exc:
        plain_target.unlink(missing_ok=True)
        logger.exception("File encryption failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to securely store the file.")

    if encrypted_target != plain_target:
        plain_target.unlink(missing_ok=True)

    document = Document(
        id=doc_id,
        filename=filename,
        file_type=ext.lstrip("."),
        category=category,
        file_path=str(encrypted_target),
        status="queued",
        total_chunks=0,
        file_size_bytes=len(body),
        uploaded_by=admin.username,
    )
    db.add(document)
    await db.commit()
    await db.refresh(document)

    # Schedule the heavy lifting to run AFTER the response is sent.
    background_tasks.add_task(_process_document_background, doc_id)

    return schemas.DocumentUploadResponse(
        id=document.id,
        filename=document.filename,
        status=document.status,
        total_chunks=0,
        message="Upload received. Processing in background — poll for status.",
    )


async def _set_status(doc_id: uuid.UUID, status: str, error: Optional[str] = None) -> None:
    """Update a document's status in a short-lived DB session."""
    async with AsyncSessionLocal() as db:
        doc = await db.get(Document, doc_id)
        if doc is None:
            return
        doc.status = status
        if error is not None:
            doc.error_message = error[:1000]
        await db.commit()


async def _process_document_background(doc_id: uuid.UUID) -> None:
    """
    Heavy upload pipeline that runs AFTER the HTTP response is sent.

    Updates Document.status as it progresses so the frontend can poll:
        queued → parsing → embedding → active | error
    """
    logger.info("[bg] Processing document %s", doc_id)

    # ---- 1. Parse ----
    try:
        await _set_status(doc_id, "parsing")
        async with AsyncSessionLocal() as db:
            doc = await db.get(Document, doc_id)
            if doc is None:
                logger.error("[bg] Document %s missing, abandoning.", doc_id)
                return
            file_path = doc.file_path

        with open_decrypted(file_path) as work_path:
            blocks, _ = extract_blocks(work_path)

        parsed_chunks = chunk_blocks(blocks)
        if not parsed_chunks:
            await _set_status(
                doc_id,
                "error",
                "Could not extract any text from this file. Verify it is not "
                "corrupted, password-protected, or a low-quality scan.",
            )
            return
    except UnsupportedFileTypeError as exc:
        await _set_status(doc_id, "error", str(exc))
        return
    except TextExtractionError as exc:
        await _set_status(doc_id, "error", str(exc))
        return
    except Exception as exc:
        logger.exception("[bg] Parsing failed for %s: %s", doc_id, exc)
        await _set_status(doc_id, "error", f"Failed to parse: {exc}")
        return

    # ---- 2. Embed ----
    try:
        await _set_status(doc_id, "embedding")
        vectors = await embed_texts([c.content for c in parsed_chunks])
    except Exception as exc:
        logger.exception("[bg] Embedding failed for %s: %s", doc_id, exc)
        await _set_status(
            doc_id, "error",
            "Failed to generate embeddings. Check the embedding service.",
        )
        return

    # ---- 3. Persist + mark active ----
    try:
        async with AsyncSessionLocal() as db:
            for pc, vec in zip(parsed_chunks, vectors):
                db.add(
                    DocumentChunk(
                        document_id=doc_id,
                        chunk_index=pc.chunk_index,
                        content=pc.content,
                        embedding=vec,
                        char_count=pc.char_count,
                        page_number=pc.page_number,
                        section_heading=pc.section_heading,
                        chunk_type=pc.chunk_type,
                    )
                )

            doc = await db.get(Document, doc_id)
            if doc is not None:
                doc.total_chunks = len(parsed_chunks)
                doc.status = "active"
                doc.error_message = None
            await db.commit()

        answer_cache.clear()
        logger.info("[bg] Indexed %s chunks for %s", len(parsed_chunks), doc_id)

    except Exception as exc:
        logger.exception("[bg] Persistence failed for %s: %s", doc_id, exc)
        await _set_status(doc_id, "error", f"Failed to store chunks: {exc}")


# ===========================================================================
# Document library
# ===========================================================================
@router.get("/documents", response_model=list[schemas.DocumentOut])
async def list_documents(
    db: AsyncSession = Depends(get_db),
    _: AdminUser = Depends(get_current_admin),
):
    result = await db.execute(
        select(Document).order_by(desc(Document.uploaded_at))
    )
    return list(result.scalars().all())


@router.get("/documents/{document_id}", response_model=schemas.DocumentOut)
async def get_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _: AdminUser = Depends(get_current_admin),
):
    """Poll this endpoint to track background processing status."""
    doc = await db.get(Document, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return doc


@router.patch("/documents/{document_id}", response_model=schemas.DocumentOut)
async def update_document(
    document_id: uuid.UUID,
    payload: schemas.DocumentUpdateRequest,
    db: AsyncSession = Depends(get_db),
    _: AdminUser = Depends(get_current_admin),
):
    document = await db.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found.")
    if payload.category is not None:
        document.category = payload.category
    await db.commit()
    await db.refresh(document)
    return document


@router.delete("/documents/{document_id}", response_model=schemas.Message)
async def delete_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _: AdminUser = Depends(get_current_admin),
):
    document = await db.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found.")

    # Remove file from disk (encrypted or plain), best effort
    try:
        remove_encrypted(document.file_path)
    except Exception:  # pragma: no cover
        logger.warning("Failed to delete file on disk for %s", document_id)

    # Cascade removes chunks
    await db.delete(document)
    await db.commit()

    # The knowledge base just changed — old cached answers may be stale.
    answer_cache.clear()

    return schemas.Message(message="Document, chunks, and embeddings removed.")


# ===========================================================================
# Search history
# ===========================================================================
@router.get("/search-history", response_model=schemas.SearchHistoryPage)
async def search_history(
    q: Optional[str] = Query(None, description="Keyword filter within question text"),
    answered: Optional[bool] = Query(None),
    session_id: Optional[uuid.UUID] = Query(None),
    date_from: Optional[datetime] = Query(None),
    date_to: Optional[datetime] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    _: AdminUser = Depends(get_current_admin),
):
    stmt = select(SearchHistory)
    count_stmt = select(func.count(SearchHistory.id))

    conditions = []
    if q:
        conditions.append(SearchHistory.question.ilike(f"%{q}%"))
    if answered is not None:
        conditions.append(SearchHistory.was_answered.is_(answered))
    if session_id is not None:
        conditions.append(SearchHistory.session_id == session_id)
    if date_from is not None:
        conditions.append(SearchHistory.asked_at >= date_from)
    if date_to is not None:
        conditions.append(SearchHistory.asked_at <= date_to)
    for cond in conditions:
        stmt = stmt.where(cond)
        count_stmt = count_stmt.where(cond)

    total = (await db.execute(count_stmt)).scalar() or 0
    rows = (
        await db.execute(
            stmt.order_by(desc(SearchHistory.asked_at))
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
    ).scalars().all()

    return schemas.SearchHistoryPage(
        items=list(rows),
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/search-history/unanswered", response_model=schemas.SearchHistoryPage
)
async def unanswered(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    _: AdminUser = Depends(get_current_admin),
):
    total = (
        await db.execute(
            select(func.count(SearchHistory.id)).where(
                SearchHistory.was_answered.is_(False)
            )
        )
    ).scalar() or 0
    rows = (
        await db.execute(
            select(SearchHistory)
            .where(SearchHistory.was_answered.is_(False))
            .order_by(desc(SearchHistory.asked_at))
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
    ).scalars().all()
    return schemas.SearchHistoryPage(
        items=list(rows),
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/search-history/export")
async def export_search_history(
    db: AsyncSession = Depends(get_db),
    _: AdminUser = Depends(get_current_admin),
):
    """Download all search history as a CSV file."""
    rows = (
        await db.execute(
            select(SearchHistory).order_by(desc(SearchHistory.asked_at))
        )
    ).scalars().all()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "asked_at",
            "session_id",
            "question",
            "was_answered",
            "response_time_ms",
            "user_feedback",
            "documents_referenced",
            "ai_answer",
        ]
    )
    for r in rows:
        writer.writerow(
            [
                r.asked_at.isoformat(),
                str(r.session_id),
                r.question,
                r.was_answered,
                r.response_time_ms,
                r.user_feedback or "",
                ";".join(r.documents_referenced or []),
                (r.ai_answer or "").replace("\n", " ")[:2000],
            ]
        )
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=auditai_search_history.csv"
        },
    )


# ===========================================================================
# Analytics
# ===========================================================================
@router.get("/analytics/summary", response_model=schemas.AnalyticsSummary)
async def analytics_summary(
    db: AsyncSession = Depends(get_db),
    _: AdminUser = Depends(get_current_admin),
):
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = now - timedelta(days=7)

    total_today = (
        await db.execute(
            select(func.count(SearchHistory.id)).where(
                SearchHistory.asked_at >= today_start
            )
        )
    ).scalar() or 0
    total_week = (
        await db.execute(
            select(func.count(SearchHistory.id)).where(
                SearchHistory.asked_at >= week_start
            )
        )
    ).scalar() or 0
    total_all = (
        await db.execute(select(func.count(SearchHistory.id)))
    ).scalar() or 0

    answered = (
        await db.execute(
            select(func.count(SearchHistory.id)).where(
                SearchHistory.was_answered.is_(True)
            )
        )
    ).scalar() or 0
    success_rate = float(answered) / float(total_all) if total_all else 0.0

    avg_rt = (
        await db.execute(
            select(func.coalesce(func.avg(SearchHistory.response_time_ms), 0))
        )
    ).scalar() or 0

    unique_sessions = (
        await db.execute(select(func.count(UserSession.id)))
    ).scalar() or 0

    # Most active hour-of-day
    hour_rows = (
        await db.execute(
            select(
                func.extract("hour", SearchHistory.asked_at).label("h"),
                func.count(SearchHistory.id).label("c"),
            ).group_by("h").order_by(desc("c"))
        )
    ).all()
    most_active_hour = int(hour_rows[0].h) if hour_rows else None

    return schemas.AnalyticsSummary(
        total_questions_today=total_today,
        total_questions_week=total_week,
        total_questions_all_time=total_all,
        answer_success_rate=round(success_rate, 4),
        avg_response_time_ms=float(avg_rt),
        unique_sessions=unique_sessions,
        most_active_hour=most_active_hour,
    )


_STOP_WORDS = {
    "the", "a", "an", "of", "and", "to", "in", "on", "is", "are", "was",
    "were", "for", "with", "by", "from", "that", "this", "as", "at", "be",
    "or", "it", "its", "what", "how", "why", "when", "where", "which",
    "do", "does", "did", "i", "you", "we", "they", "my", "our", "your",
    "can", "should", "would", "could", "may", "might", "have", "has",
    "had", "but", "if", "then", "than", "into", "about", "over", "under",
    "while", "during", "between",
}


@router.get("/analytics/top-topics", response_model=schemas.TopTopicsResponse)
async def top_topics(
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
    _: AdminUser = Depends(get_current_admin),
):
    questions = (
        await db.execute(select(SearchHistory.question))
    ).scalars().all()

    counter: Counter[str] = Counter()
    for q in questions:
        for tok in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", (q or "").lower()):
            if tok in _STOP_WORDS:
                continue
            counter[tok] += 1

    top = counter.most_common(limit)
    return schemas.TopTopicsResponse(
        topics=[schemas.TopTopic(keyword=k, count=c) for k, c in top]
    )


# ===========================================================================
# Sessions
# ===========================================================================
@router.get("/sessions", response_model=list[schemas.SessionOut])
async def list_sessions(
    db: AsyncSession = Depends(get_db),
    _: AdminUser = Depends(get_current_admin),
):
    rows = (
        await db.execute(
            select(UserSession).order_by(desc(UserSession.started_at))
        )
    ).scalars().all()
    return list(rows)


@router.get(
    "/sessions/{session_id}", response_model=list[schemas.SearchHistoryItem]
)
async def session_history(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _: AdminUser = Depends(get_current_admin),
):
    rows = (
        await db.execute(
            select(SearchHistory)
            .where(SearchHistory.session_id == session_id)
            .order_by(SearchHistory.asked_at.asc())
        )
    ).scalars().all()
    return list(rows)
