"""
Persistence for document insights (``aura_document_insights``) — what the
document reader extracted from ONE uploaded document, cached per
(audit file, document, UI language) and fingerprinted by ``content_key``.

Why a store: reading a 20-page scan costs seconds and tokens; the auditor
re-opens documents often; the chat / response writer want to CITE a document
without re-OCRing it. Rows live only in auditai's Postgres (nothing is written
back to 1audit).

Two entry styles:
  * async helpers for the FastAPI routes (they own a session)
  * ``persist_from_thread`` for the agent runtime, whose compute steps run in a
    worker thread via ``asyncio.to_thread`` — it schedules the async upsert on
    the main event loop (captured at startup by ``set_main_loop``).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import select, or_
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

import config
from database import AsyncSessionLocal
from models import DocumentInsightRow

logger = logging.getLogger(__name__)

_MAIN_LOOP: Optional[asyncio.AbstractEventLoop] = None


def set_main_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Called once at startup (inside the running loop) so worker threads can
    schedule DB writes on it."""
    global _MAIN_LOOP
    _MAIN_LOOP = loop


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------
def row_to_payload(row: DocumentInsightRow, *, include_text: bool = False) -> Dict[str, Any]:
    out = {
        "id": str(row.id),
        "audit_file_id": row.audit_file_id,
        "document_id": row.document_id,
        "document_reference": row.document_reference,
        "document_name": row.document_name,
        "mime_type": row.mime_type,
        "language": row.language,
        "content_key": row.content_key,
        "doc_type": row.doc_type,
        "summary_short": row.summary_short,
        "insight": row.insight or {},
        "read_method": row.read_method,
        "pages": row.pages,
        "model": row.model,
        "created_by": row.created_by,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }
    if include_text:
        out["extracted_text"] = row.extracted_text or ""
    return out


def row_to_list_item(row: DocumentInsightRow) -> Dict[str, Any]:
    """Compact shape for the documents LIST (badge per row): no text, no full insight."""
    ins = row.insight or {}
    # Only REAL warnings count for the list badge: checks that failed (totals,
    # VAT rate, period, VAT number). Observational red flags stay in the panel.
    warn = [c for c in (ins.get("checks") or []) if isinstance(c, dict) and c.get("status") == "warning"]
    return {
        "document_id": row.document_id,
        "document_reference": row.document_reference,
        "language": row.language,
        "content_key": row.content_key,
        "doc_type": row.doc_type,
        "title": ins.get("title"),
        "summary_short": row.summary_short,
        "confidence": ins.get("confidence"),
        "warnings_count": len(warn),
        "red_flags_count": len(ins.get("red_flags") or []),
        "suggested_working_papers": (ins.get("audit_relevance") or {}).get("suggested_working_papers") or [],
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------
async def get_insight(
    db: AsyncSession, *, audit_file_id: int, document_id: int, language: Optional[str] = None
) -> Optional[DocumentInsightRow]:
    """The row for this document — in the requested language when given; any
    language otherwise (text is language-agnostic, used by Q&A / tools)."""
    q = select(DocumentInsightRow).where(
        DocumentInsightRow.audit_file_id == int(audit_file_id),
        DocumentInsightRow.document_id == int(document_id),
    )
    if language:
        q = q.where(DocumentInsightRow.language == language)
    q = q.order_by(DocumentInsightRow.updated_at.desc()).limit(1)
    return (await db.execute(q)).scalar_one_or_none()


async def list_insights(
    db: AsyncSession, *, audit_file_id: int, organization_id: Optional[str], language: Optional[str] = None
) -> List[Dict[str, Any]]:
    """One compact entry per document (preferring the requested language)."""
    q = select(DocumentInsightRow).where(DocumentInsightRow.audit_file_id == int(audit_file_id))
    if organization_id:
        q = q.where(DocumentInsightRow.organization_id == str(organization_id))
    q = q.order_by(DocumentInsightRow.updated_at.desc())
    rows = (await db.execute(q)).scalars().all()
    best: Dict[int, DocumentInsightRow] = {}
    for r in rows:
        cur = best.get(r.document_id)
        if cur is None or (language and r.language == language and cur.language != language):
            best[r.document_id] = r
    return [row_to_list_item(r) for r in best.values()]


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------
async def upsert_insight(
    db: AsyncSession,
    *,
    organization_id: Optional[str],
    audit_file_id: int,
    document_id: int,
    document_reference: Optional[str],
    document_name: Optional[str],
    mime_type: Optional[str],
    language: str,
    content_key: str,
    insight: Dict[str, Any],
    extracted_text: Optional[str],
    read_method: Optional[str],
    pages: Optional[int],
    model: Optional[str],
    created_by: Optional[str],
) -> DocumentInsightRow:
    values = dict(
        organization_id=(str(organization_id) if organization_id else None),
        audit_file_id=int(audit_file_id),
        document_id=int(document_id),
        document_reference=document_reference,
        document_name=document_name,
        mime_type=mime_type,
        language=(language or "en"),
        content_key=content_key,
        doc_type=(str(insight.get("doc_type") or "")[:64] or None),
        summary_short=insight.get("summary_short"),
        insight=insight,
        extracted_text=(extracted_text or "")[: int(config.DOC_READER_TEXT_STORE_CHARS)],
        read_method=read_method,
        pages=pages,
        model=model,
        created_by=(str(created_by) if created_by else None),
    )
    stmt = pg_insert(DocumentInsightRow).values(**values)
    update_cols = {k: v for k, v in values.items() if k not in ("audit_file_id", "document_id", "language")}
    stmt = stmt.on_conflict_do_update(
        index_elements=["audit_file_id", "document_id", "language"],
        set_=update_cols,
    )
    await db.execute(stmt)
    await db.commit()
    row = await get_insight(db, audit_file_id=audit_file_id, document_id=document_id, language=language)
    assert row is not None
    return row


def persist_from_thread(timeout: float = 30.0, **kwargs: Any) -> Optional[Dict[str, Any]]:
    """Upsert from a worker thread (agent compute step): schedules the async write
    on the main loop and waits. Best-effort — logs and returns None on failure so
    a persistence hiccup never fails the read itself."""
    loop = _MAIN_LOOP
    if loop is None or loop.is_closed():
        logger.warning("document_insight_store: main loop not set — insight not persisted")
        return None

    async def _run() -> Dict[str, Any]:
        async with AsyncSessionLocal() as db:
            row = await upsert_insight(db, **kwargs)
            return row_to_payload(row)

    try:
        return asyncio.run_coroutine_threadsafe(_run(), loop).result(timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        logger.warning("document_insight_store: persist failed: %s", exc)
        return None


def lookup_cached_sync(
    *, audit_file_id: int, document_id: Any, content_key: str, language: Optional[str] = None, timeout: float = 15.0
) -> Optional[Dict[str, Any]]:
    """From a worker thread (agent compute step): the stored reading for this
    document IF its source is unchanged (same content_key). Prefers the viewer's
    language, falls back to any. None on miss / mismatch / any error."""
    loop = _MAIN_LOOP
    if loop is None or loop.is_closed() or document_id is None:
        return None

    async def _run() -> Optional[Dict[str, Any]]:
        async with AsyncSessionLocal() as db:
            row = await get_insight(db, audit_file_id=audit_file_id, document_id=int(document_id), language=language)
            if row is None or (language and row.content_key != content_key):
                row = await get_insight(db, audit_file_id=audit_file_id, document_id=int(document_id))
            if row is None or row.content_key != content_key:
                return None
            return row_to_payload(row)

    try:
        return asyncio.run_coroutine_threadsafe(_run(), loop).result(timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        logger.info("document_insight_store: cached lookup failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Tool bridge (chat / write loops): "what does invoice INV-123 say?"
# ---------------------------------------------------------------------------
async def lookup_for_tool(
    audit_file_id: int, query: str, *, language: Optional[str] = None, text_chars: int = 6000
) -> Dict[str, Any]:
    """Find ONE read document on this file by reference or (partial) name and
    return its insight + a text excerpt — the payload the ``get_document_insight``
    copilot tool hands the model. Own session (called via run_coroutine_threadsafe)."""
    q = (query or "").strip()
    async with AsyncSessionLocal() as db:
        base = select(DocumentInsightRow).where(DocumentInsightRow.audit_file_id == int(audit_file_id))
        rows: List[DocumentInsightRow] = []
        if q:
            like = f"%{q}%"
            rows = (await db.execute(
                base.where(or_(
                    DocumentInsightRow.document_reference.ilike(q),
                    DocumentInsightRow.document_reference.ilike(like),
                    DocumentInsightRow.document_name.ilike(like),
                    DocumentInsightRow.summary_short.ilike(like),
                )).order_by(DocumentInsightRow.updated_at.desc()).limit(5)
            )).scalars().all()
        if not rows:
            # nothing matched → tell the model what HAS been read so it can pick
            all_rows = (await db.execute(base.order_by(DocumentInsightRow.updated_at.desc()).limit(50))).scalars().all()
            return {
                "found": False,
                "hint": ("No read document matches that reference/name. Documents that have an AI reading on this file "
                         "are listed below; others must be opened in All documents and summarised first."),
                "available": [
                    {"reference": r.document_reference, "name": r.document_name, "doc_type": r.doc_type}
                    for r in all_rows
                ],
            }
        # prefer the viewer's language when the same document was read in both
        rows.sort(key=lambda r: (0 if (language and r.language == language) else 1))
        best = rows[0]
        ins = dict(best.insight or {})
        return {
            "found": True,
            "document": {"reference": best.document_reference, "name": best.document_name,
                         "document_id": best.document_id, "mime_type": best.mime_type},
            "insight": {k: ins.get(k) for k in (
                "doc_type", "title", "summary_short", "summary", "parties", "dates", "amounts", "currency",
                "line_items", "references", "key_facts", "audit_relevance", "red_flags", "checks", "confidence",
            )},
            "text_excerpt": (best.extracted_text or "")[:text_chars],
            "other_matches": [
                {"reference": r.document_reference, "name": r.document_name} for r in rows[1:]
            ],
        }
