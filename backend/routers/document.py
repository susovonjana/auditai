"""
Document reader router — the per-document AI summary behind the "AI summary"
panel in an audit file's All documents screen (and inside a working paper's
supporting documents, which share the same preview modal).

  POST /copilot/document/insight    read ONE document → NDJSON progress events,
                                    then {"type":"result", insight …}; cached per
                                    (file, document, language) and re-read only when
                                    the source changed or refresh=true
  POST /copilot/document/ask        follow-up question about a read document → NDJSON
                                    text deltas (grounded in the stored text)
  POST /copilot/document/insights   compact list of read documents on a file (badges)

Auth mirrors the other copilot routes: anonymous auditai session + a FRESH
copilot grant validated locally; 1audit-be enforces the grant again on the
content fetch. Per-org credits are checked up-front and metered after.

Feature gate: the grant must have been minted for the ``document_ai`` feature
(1audit-be's FeatureAccess — prime-admin status, org/user allowlist, plan —
runs at mint time), and ``/ask`` additionally for its ``ask_document``
sub-feature. A grant minted for another feature (chat, agent, …) is refused
with 403, so the prime admin's switches fully control this surface.

NOTE: No `from __future__ import annotations` — slowapi's decorator interferes
with FastAPI's resolution of stringified forward references (as in copilot.py).
"""
import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

import config
from config import RATE_LIMIT_ASK
from database import AsyncSessionLocal, get_db
from rate_limit import limiter
from routers.user import _load_session
import copilot_tools
import document_insight_store as store
import document_reader
import qa
import schemas
import structured
import usage_meter
from prompts import document_reader as prompts

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/copilot/document", tags=["copilot-document"])

_HEADERS = {"X-Accel-Buffering": "no", "Cache-Control": "no-cache"}
_FEATURE = "document_reader"          # usage-ledger feature name (billing)
GRANT_FEATURE = "document_ai"         # 1audit feature key the grant must carry
GRANT_SUB_FEATURE_ASK = "ask_document"


def _gate() -> None:
    if not config.DOC_READER_ENABLED:
        raise HTTPException(status_code=403, detail="The AI document reader is not enabled.")


def _validate_grant(
    audit_file_id: int, grant: str, *, sub_feature: Optional[str] = None
) -> copilot_tools.CopilotContext:
    """401 for a bad/expired grant; 403 for a grant minted for another feature
    (or without the required sub-feature) — the prime-admin gate, enforced here."""
    ctx = copilot_tools.CopilotContext(audit_file_id, grant)
    try:
        ctx.validate_grant_local()
    except copilot_tools.CopilotGrantError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    try:
        ctx.require_feature(GRANT_FEATURE, sub_feature)
    except copilot_tools.CopilotGrantError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    return ctx


def _ev(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str) + "\n"


async def _file_context(ctx: copilot_tools.CopilotContext) -> Dict[str, Any]:
    """Best-effort engagement context (period, client, currency) + the WP list,
    fetched concurrently; a failure just means less context, never an error."""
    async def _get(endpoint: str) -> Any:
        try:
            return await asyncio.to_thread(ctx.get, endpoint)
        except Exception as exc:  # noqa: BLE001
            logger.info("document reader: context fetch %s failed: %s", endpoint, exc)
            return None

    summary, wps = await asyncio.gather(_get("summary"), _get("working_papers"))
    fc: Dict[str, Any] = {}
    if isinstance(summary, dict) and not summary.get("error"):
        for k in ("name", "client", "sector", "currency", "period_start", "period_end"):
            if summary.get(k):
                fc[k] = summary[k]
    wp_list = []
    if isinstance(wps, dict) and isinstance(wps.get("working_papers"), list):
        wp_list = [w for w in wps["working_papers"] if isinstance(w, dict)][:150]
    return {"file_context": fc, "working_papers": wp_list}


# ---------------------------------------------------------------------------
# POST /copilot/document/insight
# ---------------------------------------------------------------------------
@router.post("/insight")
@limiter.limit(RATE_LIMIT_ASK)
async def document_insight(
    request: Request,
    payload: schemas.DocumentInsightRequest,
    db: AsyncSession = Depends(get_db),
):
    _gate()
    await _load_session(db, payload.session_token)
    if payload.document_id is None and not payload.document_reference:
        raise HTTPException(status_code=422, detail="document_id or document_reference is required.")
    ctx = _validate_grant(payload.audit_file_id, payload.copilot_grant)
    if not payload.cached_only:
        await usage_meter.ensure_credits(db, payload.organization_id)

    started = time.perf_counter()
    req_id = uuid.uuid4().hex
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def progress(stage: str, message: str) -> None:
        # called from the worker thread → hop back onto the loop
        loop.call_soon_threadsafe(queue.put_nowait, {"type": "progress", "stage": stage, "message": message})

    async def stream():
        usage: dict = {}
        try:
            # 1) resolve the document (metadata + presigned URL; no bytes yet)
            yield _ev({"type": "progress", "stage": "fetching", "message": "Locating the document in 1audit…"})
            try:
                meta = await asyncio.to_thread(
                    document_reader.fetch_document_meta, ctx,
                    document_id=payload.document_id, reference=payload.document_reference,
                )
            except document_reader.DocumentReaderError as exc:
                logger.warning("document reader: resolve failed (file %s, doc %s/%s): %s",
                               payload.audit_file_id, payload.document_id, payload.document_reference, exc)
                yield _ev({"type": "error", "message": str(exc)})
                return
            doc_id = meta.get("document_id") or payload.document_id
            key = document_reader.content_key_for(meta)

            # 2) cache check — same source + same language → instant
            # NOTE: the request-scoped `db` is closed before a StreamingResponse body
            # runs (FastAPI >= 0.106), so every DB access inside this generator opens
            # its own short-lived session.
            cached = None
            if doc_id is not None:
                async with AsyncSessionLocal() as sdb:
                    cached = await store.get_insight(
                        sdb, audit_file_id=payload.audit_file_id, document_id=int(doc_id), language=payload.language
                    )
            if cached is not None and cached.content_key == key and not payload.refresh:
                yield _ev({"type": "result", "cached": True, "stale": False,
                           "document": document_reader._compact_meta(meta), **store.row_to_payload(cached)})
                yield _ev({"type": "done", "response_time_ms": int((time.perf_counter() - started) * 1000)})
                return
            if payload.cached_only:
                yield _ev({"type": "result", "cached": False, "insight": None,
                           "stale": bool(cached is not None),
                           "document": document_reader._compact_meta(meta)})
                yield _ev({"type": "done", "response_time_ms": int((time.perf_counter() - started) * 1000)})
                return

            # 3) read for real — engagement context first (cheap, concurrent)
            context = await _file_context(ctx)
            task = asyncio.create_task(asyncio.to_thread(
                document_reader.build_insight, ctx,
                document_id=payload.document_id, reference=payload.document_reference,
                language=payload.language, file_context=context["file_context"],
                working_papers=context["working_papers"], progress=progress,
                usage_out=usage, meta=meta,
            ))
            # relay progress events while the thread works
            while not task.done():
                try:
                    ev = await asyncio.wait_for(queue.get(), timeout=0.5)
                    yield _ev(ev)
                except asyncio.TimeoutError:
                    continue
            while not queue.empty():
                yield _ev(queue.get_nowait())
            try:
                result = task.result()
            except document_reader.DocumentReaderError as exc:
                logger.warning("document reader: read failed (file %s, doc %s): %s",
                               payload.audit_file_id, doc_id, exc)
                yield _ev({"type": "error", "message": str(exc)})
                return
            except Exception as exc:  # noqa: BLE001
                logger.exception("document reader failed: %s", exc)
                msg = qa.friendly_llm_error(exc) if qa.is_quota_error(exc) else "Reading the document failed. Please retry."
                yield _ev({"type": "error", "message": msg})
                return

            # 4) persist (best-effort) and emit
            yield _ev({"type": "progress", "stage": "saving", "message": "Saving the summary…"})
            payload_out: Dict[str, Any]
            try:
                async with AsyncSessionLocal() as sdb:
                    row = await store.upsert_insight(
                    sdb,
                    organization_id=payload.organization_id,
                    audit_file_id=payload.audit_file_id,
                    document_id=int(doc_id) if doc_id is not None else 0,
                    document_reference=meta.get("reference"),
                    document_name=meta.get("name"),
                    mime_type=meta.get("mime_type"),
                    language=payload.language,
                    content_key=result["content_key"],
                    insight=result["insight"],
                    extracted_text=result.get("extracted_text"),
                    read_method=result.get("read_method"),
                    pages=result.get("pages"),
                    model=str(usage.get("model") or ""),
                    created_by=payload.user_id,
                    )
                payload_out = store.row_to_payload(row)
            except Exception as exc:  # noqa: BLE001 — never lose the result over a cache write
                logger.warning("document insight persist failed: %s", exc)
                payload_out = {
                    "document_id": doc_id, "document_reference": meta.get("reference"),
                    "document_name": meta.get("name"), "language": payload.language,
                    "content_key": result["content_key"], "doc_type": result["insight"].get("doc_type"),
                    "summary_short": result["insight"].get("summary_short"), "insight": result["insight"],
                    "read_method": result.get("read_method"), "pages": result.get("pages"),
                }
            # cost transparency: tokens + credits this read spent (0 for a cache hit)
            tokens = {"input": int(usage.get("input", 0) or 0), "output": int(usage.get("output", 0) or 0)}
            credits = usage_meter.credits_for("smart", tokens["input"], tokens["output"]) if usage else 0
            yield _ev({"type": "result", "cached": False, "stale": False,
                       "document": result["document"], "tokens": tokens, "credits": credits, **payload_out})
        finally:
            if usage:
                try:
                    async with AsyncSessionLocal() as mb:
                        await usage_meter.record_usage(
                            mb, organization_id=payload.organization_id, user_id=payload.user_id,
                            feature=_FEATURE, tier="smart", usage=usage, request_id=req_id,
                        )
                except Exception:  # pragma: no cover - metering is best-effort
                    pass
        yield _ev({"type": "done", "response_time_ms": int((time.perf_counter() - started) * 1000)})

    return StreamingResponse(stream(), media_type="application/x-ndjson", headers=_HEADERS)


# ---------------------------------------------------------------------------
# POST /copilot/document/ask
# ---------------------------------------------------------------------------
@router.post("/ask")
@limiter.limit(RATE_LIMIT_ASK)
async def document_ask(
    request: Request,
    payload: schemas.DocumentAskRequest,
    db: AsyncSession = Depends(get_db),
):
    _gate()
    await _load_session(db, payload.session_token)
    _validate_grant(payload.audit_file_id, payload.copilot_grant, sub_feature=GRANT_SUB_FEATURE_ASK)
    await usage_meter.ensure_credits(db, payload.organization_id)

    row = await store.get_insight(db, audit_file_id=payload.audit_file_id, document_id=payload.document_id, language=payload.language)
    if row is None:
        row = await store.get_insight(db, audit_file_id=payload.audit_file_id, document_id=payload.document_id)
    if row is None:
        raise HTTPException(status_code=409, detail="Summarise this document with AI first, then ask questions about it.")

    started = time.perf_counter()
    req_id = uuid.uuid4().hex
    system = prompts.ASK_SYSTEM_PROMPT
    text_excerpt = (row.extracted_text or "")[: int(config.DOC_READER_ASK_CONTEXT_CHARS)]
    # Two content blocks: the document context (identical for every question about
    # this document) carries a prompt-cache breakpoint, so follow-up questions in
    # the cache window pay ~10% for the bulk of the request; the question follows.
    prompt = [
        {
            "type": "text",
            "text": prompts.build_ask_context(name=row.document_name, insight=row.insight or {}, text_excerpt=text_excerpt),
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": prompts.build_ask_question(
                question=payload.question, language=payload.language,
                history=[t.model_dump() for t in payload.history],
            ),
        },
    ]

    async def stream():
        usage: dict = {}
        yield _ev({"type": "meta", "document_id": row.document_id, "document_name": row.document_name})
        try:
            async for piece in structured.astream_text(system, prompt, temperature=0.1, max_output_tokens=1500, usage_out=usage):
                if piece:
                    yield _ev({"type": "delta", "text": piece})
        except Exception as exc:  # noqa: BLE001
            logger.exception("document ask stream error: %s", exc)
            msg = qa.friendly_llm_error(exc) if qa.is_quota_error(exc) else "Generation interrupted; please retry."
            yield _ev({"type": "error", "message": msg})
        if usage:
            try:
                async with AsyncSessionLocal() as mb:
                    await usage_meter.record_usage(
                        mb, organization_id=payload.organization_id, user_id=payload.user_id,
                        feature=_FEATURE, tier="smart", usage=usage, request_id=req_id,
                    )
            except Exception:  # pragma: no cover
                pass
        yield _ev({"type": "done", "response_time_ms": int((time.perf_counter() - started) * 1000)})

    return StreamingResponse(stream(), media_type="application/x-ndjson", headers=_HEADERS)


# ---------------------------------------------------------------------------
# POST /copilot/document/insights  (list for the badges)
# ---------------------------------------------------------------------------
@router.post("/insights")
async def document_insights(
    payload: schemas.DocumentInsightListRequest,
    db: AsyncSession = Depends(get_db),
):
    _gate()
    await _load_session(db, payload.session_token)
    _validate_grant(payload.audit_file_id, payload.copilot_grant)
    items = await store.list_insights(
        db, audit_file_id=payload.audit_file_id, organization_id=payload.organization_id, language=payload.language
    )
    return {"audit_file_id": payload.audit_file_id, "insights": items, "count": len(items)}
