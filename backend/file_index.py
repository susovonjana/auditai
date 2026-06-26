"""
Per-file RAG index (phase 2) — semantic search over ONE audit file's
working-paper NARRATIVE.

The copilot answers most file questions with always-fresh LIVE tools (trial
balance, materiality, samples, …). Those tools are weak at ONE thing: finding
qualitative text spread across many working papers ("which WP discusses going
concern / related parties?", "what did we conclude about revenue recognition?").
This module fills that gap: it chunks + embeds the file's WP narrative into
``audit_file_chunks`` (pgvector) and exposes ``retrieve_file`` for a ``search_file``
tool.

Freshness (the gating concern): the index is a transient cache — 1audit-be stays
authoritative. It is rebuilt LAZILY:
  * each search fetches a cheap change-signature (WP count + latest updated_at);
  * if the signature differs from what we indexed, OR a TTL backstop has elapsed,
    we fetch the full narrative bundle, and rebuild only if its CONTENT HASH
    changed (so a no-op edit doesn't re-embed).
Rebuilds are serialized per file with an asyncio lock, and the chat route also
prewarms the index at session start so the first search is usually a cache hit.

Purely additive: rows live only in auditai's own Postgres.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from sqlalchemy import select, delete

from chunker import chunk_text
from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COPILOT_FILE_INDEX_MAX_CHUNKS,
    COPILOT_FILE_INDEX_TTL_SEC,
    COPILOT_FILE_SEARCH_CANDIDATES,
    COPILOT_FILE_SEARCH_TOP_K,
    USE_RERANKER,
)
from database import AsyncSessionLocal
from embeddings import embed_query, embed_texts
from models import AuditFileChunk, AuditFileIndex
from reranker import rerank

logger = logging.getLogger(__name__)

# Per-file rebuild locks so two concurrent requests (e.g. prewarm + a search)
# never rebuild the same file's index twice. Created lazily on the running loop.
_locks: Dict[int, asyncio.Lock] = {}


def _lock_for(audit_file_id: int) -> asyncio.Lock:
    lk = _locks.get(audit_file_id)
    if lk is None:
        lk = asyncio.Lock()
        _locks[audit_file_id] = lk
    return lk


# ---------------------------------------------------------------------------
# Bundle → chunk texts
# ---------------------------------------------------------------------------
def _bundle_to_chunks(bundle: Any) -> List[Tuple[str, str]]:
    """Flatten the be ``wp_content_bundle`` into (source_ref, chunk_text) pairs.
    Each working paper's narrative is composed into one text (procedure questions,
    notes, free-text answers, titles, comments), then chunked — and every chunk
    keeps its working-paper label so the model can cite / drill into it."""
    wps = (bundle or {}).get("working_papers") or []
    out: List[Tuple[str, str]] = []
    for wp in wps:
        ref = (wp.get("reference") or "").strip()
        name = (wp.get("name") or "").strip()
        src = " - ".join([p for p in (ref, name) if p]) or "Working paper"
        parts: List[str] = [f"Working paper: {src}"]
        for it in wp.get("items") or []:
            kind = it.get("kind")
            if kind == "title" and it.get("text"):
                parts.append(f"Section: {it['text']}")
            elif kind == "comment" and it.get("text"):
                parts.append(f"Comment: {it['text']}")
            elif kind == "procedure":
                if it.get("question"):
                    parts.append(f"Procedure: {it['question']}")
                if it.get("note"):
                    parts.append(f"Note: {it['note']}")
                for r in it.get("responses") or []:
                    if r:
                        parts.append(f"Response: {r}")
        text = "\n".join(p for p in parts if p and p.strip())
        if not text.strip():
            continue
        for piece in chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP):
            if piece and piece.strip():
                out.append((src, piece))
                if len(out) >= COPILOT_FILE_INDEX_MAX_CHUNKS:
                    logger.info(
                        "file index: chunk cap %d reached — narrative truncated",
                        COPILOT_FILE_INDEX_MAX_CHUNKS,
                    )
                    return out
    return out


def _content_hash(chunks: List[Tuple[str, str]]) -> str:
    h = hashlib.sha256()
    for src, txt in chunks:
        h.update((src or "").encode("utf-8", "ignore"))
        h.update(b"\x00")
        h.update((txt or "").encode("utf-8", "ignore"))
        h.update(b"\x01")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# (Re)build
# ---------------------------------------------------------------------------
async def ensure_index(
    audit_file_id: int,
    fetch_signature: Callable[[], Any],
    fetch_bundle: Callable[[], Any],
) -> None:
    """Make sure this file's index is fresh, rebuilding only when needed.

    ``fetch_signature`` / ``fetch_bundle`` are BLOCKING callables (HTTP to be);
    they are run in a thread so the event loop is never blocked. Any fetch error
    leaves the existing index untouched (a stale answer beats no answer)."""
    async with _lock_for(audit_file_id):
        async with AsyncSessionLocal() as db:
            meta = await db.get(AuditFileIndex, audit_file_id)
            now = datetime.now(timezone.utc)

            # Cheap change-signature.
            signature: Optional[str] = None
            try:
                sig_payload = await asyncio.to_thread(fetch_signature)
                if isinstance(sig_payload, dict):
                    signature = sig_payload.get("signature")
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("file index signature fetch failed (file %s): %s", audit_file_id, exc)

            indexed_at = meta.indexed_at if meta else None
            if indexed_at is not None and indexed_at.tzinfo is None:
                indexed_at = indexed_at.replace(tzinfo=timezone.utc)
            within_ttl = (
                indexed_at is not None
                and (now - indexed_at).total_seconds() < COPILOT_FILE_INDEX_TTL_SEC
            )
            if (
                meta is not None
                and meta.chunk_count > 0
                and signature is not None
                and meta.signature == signature
                and within_ttl
            ):
                return  # fresh — reuse existing chunks

            # Need to (re)check content. Fetch the full narrative bundle.
            try:
                bundle = await asyncio.to_thread(fetch_bundle)
            except Exception as exc:
                logger.warning("file index bundle fetch failed (file %s): %s", audit_file_id, exc)
                return
            if isinstance(bundle, dict) and bundle.get("error"):
                logger.warning("file index bundle error (file %s): %s", audit_file_id, bundle.get("error"))
                return

            chunks = _bundle_to_chunks(bundle)
            content_hash = _content_hash(chunks)

            # Content unchanged since last build → just refresh freshness markers.
            if meta is not None and meta.chunk_count > 0 and meta.content_hash == content_hash:
                meta.signature = signature
                meta.indexed_at = now
                await db.commit()
                return

            if not chunks:
                # Nothing to index (no narrative). Record an empty index so we
                # don't re-fetch the bundle every search within the TTL window.
                await db.execute(
                    delete(AuditFileChunk).where(AuditFileChunk.audit_file_id == audit_file_id)
                )
                if meta is None:
                    meta = AuditFileIndex(audit_file_id=audit_file_id)
                    db.add(meta)
                meta.signature = signature
                meta.content_hash = content_hash
                meta.chunk_count = 0
                meta.indexed_at = now
                await db.commit()
                return

            # Rebuild: embed and REPLACE this file's chunks wholesale.
            t0 = datetime.now(timezone.utc)
            vectors = await embed_texts([c[1] for c in chunks])
            await db.execute(
                delete(AuditFileChunk).where(AuditFileChunk.audit_file_id == audit_file_id)
            )
            for i, ((src, txt), vec) in enumerate(zip(chunks, vectors)):
                db.add(
                    AuditFileChunk(
                        audit_file_id=audit_file_id,
                        chunk_index=i,
                        content=txt,
                        embedding=vec,
                        source_ref=src,
                        char_count=len(txt),
                    )
                )
            if meta is None:
                meta = AuditFileIndex(audit_file_id=audit_file_id)
                db.add(meta)
            meta.signature = signature
            meta.content_hash = content_hash
            meta.chunk_count = len(chunks)
            meta.indexed_at = now
            await db.commit()
            logger.info(
                "file index rebuilt (file %s): %d chunks in %dms",
                audit_file_id, len(chunks),
                int((datetime.now(timezone.utc) - t0).total_seconds() * 1000),
            )


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
async def _search(db, audit_file_id: int, query: str, top_k: int) -> List[Dict[str, Any]]:
    emb = await embed_query(query)
    stmt = (
        select(
            AuditFileChunk.content,
            AuditFileChunk.source_ref,
            AuditFileChunk.embedding.cosine_distance(emb).label("dist"),
        )
        .where(AuditFileChunk.audit_file_id == audit_file_id)
        .order_by("dist")
        .limit(COPILOT_FILE_SEARCH_CANDIDATES)
    )
    rows = (await db.execute(stmt)).all()
    if not rows:
        return []

    order = list(range(len(rows)))
    if USE_RERANKER and len(rows) > 1:
        try:
            scores = await rerank(query, [r.content for r in rows])
            order = sorted(order, key=lambda i: scores[i], reverse=True)
        except Exception as exc:  # pragma: no cover - reranker is best-effort
            logger.warning("file search rerank failed (file %s): %s", audit_file_id, exc)

    results: List[Dict[str, Any]] = []
    for i in order[:top_k]:
        r = rows[i]
        results.append(
            {"source": r.source_ref or "Working paper", "content": (r.content or "")[:1200]}
        )
    return results


async def retrieve_file(
    audit_file_id: int,
    query: str,
    fetch_signature: Callable[[], Any],
    fetch_bundle: Callable[[], Any],
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Ensure the index is fresh, then return the top narrative passages for the
    query. Each result is {source, content}. Returns [] on any failure."""
    k = top_k or COPILOT_FILE_SEARCH_TOP_K
    try:
        await ensure_index(audit_file_id, fetch_signature, fetch_bundle)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("file index ensure failed (file %s): %s", audit_file_id, exc)
    try:
        async with AsyncSessionLocal() as db:
            return await _search(db, audit_file_id, query, k)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("file search failed (file %s): %s", audit_file_id, exc)
        return []


async def prewarm(
    audit_file_id: int,
    fetch_signature: Callable[[], Any],
    fetch_bundle: Callable[[], Any],
) -> None:
    """Fire-and-forget index warm at chat session start — never raises."""
    try:
        await ensure_index(audit_file_id, fetch_signature, fetch_bundle)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("file index prewarm failed (file %s): %s", audit_file_id, exc)
