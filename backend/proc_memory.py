"""
Procedure memory (ticket A-1b) — the firm "house style" learning store.

Every accepted AI-assisted procedure is embedded by its retrieval key
``audit_area + risk_summary`` and stored in ``proc_memory``. When drafting a new
procedure we retrieve the closest past procedures for the SAME organization
(preferring the same client sector) and pass them to the model as few-shot
examples, so output increasingly matches how this firm actually writes.

Org-scoped and additive: this module only ever reads/appends rows in auditai's
own Postgres. It never touches 1audit's database, and contains no update/delete.
"""
from __future__ import annotations

import hashlib
import logging
import re
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from embeddings import embed_query, embed_text, embed_texts
from models import ProcMemory

logger = logging.getLogger(__name__)

# Keep a stored procedure bounded so a pathological payload can't bloat the row.
_MAX_PROCEDURE_CHARS = 8000
# One agent run can approve a large program; cap what a single run may add so a
# runaway payload can't flood the memory.
_MAX_ROWS_PER_INGEST = 40
# A procedure shorter than this (tag-stripped) is boilerplate, not house style.
_MIN_PROCEDURE_TEXT_CHARS = 40


def _normalize_text(html: Optional[str]) -> str:
    """Tag-stripped, lowercased, whitespace-collapsed text — the dedupe basis, so
    markup/whitespace-only differences hash identically while any wording change
    produces a new row."""
    s = re.sub(r"<[^>]*>", " ", str(html or ""))
    s = s.replace("&nbsp;", " ").replace("&amp;", "&")
    return re.sub(r"\s+", " ", s).strip().lower()


def content_hash(audit_area: Optional[str], procedure_html: Optional[str]) -> str:
    """Stable per-procedure dedupe key: sha256 over the normalized area + text.
    The area participates so the same wording stored for two different working
    papers stays two examples (each retrievable by its own area)."""
    basis = f"{_normalize_text(audit_area)}\n{_normalize_text(procedure_html)}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def memory_key(audit_area: Optional[str], risk_summary: Optional[str]) -> str:
    """The text we embed and search on: the audit area + the risk it responds to."""
    parts = [(audit_area or "").strip(), (risk_summary or "").strip()]
    return " — ".join(p for p in parts if p) or "audit procedure"


def summarize_risks(risks: List[dict]) -> str:
    """Compact one-line risk summary from the linked risk dicts (title/description)."""
    bits: List[str] = []
    for r in risks or []:
        t = (r.get("title") or "").strip()
        d = (r.get("description") or "").strip()
        if t and d:
            bits.append(f"{t}: {d}")
        elif t or d:
            bits.append(t or d)
    return "; ".join(bits)[:2000]


async def _existing_hashes(
    db: AsyncSession, organization_id: str, hashes: List[str]
) -> set:
    """The subset of ``hashes`` this org already stores (one indexed query)."""
    if not hashes:
        return set()
    res = await db.execute(
        select(ProcMemory.content_hash).where(
            ProcMemory.organization_id == str(organization_id),
            ProcMemory.content_hash.in_(hashes),
        )
    )
    return {h for (h,) in res.all() if h}


async def add_memory(
    db: AsyncSession,
    *,
    organization_id: Optional[str],
    client_sector: Optional[str],
    audit_area: Optional[str],
    risk_summary: Optional[str],
    assertions: Optional[List[str]],
    procedure_html: str,
    confirmed_by: Optional[str] = None,
    source: Optional[str] = None,
    source_key: Optional[str] = None,
) -> Optional[uuid.UUID]:
    """Embed and append one accepted procedure. Returns the new row id, or None
    when there is nothing worth storing (empty procedure, no org scope, or the
    same content is already in this org's memory)."""
    html = (procedure_html or "").strip()
    if not html or not organization_id:
        return None
    digest = content_hash(audit_area, html)
    if await _existing_hashes(db, str(organization_id), [digest]):
        return None  # already learned — markup-only differences dedupe here
    key = memory_key(audit_area, risk_summary)
    embedding = await embed_text(key)
    row = ProcMemory(
        organization_id=str(organization_id),
        client_sector=(client_sector or None),
        audit_area=(audit_area or None),
        risk_summary=(risk_summary or None),
        assertions=list(assertions or []),
        procedure_html=html[:_MAX_PROCEDURE_CHARS],
        embedding=embedding,
        confirmed_by=(confirmed_by or None),
        source=(source or None),
        source_key=(source_key or None),
        content_hash=digest,
    )
    db.add(row)
    try:
        await db.commit()
    except IntegrityError:  # concurrent writer landed the same hash first
        await db.rollback()
        return None
    return row.id


async def add_memories(
    db: AsyncSession,
    rows: List[Dict[str, Any]],
    *,
    organization_id: Optional[str],
    source: str,
    max_rows: int = 500,
) -> int:
    """Batch-append procedures for ONE org (feedback ingest / seeding): dedupe
    in-batch and against stored hashes, embed all retrieval keys in one model
    call, insert in one commit. Returns the number of rows actually added.
    ``max_rows`` is a hard backstop (a seed export is itself capped at 500; the
    per-run feedback extract at ``_MAX_ROWS_PER_INGEST``).

    Each row dict: {audit_area, risk_summary?, client_sector?, assertions?,
    procedure_html, confirmed_by?, source_key?}.
    """
    if not organization_id:
        return 0
    org = str(organization_id)

    prepared: List[Dict[str, Any]] = []
    seen: set = set()
    for r in rows or []:
        html = str(r.get("procedure_html") or "").strip()
        if not html:
            continue
        digest = content_hash(r.get("audit_area"), html)
        if digest in seen:
            continue
        seen.add(digest)
        prepared.append({**r, "procedure_html": html, "content_hash": digest})
        if len(prepared) >= max(1, int(max_rows)):
            break
    if not prepared:
        return 0

    stored = await _existing_hashes(db, org, [p["content_hash"] for p in prepared])
    prepared = [p for p in prepared if p["content_hash"] not in stored]
    if not prepared:
        return 0

    keys = [memory_key(p.get("audit_area"), p.get("risk_summary")) for p in prepared]
    embeddings = await embed_texts(keys)

    for p, embedding in zip(prepared, embeddings):
        db.add(
            ProcMemory(
                organization_id=org,
                client_sector=(p.get("client_sector") or None),
                audit_area=(p.get("audit_area") or None),
                risk_summary=(p.get("risk_summary") or None),
                assertions=list(p.get("assertions") or []),
                procedure_html=p["procedure_html"][:_MAX_PROCEDURE_CHARS],
                embedding=embedding,
                confirmed_by=(p.get("confirmed_by") or None),
                source=source,
                source_key=(p.get("source_key") or None),
                content_hash=p["content_hash"],
            )
        )
    try:
        await db.commit()
    except IntegrityError:
        # A concurrent ingest landed overlapping hashes: fall back to per-row
        # adds, which re-check the hash individually.
        await db.rollback()
        added = 0
        for p in prepared:
            rid = await add_memory(
                db,
                organization_id=org,
                client_sector=p.get("client_sector"),
                audit_area=p.get("audit_area"),
                risk_summary=p.get("risk_summary"),
                assertions=p.get("assertions"),
                procedure_html=p["procedure_html"],
                confirmed_by=p.get("confirmed_by"),
                source=source,
                source_key=p.get("source_key"),
            )
            if rid is not None:
                added += 1
        return added
    return len(prepared)


def extract_memory_rows(
    approved_payload: Optional[Dict[str, Any]],
    *,
    run_id: str,
    audit_area: Optional[str],
    risk_summary: Optional[str],
    client_sector: Optional[str],
    confirmed_by: Optional[str],
) -> List[Dict[str, Any]]:
    """The APPROVED (auditor-edited) bulk-create payload -> add_memories rows.
    Only real procedures (section_type 2) with substantive text are worth
    learning; titles/comments carry structure, not house style. Pure function so
    the ingest policy is offline-testable."""
    out: List[Dict[str, Any]] = []
    sections = (approved_payload or {}).get("sections") or []
    for s in sections:
        if not isinstance(s, dict) or int(s.get("section_type") or 0) != 2:
            continue
        html = str(s.get("procedure") or "").strip()
        if len(_normalize_text(html)) < _MIN_PROCEDURE_TEXT_CHARS:
            continue
        out.append(
            {
                "audit_area": audit_area,
                "risk_summary": risk_summary,
                "client_sector": client_sector,
                "assertions": [
                    str(a).strip()
                    for a in (s.get("assertions") or [])
                    if str(a or "").strip()
                ],
                "procedure_html": html,
                "confirmed_by": confirmed_by,
                "source_key": f"agent:{run_id}:{s.get('temp_id')}",
            }
        )
        if len(out) >= _MAX_ROWS_PER_INGEST:
            break
    return out


async def search_examples(
    db: AsyncSession,
    *,
    organization_id: Optional[str],
    client_sector: Optional[str],
    audit_area: Optional[str],
    risk_summary: Optional[str],
    k: int = 3,
) -> List[ProcMemory]:
    """Up to ``k`` of THIS org's closest past procedures, by cosine distance on
    ``audit_area + risk_summary``, preferring the same client sector. Org scoping
    is mandatory — a firm only ever sees its own procedures (returns [] without
    an organization_id)."""
    if not organization_id:
        return []
    key = memory_key(audit_area, risk_summary)
    try:
        embedding = await embed_query(key)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("proc_memory embed failed: %s", exc)
        return []

    sector = (client_sector or "").strip() or None
    order_by = []
    if sector is not None:
        # Same-sector examples first, then nearest by meaning.
        order_by.append((ProcMemory.client_sector == sector).desc())
    order_by.append(ProcMemory.embedding.cosine_distance(embedding))

    stmt = (
        select(ProcMemory)
        .where(ProcMemory.organization_id == str(organization_id))
        .order_by(*order_by)
        .limit(max(1, int(k)))
    )
    res = await db.execute(stmt)
    return list(res.scalars().all())
