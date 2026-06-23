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

import logging
import uuid
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from embeddings import embed_query, embed_text
from models import ProcMemory

logger = logging.getLogger(__name__)

# Keep a stored procedure bounded so a pathological payload can't bloat the row.
_MAX_PROCEDURE_CHARS = 8000


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
) -> Optional[uuid.UUID]:
    """Embed and append one accepted procedure. Returns the new row id, or None
    when there is nothing worth storing (empty procedure or no org scope)."""
    html = (procedure_html or "").strip()
    if not html or not organization_id:
        return None
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
    )
    db.add(row)
    await db.commit()
    return row.id


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
