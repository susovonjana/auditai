"""
Per-file "data changed" marker for the copilot's live-data cache.

1audit-be pings auditai whenever an audit file is edited; that bumps
``audit_file_cache_state.changed_at`` (the shared source of truth). Before each
file-data chat answer, ``refresh_if_changed`` compares that timestamp to what THIS
worker last applied and, if newer, clears the file's in-memory data cache so the
next fetch is fresh. This lets the data cache live long for IDLE files (few
re-fetches) while edits show up immediately — and stays correct across multiple
auditai workers, since the marker lives in Postgres, not process memory.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict

from sqlalchemy import select, func
from sqlalchemy.dialects.postgresql import insert as pg_insert

from copilot_tools import clear_file_cache
from database import AsyncSessionLocal
from models import AuditFileCacheState

logger = logging.getLogger(__name__)

# Per-worker memory of the last changed_at this process has already applied for a
# file. (Process-local is fine — the DB row is the shared truth; each worker syncs
# its OWN cache independently.)
_last_synced: Dict[int, datetime] = {}


async def record_changed(audit_file_id: int) -> None:
    """Mark this file's data as changed now (upsert changed_at = now()). Called by
    the internal 'file changed' endpoint that 1audit-be pings on any edit."""
    fid = int(audit_file_id)
    async with AsyncSessionLocal() as db:
        stmt = (
            pg_insert(AuditFileCacheState)
            .values(audit_file_id=fid, changed_at=func.now())
            .on_conflict_do_update(
                index_elements=[AuditFileCacheState.audit_file_id],
                set_={"changed_at": func.now()},
            )
        )
        await db.execute(stmt)
        await db.commit()


async def refresh_if_changed(audit_file_id: int) -> None:
    """If this file changed since this worker last synced it, clear that file's
    cached fetches so the next question re-fetches fresh. One cheap indexed local
    DB read per file-chat question (NOT a be call). Never raises — on any error it
    leaves the cache as-is (the TTL backstop still bounds staleness)."""
    fid = int(audit_file_id)
    try:
        async with AsyncSessionLocal() as db:
            changed_at = (
                await db.execute(
                    select(AuditFileCacheState.changed_at).where(
                        AuditFileCacheState.audit_file_id == fid
                    )
                )
            ).scalar_one_or_none()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("file cache-state read failed (file %s): %s", fid, exc)
        return
    if changed_at is None:
        return  # never marked changed → nothing cached to invalidate
    last = _last_synced.get(fid)
    if last is None or changed_at > last:
        removed = clear_file_cache(fid)
        _last_synced[fid] = changed_at
        if removed:
            logger.info("file %s changed → cleared %d cached entries", fid, removed)
