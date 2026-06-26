"""
Internal service-to-service endpoints (NOT user-facing).

Protected by a shared secret in the ``X-Internal-Secret`` header (matching
``INTERNAL_SHARED_SECRET``). Today: the "file changed" ping 1audit-be sends on any
audit-file edit so the copilot's data cache for that file is invalidated.
"""
import logging

from fastapi import APIRouter, Depends, Header, HTTPException

import file_cache_state
from config import INTERNAL_SHARED_SECRET

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/internal", tags=["internal"])


async def _require_internal_secret(x_internal_secret: str = Header(default="")) -> None:
    """Fail closed: if no secret is configured, the internal API is disabled."""
    if not INTERNAL_SHARED_SECRET:
        raise HTTPException(status_code=503, detail="Internal API is not configured.")
    if x_internal_secret != INTERNAL_SHARED_SECRET:
        raise HTTPException(status_code=403, detail="Invalid internal secret.")


@router.post("/copilot/audit_files/{audit_file_id}/changed")
async def mark_file_changed(
    audit_file_id: int, _: None = Depends(_require_internal_secret)
):
    """1audit-be calls this after any edit to an audit file. Records the change so
    the next copilot question for this file re-fetches fresh data."""
    await file_cache_state.record_changed(audit_file_id)
    return {"ok": True, "audit_file_id": audit_file_id}
