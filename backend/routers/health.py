"""
Health-check endpoints for ALB / ECS probes and observability.

  GET /health         — fast liveness probe (no I/O, always 200 if process is up)
  GET /health/live    — alias for /health (Kubernetes naming convention)
  GET /health/ready   — readiness probe (verifies DB connectivity; 503 if degraded)
"""
from datetime import datetime, timezone

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from sqlalchemy import text

from database import engine

router = APIRouter(tags=["health"])


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@router.get("/health")
async def liveness():
    """Liveness — confirms the process is running. No external I/O."""
    return {
        "status": "ok",
        "service": "AuditAI",
        "time": _now_iso(),
    }


@router.get("/health/live")
async def liveness_alias():
    return await liveness()


@router.get("/health/ready")
async def readiness():
    """
    Readiness — confirms the service can serve traffic.
    Currently verifies DB connectivity. Returns 503 if any check fails so
    the ALB can route traffic away from a degraded task.
    """
    checks: dict[str, str] = {}
    overall_ok = True

    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:
        checks["database"] = f"error: {type(exc).__name__}: {exc}"
        overall_ok = False

    payload = {
        "status": "ok" if overall_ok else "degraded",
        "service": "AuditAI",
        "time": _now_iso(),
        "checks": checks,
    }
    code = status.HTTP_200_OK if overall_ok else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(content=payload, status_code=code)
