"""
AuditAI — FastAPI application entry point.

Run locally:
    uvicorn main:app --reload --port 8000
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from auth import hash_password
from config import ADMIN_PASSWORD, ADMIN_USERNAME, FRONTEND_ORIGIN
from database import AsyncSessionLocal, init_db
from models import AdminUser
from rate_limit import limiter
from routers import admin as admin_router
from routers import user as user_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("auditai")


async def _seed_initial_admin() -> None:
    """Create the initial admin account on first run."""
    if not ADMIN_USERNAME or not ADMIN_PASSWORD:
        logger.warning("ADMIN_USERNAME / ADMIN_PASSWORD not set — skipping seed.")
        return

    async with AsyncSessionLocal() as db:
        existing = (
            await db.execute(
                select(AdminUser).where(AdminUser.username == ADMIN_USERNAME)
            )
        ).scalar_one_or_none()
        if existing:
            logger.info("Admin '%s' already exists.", ADMIN_USERNAME)
            return

        admin = AdminUser(
            username=ADMIN_USERNAME,
            password_hash=hash_password(ADMIN_PASSWORD),
            role="superadmin",
        )
        db.add(admin)
        await db.commit()
        logger.info("Seeded initial admin user '%s'.", ADMIN_USERNAME)


async def _recover_stale_uploads() -> None:
    """
    Mark any document stuck mid-processing as error. Background tasks die
    when uvicorn restarts; on next boot we surface this clearly so the
    admin knows to re-upload, instead of leaving zombie 'parsing' rows.
    """
    try:
        from sqlalchemy import update
        from models import Document

        STALE_STATES = ("queued", "parsing", "embedding")
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                update(Document)
                .where(Document.status.in_(STALE_STATES))
                .values(
                    status="error",
                    error_message="Upload was interrupted by a server restart. Please re-upload.",
                )
            )
            if result.rowcount:
                logger.warning(
                    "Recovered %s stale upload(s) — marked as error.",
                    result.rowcount,
                )
            await db.commit()
    except Exception as exc:
        logger.warning("Stale-upload recovery skipped: %s", exc)


async def _warm_up_models() -> None:
    """
    Pre-load the embedding model and reranker so the first user request
    doesn't pay the 10-second cold-start cost.
    """
    try:
        from embeddings import embed_query
        from reranker import rerank

        logger.info("Warming up embedding model...")
        await embed_query("warmup")
        logger.info("Warming up reranker model...")
        await rerank("warmup", ["sample passage one", "sample passage two"])
        logger.info("Models warm and ready.")
    except Exception as exc:
        # Don't block startup — first request will pay the cold-start cost.
        logger.warning("Model warm-up failed (non-fatal): %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB + seed admin + warm up models. Shutdown: nothing special."""
    logger.info("AuditAI starting up...")
    try:
        await init_db()
        await _seed_initial_admin()
        await _recover_stale_uploads()
        await _warm_up_models()
    except Exception as exc:  # pragma: no cover
        logger.exception("Startup error: %s", exc)
    yield
    logger.info("AuditAI shutting down...")


app = FastAPI(
    title="AuditAI",
    description="AI-powered audit knowledge assistant with pgvector semantic search.",
    version="1.0.0",
    lifespan=lifespan,
)

# --- Rate limiting (slowapi) ---
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# --- CORS ---
# FRONTEND_ORIGIN can be comma-separated for multi-origin (e.g. standalone
# AuditAI UI + the embedded widget inside 1audit).
_extra_origins = [o.strip() for o in (FRONTEND_ORIGIN or "").split(",") if o.strip()]
allowed_origins = [
    *_extra_origins,
    "http://localhost:5173",       # standalone AuditAI dev
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://localhost:3002",       # 1audit dev port
    "http://127.0.0.1:3002",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(dict.fromkeys(allowed_origins)),  # de-dup, preserve order
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routers ---
app.include_router(admin_router.router)
app.include_router(user_router.router)


@app.get("/")
async def root():
    return {
        "service": "AuditAI",
        "status": "running",
        "docs": "/docs",
        "user_endpoints": ["/session/start", "/ask", "/feedback", "/health"],
        "admin_endpoints": ["/admin/login", "/admin/status", "/admin/upload"],
    }
