"""
AuditAI — FastAPI application entry point.

Run locally:
    uvicorn main:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select, text
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from auth import get_current_admin, hash_password
from config import ADMIN_PASSWORD, ADMIN_USERNAME, FRONTEND_ORIGIN
from database import AsyncSessionLocal, engine, init_db
from models import AdminUser
from rate_limit import limiter
from routers import admin as admin_router
from routers import agent as agent_router
from routers import copilot as copilot_router
from routers import health as health_router
from routers import internal as internal_router
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
    doesn't pay the 10-second cold-start cost. Loaded in parallel so total
    warm-up time is max(embed, rerank) instead of their sum.
    """
    try:
        from embeddings import embed_query
        from reranker import rerank

        logger.info("Warming up embedding + reranker models in parallel...")
        results = await asyncio.gather(
            embed_query("warmup"),
            rerank("warmup", ["sample passage one", "sample passage two"]),
            return_exceptions=True,
        )
        for name, r in zip(("embedding", "reranker"), results):
            if isinstance(r, Exception):
                logger.warning("Warm-up of %s failed (non-fatal): %s", name, r)
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
    "https://beta.1audit.com/"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(dict.fromkeys(allowed_origins)),  # de-dup, preserve order
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routers ---
app.include_router(health_router.router)
app.include_router(admin_router.router)
app.include_router(user_router.router)
app.include_router(copilot_router.router)
app.include_router(internal_router.router)
app.include_router(agent_router.router)


@app.get("/")
async def root():
    return {
        "service": "AuditAI",
        "status": "running",
        "docs": "/docs",
        "user_endpoints": ["/session/start", "/ask", "/feedback", "/health"],
        "admin_endpoints": ["/admin/login", "/admin/status", "/admin/upload", "/migrate"],
    }


async def _run_alembic(args: list[str]) -> tuple[int, str]:
    backend_dir = Path(__file__).resolve().parent
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "alembic", *args,
        cwd=str(backend_dir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=300)
    except asyncio.TimeoutError:
        proc.kill()
        raise HTTPException(status_code=504, detail="Alembic command timed out after 5 minutes")
    return proc.returncode, stdout.decode("utf-8", errors="replace")


@app.post("/migrate")
async def run_migrations(
    action: str = "upgrade",
    target: str = "head",
    admin: AdminUser = Depends(get_current_admin),
):
    """
    Manage Alembic migrations. Admin JWT required.

    Query params:
      action: one of "upgrade" (default), "stamp", "current", "history", "inspect"
      target: revision target for upgrade/stamp (default "head")

    Typical first-time fix when DB was created by create_all() (no alembic_version):
      1) /migrate?action=inspect            -> see actual columns of user_sessions
      2) /migrate?action=current            -> shows alembic state (likely empty)
      3) /migrate?action=stamp&target=005_qa_user_org   -> mark 1..5 as applied
      4) /migrate                            -> upgrade head (runs 006, 007)
    """
    logger.info("/migrate action=%s target=%s requested by %s", action, target, admin.username)

    if action == "inspect":
        async with engine.connect() as conn:
            tables = (await conn.execute(text(
                "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename"
            ))).scalars().all()
            user_sessions_cols = (await conn.execute(text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema='public' AND table_name='user_sessions' "
                "ORDER BY ordinal_position"
            ))).scalars().all()
            search_history_cols = (await conn.execute(text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema='public' AND table_name='search_history' "
                "ORDER BY ordinal_position"
            ))).scalars().all()
            try:
                alembic_version = (await conn.execute(text(
                    "SELECT version_num FROM alembic_version"
                ))).scalars().all()
            except Exception:
                alembic_version = None
        return {
            "ok": True,
            "tables": list(tables),
            "user_sessions_columns": list(user_sessions_cols),
            "search_history_columns": list(search_history_cols),
            "alembic_version_rows": alembic_version,
        }

    if action == "repair":
        # Idempotent DDL — adds columns/indexes from migrations 005..007 if missing.
        # Safe to run multiple times. Use when create_all() built tables before
        # migrations were written, so columns the models expect aren't in the DB.
        statements = [
            'ALTER TABLE search_history ADD COLUMN IF NOT EXISTS user_id TEXT',
            'ALTER TABLE search_history ADD COLUMN IF NOT EXISTS organization_id TEXT',
            'ALTER TABLE search_history ADD COLUMN IF NOT EXISTS prompt_tokens INTEGER',
            'ALTER TABLE search_history ADD COLUMN IF NOT EXISTS completion_tokens INTEGER',
            'ALTER TABLE search_history ADD COLUMN IF NOT EXISTS total_tokens INTEGER',
            'CREATE INDEX IF NOT EXISTS ix_search_history_user_id ON search_history(user_id)',
            'CREATE INDEX IF NOT EXISTS ix_search_history_organization_id ON search_history(organization_id)',
            'ALTER TABLE user_sessions ADD COLUMN IF NOT EXISTS user_id TEXT',
            'ALTER TABLE user_sessions ADD COLUMN IF NOT EXISTS organization_id TEXT',
            'CREATE INDEX IF NOT EXISTS ix_user_sessions_user_id ON user_sessions(user_id)',
            'CREATE INDEX IF NOT EXISTS ix_user_sessions_organization_id ON user_sessions(organization_id)',
        ]
        applied = []
        async with engine.begin() as conn:
            for stmt in statements:
                await conn.execute(text(stmt))
                applied.append(stmt)
        return {"ok": True, "action": "repair", "applied": applied}

    if action == "current":
        code, output = await _run_alembic(["current"])
    elif action == "history":
        code, output = await _run_alembic(["history"])
    elif action == "stamp":
        code, output = await _run_alembic(["stamp", target])
    elif action == "upgrade":
        code, output = await _run_alembic(["upgrade", target])
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {action}")

    if code != 0:
        raise HTTPException(
            status_code=500,
            detail={"error": f"alembic {action} failed", "exit_code": code, "output": output},
        )
    return {"ok": True, "action": action, "target": target, "output": output}



