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

from auth import hash_password
from config import ADMIN_PASSWORD, ADMIN_USERNAME, FRONTEND_ORIGIN
from database import AsyncSessionLocal, init_db
from models import AdminUser
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB + seed admin. Shutdown: nothing special needed."""
    logger.info("AuditAI starting up...")
    try:
        await init_db()
        await _seed_initial_admin()
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

# --- CORS ---
allowed_origins = [
    FRONTEND_ORIGIN,
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
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
