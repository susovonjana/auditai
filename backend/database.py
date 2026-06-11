"""
PostgreSQL + pgvector connection setup.
Uses SQLAlchemy async with the asyncpg driver.

TLS / Amazon RDS (Aurora) notes
-------------------------------
The asyncpg driver does NOT understand libpq-style connection params such as
``sslmode=verify-full`` or ``sslrootcert=./global-bundle.pem`` (those belong to
psycopg2). Instead, TLS is configured by handing asyncpg a real
``ssl.SSLContext`` through SQLAlchemy's ``connect_args``.

To connect to the Aurora instance, set in your environment / .env:

    DATABASE_URL=postgresql+asyncpg://prod_aura:<password>@one-audit-aura-instance-1.cjwpxsh7zcyc.eu-west-1.rds.amazonaws.com:5432/aura
    DB_SSLMODE=verify-full
    DB_SSL_ROOT_CERT=./global-bundle.pem   # AWS RDS global CA bundle

Download the CA bundle once with:
    curl -o global-bundle.pem https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem
"""
import ssl
from pathlib import Path

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text

from config import DATABASE_URL, DB_SSL_MODE, DB_SSL_ROOT_CERT

BASE_DIR = Path(__file__).resolve().parent


class Base(DeclarativeBase):
    """Shared declarative base for all SQLAlchemy ORM models."""
    pass


# asyncpg ignores ?sslmode= in the URL; SSL must be passed via connect_args.
# verify-full = CA chain check + hostname match (what RDS requires).
def _build_connect_args() -> dict:
    if not DB_SSL_MODE or DB_SSL_MODE == "disable":
        return {}
    if DB_SSL_MODE == "verify-full":
        ctx = ssl.create_default_context(cafile=DB_SSL_ROOT_CERT)
        return {"ssl": ctx}
    if DB_SSL_MODE == "verify-ca":
        ctx = ssl.create_default_context(cafile=DB_SSL_ROOT_CERT)
        ctx.check_hostname = False
        return {"ssl": ctx}
    if DB_SSL_MODE in ("require", "prefer", "allow"):
        return {"ssl": True}
    raise ValueError(f"Unsupported DB_SSL_MODE: {DB_SSL_MODE!r}")


# Async engine — pool_pre_ping helps recover from dropped DB connections
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    connect_args=_build_connect_args(),
)

# Session factory — yields AsyncSession instances
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db() -> AsyncSession:
    """
    FastAPI dependency that yields a database session.
    Used with: db: AsyncSession = Depends(get_db)
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def ensure_pgvector_extension() -> None:
    """
    Make sure the pgvector extension is enabled in the database.
    Called once at application startup.
    """
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))


async def init_db() -> None:
    """
    Create all tables if they do not exist.
    Useful for local development; production should rely on Alembic migrations.
    """
    # Importing models registers them with Base.metadata
    import models  # noqa: F401

    await ensure_pgvector_extension()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        # Create ivfflat index for fast vector similarity search (idempotent)
        await conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
                ON document_chunks
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
                """
            )
        )
