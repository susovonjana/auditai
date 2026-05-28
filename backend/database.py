"""
PostgreSQL + pgvector connection setup.
Uses SQLAlchemy async with asyncpg driver.
"""
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text

from config import DATABASE_URL


class Base(DeclarativeBase):
    """Shared declarative base for all SQLAlchemy ORM models."""
    pass


# Async engine — pool_pre_ping helps recover from dropped DB connections
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
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
