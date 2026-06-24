"""
Re-embed every stored vector with the CURRENT LOCAL_EMBEDDING_MODEL.

Why: the embedding model is a coordinate system. The moment you switch models
(e.g. English-only all-MiniLM/bge-small-en  ->  multilingual-e5-small for Arabic),
every vector already in the DB is in the OLD model's space and is no longer
comparable to vectors the new model produces. Cosine scores become meaningless
until everything is re-embedded in the new space.

What it does (purely additive — NO deletes, NO schema/DDL change):
  * document_chunks   .embedding  <- embed_text(content)
  * tb_mapping_memory .embedding  <- embed_text(account_text)
  * proc_memory       .embedding  <- embed_text(memory_key(audit_area, risk_summary))
Each row is UPDATED in place (the source text already lives in the row), then the
three ivfflat indexes are REINDEXed so their centroids retrain on the new vectors.
search_history.question_embedding is a transient query log and is intentionally
left alone (it naturally refreshes as new questions arrive).

The vectors stay 384-dim, so the Vector(384) columns and indexes are unchanged —
this only rewrites values. Safe to re-run (idempotent: re-embeds to the same space).

Run inside the auditai backend (so DATABASE_URL + the model are configured):
    python reembed_all.py
"""
from __future__ import annotations

import asyncio
import logging
from typing import List, Tuple

from sqlalchemy import select, text

import proc_memory
from config import LOCAL_EMBEDDING_MODEL
from database import AsyncSessionLocal
from embeddings import embed_texts
from models import DocumentChunk, ProcMemory, TbMappingMemory

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("reembed_all")

_BATCH = 256  # rows embedded per model.encode call / committed per round

# The ivfflat indexes to rebuild after the values change (names from database.py).
_INDEXES = (
    "document_chunks_embedding_idx",
    "proc_memory_embedding_idx",
    "tb_mapping_memory_embedding_idx",
)


async def _reembed_table(label: str, model, id_attr: str, text_for) -> int:
    """Re-embed one table in batches. `text_for(row)` returns the text to embed."""
    done = 0
    async with AsyncSessionLocal() as session:
        rows = (await session.execute(select(model))).scalars().all()
        total = len(rows)
        if total == 0:
            logger.info("%s: nothing to re-embed (0 rows).", label)
            return 0
        logger.info("%s: re-embedding %d rows...", label, total)
        for start in range(0, total, _BATCH):
            batch = rows[start : start + _BATCH]
            texts: List[str] = [text_for(r) or "" for r in batch]
            vectors = await embed_texts(texts)
            for row, vec in zip(batch, vectors):
                row.embedding = vec
            await session.commit()
            done += len(batch)
            logger.info("%s: %d/%d", label, done, total)
    return done


async def _reindex() -> None:
    """REINDEX the vector indexes so ivfflat centroids retrain on the new vectors."""
    async with AsyncSessionLocal() as session:
        for idx in _INDEXES:
            try:
                await session.execute(text(f"REINDEX INDEX {idx}"))
                await session.commit()
                logger.info("REINDEXed %s", idx)
            except Exception as exc:  # index may not exist yet on a fresh DB
                await session.rollback()
                logger.warning("REINDEX %s skipped: %s", idx, exc)


async def run() -> None:
    logger.info("Re-embedding ALL stored vectors with model: %s", LOCAL_EMBEDDING_MODEL)

    plan: Tuple[Tuple[str, object, str, object], ...] = (
        ("document_chunks", DocumentChunk, "id", lambda r: r.content),
        ("tb_mapping_memory", TbMappingMemory, "id", lambda r: r.account_text),
        (
            "proc_memory",
            ProcMemory,
            "id",
            lambda r: proc_memory.memory_key(r.audit_area, r.risk_summary),
        ),
    )

    grand_total = 0
    for label, model, id_attr, text_for in plan:
        grand_total += await _reembed_table(label, model, id_attr, text_for)

    if grand_total:
        logger.info("Rebuilding vector indexes...")
        await _reindex()

    logger.info("Done. Re-embedded %d rows total.", grand_total)


if __name__ == "__main__":
    asyncio.run(run())
