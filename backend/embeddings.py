"""
Local embeddings via sentence-transformers (FREE).

Default model:  BAAI/bge-small-en-v1.5  (384-dim, ~133 MB)
The model is downloaded once to ~/.cache/huggingface and then runs entirely
on your machine — no API key, no rate limits, no internet needed after the
first download.

bge models work best with an asymmetric prefix on the QUERY side. We embed:
  - documents (chunks) with no prefix
  - queries (questions)   with the standard bge instruction prefix
"""
from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

from config import EMBEDDING_DIMENSIONS, LOCAL_EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# Lazy singleton — loading takes a few seconds, so we do it once and keep it
_model: Optional["SentenceTransformer"] = None  # type: ignore[name-defined]

# Recommended instruction prefix for bge-* English models on the query side.
# Empty for documents.
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _is_bge_model(name: str) -> bool:
    return name.lower().startswith(("bge", "baai/bge"))


def _load_model():
    """Load the sentence-transformers model (sync — call from a thread)."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Run: pip install -r requirements.txt"
            ) from exc

        logger.info(
            "Loading local embedding model '%s' (downloads on first run)...",
            LOCAL_EMBEDDING_MODEL,
        )
        _model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
        logger.info("Embedding model ready.")
    return _model


def _encode_sync(texts: List[str]) -> List[List[float]]:
    """Run the actual encoding on whichever thread the caller chose."""
    model = _load_model()
    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine-friendly
    )
    return vectors.tolist()


async def embed_text(text: str) -> List[float]:
    """Embed a single passage (document chunk). No query prefix."""
    cleaned = (text or "").replace("\n", " ").strip()
    if not cleaned:
        return [0.0] * EMBEDDING_DIMENSIONS
    vectors = await asyncio.to_thread(_encode_sync, [cleaned])
    return vectors[0]


async def embed_query(question: str) -> List[float]:
    """
    Embed a user QUERY (question). For bge-* models we prepend the official
    instruction prefix, which materially improves retrieval quality.
    """
    cleaned = (question or "").replace("\n", " ").strip()
    if not cleaned:
        return [0.0] * EMBEDDING_DIMENSIONS

    if _is_bge_model(LOCAL_EMBEDDING_MODEL):
        cleaned = _QUERY_PREFIX + cleaned

    vectors = await asyncio.to_thread(_encode_sync, [cleaned])
    return vectors[0]


async def embed_texts(
    texts: List[str],
    batch_size: int = 100,  # kept for API compatibility; ignored
) -> List[List[float]]:
    """Embed many document passages (no prefix)."""
    if not texts:
        return []
    cleaned = [(t or "").replace("\n", " ").strip() for t in texts]
    safe_batch = [c if c else " " for c in cleaned]
    vectors = await asyncio.to_thread(_encode_sync, safe_batch)
    return vectors


async def embed_queries(questions: List[str]) -> List[List[float]]:
    """
    Batch-embed multiple QUERIES (with the bge prefix on each). One thread
    hop, one model.encode call — ~1.5× the cost of a single query, not N×.
    Used by multi-query expansion in retrieve_chunks.
    """
    if not questions:
        return []
    cleaned = [(q or "").replace("\n", " ").strip() for q in questions]
    if _is_bge_model(LOCAL_EMBEDDING_MODEL):
        cleaned = [_QUERY_PREFIX + (q if q else " ") for q in cleaned]
    else:
        cleaned = [q if q else " " for q in cleaned]
    vectors = await asyncio.to_thread(_encode_sync, cleaned)
    return vectors
