"""
Local embeddings via sentence-transformers (FREE).

Default model:  intfloat/multilingual-e5-small  (384-dim, ~470 MB, ~100 langs)
Chosen for strong Arabic + English retrieval (TB account names, COA labels and
procedures are frequently Arabic). The model is downloaded once to
~/.cache/huggingface and then runs entirely on your machine — no API key, no
rate limits, no internet needed after the first download.

Embedding is asymmetric — stored text and lookups get different prefixes:
  - passages (chunks / stored account text)  via embed_text/embed_texts
  - queries  (questions / accounts to map)   via embed_query/embed_queries
The exact prefix is chosen per model family (see _query_prefix/_passage_prefix):
e5 → "query: " / "passage: ", bge → query-only, MiniLM → none.
"""
from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

from config import EMBEDDING_DIMENSIONS, LOCAL_EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# Lazy singleton — loading takes a few seconds, so we do it once and keep it
_model: Optional["SentenceTransformer"] = None  # type: ignore[name-defined]

# Asymmetric instruction prefixes vary by model family:
#   - e5 (multilingual-e5-*) wants a prefix on BOTH sides: "query: " / "passage: "
#   - bge-* wants an instruction prefix on the QUERY side only
#   - MiniLM / paraphrase-multilingual want NO prefix at all
# Using the wrong scheme (or none) noticeably degrades e5 retrieval quality,
# so we pick the right prefix from the active model name.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_E5_QUERY_PREFIX = "query: "
_E5_PASSAGE_PREFIX = "passage: "


def _is_bge_model(name: str) -> bool:
    return name.lower().startswith(("bge", "baai/bge"))


def _is_e5_model(name: str) -> bool:
    # intfloat/multilingual-e5-* and intfloat/e5-* all use the query:/passage: scheme.
    return "e5" in name.lower()


def _query_prefix() -> str:
    """Prefix prepended to QUERY/lookup text for the active model family."""
    if _is_e5_model(LOCAL_EMBEDDING_MODEL):
        return _E5_QUERY_PREFIX
    if _is_bge_model(LOCAL_EMBEDDING_MODEL):
        return _BGE_QUERY_PREFIX
    return ""


def _passage_prefix() -> str:
    """Prefix prepended to PASSAGE/stored text. Only e5 wants one."""
    if _is_e5_model(LOCAL_EMBEDDING_MODEL):
        return _E5_PASSAGE_PREFIX
    return ""


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
    """Embed a single PASSAGE (document chunk / stored text)."""
    cleaned = (text or "").replace("\n", " ").strip()
    if not cleaned:
        return [0.0] * EMBEDDING_DIMENSIONS
    cleaned = _passage_prefix() + cleaned
    vectors = await asyncio.to_thread(_encode_sync, [cleaned])
    return vectors[0]


async def embed_query(question: str) -> List[float]:
    """
    Embed a single QUERY/lookup. The active model's query prefix (if any) is
    prepended, which materially improves retrieval quality for bge/e5.
    """
    cleaned = (question or "").replace("\n", " ").strip()
    if not cleaned:
        return [0.0] * EMBEDDING_DIMENSIONS
    cleaned = _query_prefix() + cleaned
    vectors = await asyncio.to_thread(_encode_sync, [cleaned])
    return vectors[0]


async def embed_texts(
    texts: List[str],
    batch_size: int = 100,  # kept for API compatibility; ignored
) -> List[List[float]]:
    """Embed many PASSAGES (stored text), applying the passage prefix if any."""
    if not texts:
        return []
    pp = _passage_prefix()
    cleaned = [(t or "").replace("\n", " ").strip() for t in texts]
    safe_batch = [(pp + c) if c else " " for c in cleaned]
    vectors = await asyncio.to_thread(_encode_sync, safe_batch)
    return vectors


async def embed_queries(questions: List[str]) -> List[List[float]]:
    """
    Batch-embed multiple QUERIES (with the model's query prefix on each). One
    thread hop, one model.encode call — ~1.5× the cost of a single query, not N×.
    Used by multi-query expansion in retrieve_chunks and TB account matching.
    """
    if not questions:
        return []
    qp = _query_prefix()
    cleaned = [(q or "").replace("\n", " ").strip() for q in questions]
    cleaned = [(qp + q) if q else " " for q in cleaned]
    vectors = await asyncio.to_thread(_encode_sync, cleaned)
    return vectors
