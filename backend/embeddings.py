"""
Local embeddings via sentence-transformers (FREE).

The model is downloaded once to ~/.cache/huggingface (about 80 MB for
all-MiniLM-L6-v2) and then runs entirely on your machine — no API key,
no rate limits, no internet needed after the first download.

Output dimension defaults to 384 (configured in config.EMBEDDING_DIMENSIONS).
"""
from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

from config import EMBEDDING_DIMENSIONS, LOCAL_EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# Lazy singleton — loading the model takes a couple of seconds, so we only
# do it on the first call and keep it in memory afterwards.
_model: Optional["SentenceTransformer"] = None  # type: ignore[name-defined]


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
    """Return a single embedding vector for a single string."""
    cleaned = (text or "").replace("\n", " ").strip()
    if not cleaned:
        return [0.0] * EMBEDDING_DIMENSIONS

    vectors = await asyncio.to_thread(_encode_sync, [cleaned])
    return vectors[0]


async def embed_texts(
    texts: List[str],
    batch_size: int = 100,  # kept for API compatibility; ignored
) -> List[List[float]]:
    """
    Embed many strings. Returns a list with the same length and order
    as the input.
    """
    if not texts:
        return []

    cleaned = [(t or "").replace("\n", " ").strip() for t in texts]
    safe_batch = [c if c else " " for c in cleaned]

    vectors = await asyncio.to_thread(_encode_sync, safe_batch)
    return vectors
