"""
Cross-encoder reranker.

Vector search retrieves the "most similar" chunks; a cross-encoder scores
how *useful* each chunk actually is for answering the question. Running
the cross-encoder on the top 20 retrieved chunks and keeping the top K
typically improves answer accuracy by 15-25%.

Default model: cross-encoder/ms-marco-MiniLM-L-6-v2 (~80 MB)
"""
from __future__ import annotations

import asyncio
import logging
from typing import List, Tuple

from config import RERANKER_MODEL

logger = logging.getLogger(__name__)

_reranker: object | None = None  # CrossEncoder instance


def _load_reranker():
    """Lazy load the cross-encoder model. Sync — call from a worker thread."""
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is not installed."
            ) from exc

        logger.info("Loading reranker model '%s' (first time downloads ~80 MB)...", RERANKER_MODEL)
        _reranker = CrossEncoder(RERANKER_MODEL)
        logger.info("Reranker ready.")
    return _reranker


def _rerank_sync(question: str, passages: List[str]) -> List[float]:
    """Score each (question, passage) pair. Returns a list of float scores."""
    model = _load_reranker()
    pairs = [[question, p] for p in passages]
    scores = model.predict(pairs)
    return [float(s) for s in scores]


async def rerank(
    question: str, passages: List[str]
) -> List[float]:
    """Async wrapper. Returns relevance scores aligned with `passages`."""
    if not passages:
        return []
    return await asyncio.to_thread(_rerank_sync, question, passages)
