"""
Thread-safe in-memory answer cache.

When the same question is asked again within the TTL window, we serve the
previously computed answer instantly — no embedding, no retrieval, no
reranking, no Gemini call. Typical real-world cache hit rate in an audit
firm is 20-40% (auditors ask similar questions across files), so this
gives a meaningful speedup for the most repetitive questions.

Cache invalidation: cleared whenever an admin uploads or deletes a
document (the knowledge base changed, so old answers may be stale).
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Optional

logger = logging.getLogger(__name__)


CACHE_TTL_SECONDS = 3600   # 1 hour
CACHE_MAX_ENTRIES = 500    # rough cap so memory stays bounded


class AnswerCache:
    """Bounded LRU cache with per-entry TTL."""

    def __init__(
        self,
        max_entries: int = CACHE_MAX_ENTRIES,
        ttl_seconds: int = CACHE_TTL_SECONDS,
    ) -> None:
        self._max = max_entries
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        # key -> (inserted_at_epoch, payload_dict)
        self._store: "OrderedDict[str, tuple[float, dict]]" = OrderedDict()

    # ---- key helpers ----

    @staticmethod
    def _normalize(question: str) -> str:
        return " ".join((question or "").lower().split())

    @classmethod
    def _key(cls, question: str, language: str) -> str:
        norm = cls._normalize(question)
        lang = (language or "en").lower()
        return hashlib.sha256(f"{lang}|{norm}".encode("utf-8")).hexdigest()

    # ---- core API ----

    def get(self, question: str, language: str) -> Optional[dict]:
        key = self._key(question, language)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            inserted_at, payload = entry
            if time.time() - inserted_at > self._ttl:
                # Expired
                del self._store[key]
                return None
            # Refresh LRU position
            self._store.move_to_end(key)
            return payload

    def set(self, question: str, language: str, payload: dict) -> None:
        key = self._key(question, language)
        with self._lock:
            if key in self._store:
                del self._store[key]
            self._store[key] = (time.time(), payload)
            self._store.move_to_end(key)
            # Evict oldest while over the cap
            while len(self._store) > self._max:
                evicted_key, _ = self._store.popitem(last=False)
                logger.debug("Cache evicted oldest entry %s", evicted_key[:8])

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
        logger.info("Answer cache cleared.")

    def size(self) -> int:
        with self._lock:
            return len(self._store)


# Module-level singleton — shared by /ask and /ask/stream
answer_cache = AnswerCache()
