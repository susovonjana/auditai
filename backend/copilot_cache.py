"""
Short-TTL, thread-safe in-memory cache for the copilot's file-data fetches.

The file-mode chat tool loop runs in a worker thread and re-fetches the same
audit-file data (trial balance, summary, risks, …) on every question. This LRU
cache, keyed by ``audit_file_id:endpoint:args`` with a per-entry TTL, lets a burst
of questions reuse one fetch instead of re-querying 1audit-be each time.

Bounded staleness by design: an entry lives at most ``ttl_seconds`` from when it
was fetched, so an edit upstream is reflected within that window. The grant is
re-validated once per chat request (see CopilotContext.validate_grant), so a
cache hit never serves data to an unauthorized request.
"""
import threading
import time
from collections import OrderedDict
from typing import Any, Optional


class TTLCache:
    """Bounded LRU cache with a per-entry, fetch-time TTL. Thread-safe."""

    def __init__(self, ttl_seconds: float, max_entries: int = 256):
        self._ttl = float(ttl_seconds)
        self._max = int(max_entries)
        self._store: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._ttl > 0

    def get(self, key: str) -> Optional[Any]:
        """Return the cached value if present and still fresh, else None."""
        if self._ttl <= 0:
            return None
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            inserted_at, value = entry
            if time.time() - inserted_at > self._ttl:
                del self._store[key]
                return None
            self._store.move_to_end(key)  # mark recently used
            return value

    def set(self, key: str, value: Any) -> None:
        if self._ttl <= 0:
            return
        with self._lock:
            if key in self._store:
                del self._store[key]
            self._store[key] = (time.time(), value)
            self._store.move_to_end(key)
            while len(self._store) > self._max:
                self._store.popitem(last=False)  # evict least-recently-used

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def clear_prefix(self, prefix: str) -> int:
        """Drop every entry whose key starts with ``prefix``. Used to invalidate
        all of ONE audit file's cached fetches (keys are ``<id>:<endpoint>:<args>``)
        when 1audit reports that file changed. Returns the number removed."""
        with self._lock:
            doomed = [k for k in self._store if k.startswith(prefix)]
            for k in doomed:
                del self._store[k]
            return len(doomed)

    def size(self) -> int:
        with self._lock:
            return len(self._store)
