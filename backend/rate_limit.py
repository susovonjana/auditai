"""
Rate-limiting setup.

A single `limiter` is shared across all routers. Each protected endpoint
declares its own limit via the `@limiter.limit(...)` decorator.

Keying is by remote IP. Behind a reverse proxy (nginx, Caddy), make sure
the proxy forwards X-Forwarded-For so slowapi sees the real client IP,
not the proxy's IP.
"""
from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address


# Conservative default for any unannotated endpoint.
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["1000/day"],
)
