"""Global rate limiter (slowapi).

Per-IP limits protect slow/expensive endpoints from overwhelm. Disable entirely
for local development or CI with ``export DISABLE_RATELIMIT=1``.

Wire-up lives in ``backend/main.py``:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

Endpoints opt in with ``@limiter.limit("N/minute")`` and must accept a
``request: Request`` parameter (slowapi reads the client IP from it).
"""

from __future__ import annotations

import os

from slowapi import Limiter
from slowapi.util import get_remote_address

RATELIMIT_DISABLED: bool = os.getenv("DISABLE_RATELIMIT", "").lower() in (
    "1",
    "true",
    "yes",
)

limiter = Limiter(key_func=get_remote_address, enabled=not RATELIMIT_DISABLED)
