"""Small in-memory per-IP token bucket for the /copilot endpoints, plus the client-IP helper.

NOTE: state lives in this process only. A multi-instance deploy needs a shared store (e.g. Redis) to enforce one
limit across instances. Tokens are never logged; keys are client IPs held in memory only.
"""
import os
import threading
import time
from typing import Callable, Optional

from fastapi import HTTPException, Request

MAX_KEYS = 10_000


def _num(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.getenv(name) or default))
    except ValueError:
        return default


def rate_per_min() -> float:
    """FLYRELY_COPILOT_RATE_PER_MIN: sustained requests per minute per IP (default 60; 0 disables)."""
    return _num("FLYRELY_COPILOT_RATE_PER_MIN", 60)


def rate_burst() -> float:
    """FLYRELY_COPILOT_RATE_BURST: bucket size (default 30)."""
    return max(1.0, _num("FLYRELY_COPILOT_RATE_BURST", 30))


def trusted_proxies() -> set[str]:
    """FLYRELY_COPILOT_TRUSTED_PROXIES: comma-separated proxy IPs. Empty (default) = never trust X-Forwarded-For."""
    return {p.strip() for p in (os.getenv("FLYRELY_COPILOT_TRUSTED_PROXIES") or "").split(",") if p.strip()}


def client_ip(request: Request) -> str:
    peer = request.client.host if request.client else "unknown"
    trusted = trusted_proxies()
    if peer in trusted:
        hops = [h.strip() for h in request.headers.get("x-forwarded-for", "").split(",") if h.strip()]
        for hop in reversed(hops):   # rightmost entry that is not one of our own proxies
            if hop not in trusted:
                return hop
    return peer


class RateLimiter:
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._buckets: dict[str, tuple[float, float]] = {}   # ip -> (tokens, last)
        self._lock = threading.Lock()

    def allow(self, key: str, per_min: float, burst: float) -> bool:
        now = self._clock()
        with self._lock:
            tokens, last = self._buckets.get(key, (burst, now))
            tokens = min(burst, tokens + (now - last) * per_min / 60.0)
            ok = tokens >= 1.0
            if ok:
                tokens -= 1.0
            if len(self._buckets) >= MAX_KEYS and key not in self._buckets:
                full = [k for k, (t, _) in self._buckets.items() if t >= burst]
                for k in full or list(self._buckets)[: MAX_KEYS // 2]:
                    self._buckets.pop(k, None)
            self._buckets[key] = (tokens, now)
            return ok


def rate_limit(request: Request) -> None:
    """FastAPI dependency for every /copilot route."""
    per_min = rate_per_min()
    if per_min <= 0:
        return
    state = request.app.state
    limiter: Optional[RateLimiter] = getattr(state, "copilot_limiter", None)
    if limiter is None:
        limiter = state.copilot_limiter = RateLimiter()
    if not limiter.allow(client_ip(request), per_min, rate_burst()):
        raise HTTPException(status_code=429, detail="rate limit exceeded", headers={"Retry-After": "10"})
