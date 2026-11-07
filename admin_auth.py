"""Admin-only access for /usage and /usage/export.

The caller sends `Authorization: Bearer <Supabase access token>`. We validate it by asking Supabase
(GET {SUPABASE_URL}/auth/v1/user with the anon key), cache the answer for ~60 s, and allow only emails listed in
FLYRELY_ADMIN_EMAILS (comma-separated, case-insensitive). Missing config fails closed.

Env vars are read per request so they can change without a code reload (and so tests can set them).
"""

import hashlib
import logging
import os
import time
from typing import Optional

import httpx
from fastapi import Header, HTTPException

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 60
CACHE_MAX = 256
SUPABASE_TIMEOUT_SECONDS = 5.0

# sha256(token) -> (expiry monotonic time, email). Only successful validations are cached.
_cache: dict[str, tuple[float, str]] = {}


def clear_cache() -> None:
    _cache.clear()


def admin_emails() -> set[str]:
    return {e.strip().lower() for e in os.getenv("FLYRELY_ADMIN_EMAILS", "").split(",") if e.strip()}


def _bearer(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    token = token.strip()
    return token if scheme.lower() == "bearer" and token else None


async def _lookup_email(token: str) -> str:
    """Email of the Supabase user behind `token`. 401 if Supabase rejects it, 503 if it can't be reached."""
    url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
    anon = os.getenv("SUPABASE_ANON_KEY", "").strip()
    if not url or not anon:
        logger.error("SUPABASE_URL / SUPABASE_ANON_KEY not set; admin endpoints are closed")
        raise HTTPException(status_code=403, detail="Admin access is not configured")

    key = hashlib.sha256(token.encode()).hexdigest()
    hit = _cache.get(key)
    if hit and hit[0] > time.monotonic():
        return hit[1]

    try:
        async with httpx.AsyncClient(timeout=SUPABASE_TIMEOUT_SECONDS) as client:
            resp = await client.get(f"{url}/auth/v1/user",
                                    headers={"Authorization": f"Bearer {token}", "apikey": anon})
    except httpx.HTTPError as e:
        logger.error("Supabase auth lookup failed: %s", e)
        raise HTTPException(status_code=503, detail="Could not verify sign-in; try again")
    if resp.status_code in (401, 403):
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    if resp.status_code != 200:
        logger.error("Supabase auth lookup returned %s", resp.status_code)
        raise HTTPException(status_code=503, detail="Could not verify sign-in; try again")
    try:
        user = resp.json()
    except ValueError:
        raise HTTPException(status_code=503, detail="Could not verify sign-in; try again")
    email = (user.get("email") or "").strip().lower() if isinstance(user, dict) else ""
    # An address nobody has confirmed must not unlock admin access.
    confirmed = isinstance(user, dict) and (user.get("email_confirmed_at") or user.get("confirmed_at"))
    if not email or not confirmed:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    if len(_cache) >= CACHE_MAX:
        now = time.monotonic()
        for k in [k for k, v in _cache.items() if v[0] <= now] or list(_cache)[:CACHE_MAX // 2]:
            _cache.pop(k, None)
    _cache[key] = (time.monotonic() + CACHE_TTL_SECONDS, email)
    return email


async def require_admin(authorization: Optional[str] = Header(default=None)) -> str:
    """FastAPI dependency: returns the admin's email, or raises 401 / 403 (fail closed)."""
    token = _bearer(authorization)
    if token is None:
        raise HTTPException(status_code=401, detail="Sign in required", headers={"WWW-Authenticate": "Bearer"})
    allowed = admin_emails()
    if not allowed:
        logger.error("FLYRELY_ADMIN_EMAILS not set; admin endpoints are closed")
        raise HTTPException(status_code=403, detail="Admin access is not configured")
    email = await _lookup_email(token)
    if email not in allowed:
        raise HTTPException(status_code=403, detail="Admin only")
    return email
