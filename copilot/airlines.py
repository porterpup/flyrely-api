"""Manage-booking link and customer-service phone for the ticketing (marketing) airline, from copilot/data/airlines.json.

Hand-curated (docs/PLAN_DISRUPTION_COPILOT.md §3.4). Lookup is by marketing code only: regional operators (OO, YX,
MQ, ...) return None. A carrier with status `unknown` (or a missing URL/phone) returns None; `verified` and `official_source_unconfirmed` are shown, so the card falls
back to "Contact your airline" instead of showing a guessed value.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

from .schemas import AirlineInfo

SHOWN_STATUSES = ("verified", "official_source_unconfirmed")
AIRLINES_PATH = Path(__file__).resolve().parent / "data" / "airlines.json"


@lru_cache(maxsize=1)
def _load(path: Path = AIRLINES_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["airlines"]


def lookup(code: Optional[str], path: Path = AIRLINES_PATH) -> Optional[AirlineInfo]:
    """Return the AirlineInfo for a marketing code (e.g. "DL"), or None if unknown or unverified."""
    if not code:
        return None
    entry = _load(path).get(code.strip().upper())
    if not entry:
        return None
    verification = entry.get("verification") or ("verified" if entry.get("verified_on") else "unknown")
    if verification not in SHOWN_STATUSES:
        return None
    if not (entry.get("manage_booking_url") and entry.get("phone")):
        return None
    return AirlineInfo(
        code=code.strip().upper(),
        name=entry["name"],
        manage_booking_url=entry["manage_booking_url"],
        phone=entry["phone"],
        phone_display=entry["phone_display"],
        verified_on=entry.get("verified_on"),
        verification=verification,
        other_channels=entry.get("other_channels") or {},
    )


def get_entry(code: Optional[str], path: Path = AIRLINES_PATH) -> Optional[dict]:
    """Raw JSON entry (incl. unknown carriers like NK with status_url and note), or None if the code is absent."""
    return _load(path).get(code.strip().upper()) if code else None


__all__ = ["lookup", "get_entry", "AIRLINES_PATH"]
