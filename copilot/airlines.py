"""Manage-booking link and customer-service phone for the ticketing (marketing) airline, from copilot/data/airlines.json.

Hand-curated (docs/PLAN_DISRUPTION_COPILOT.md §3.4). Lookup is by marketing code only: regional operators (OO, YX,
MQ, ...) return None. A carrier whose manage-trip URL or phone is not verified also returns None, so the card falls
back to "Contact your airline" instead of showing a guessed value.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

from .schemas import AirlineInfo

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
    if not (entry.get("manage_booking_url") and entry.get("phone") and entry.get("verified_on")):
        return None
    return AirlineInfo(
        code=code.strip().upper(),
        name=entry["name"],
        manage_booking_url=entry["manage_booking_url"],
        phone=entry["phone"],
        phone_display=entry["phone_display"],
        verified_on=entry["verified_on"],
    )


__all__ = ["lookup", "AIRLINES_PATH"]
