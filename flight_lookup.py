"""GET /flights/lookup: airline + flight number + date -> the leg(s) that flight usually flies.

Backed by the BTS "typical schedule" (models/route_schedule.csv.gz, read through copilot.schedule.BtsTypicalSchedule).
It reflects the last months of BTS data, not today's filed schedule, so times can be off after a seasonal change.
The copilot package is imported lazily: with the copilot flag off, importing `main` must not import it.

TODO(live linkage): for a date within ~24h, prefer the live FAA TFMS schedule. There is no simple reader on master yet
(swim/ only archives raw messages); another agent is building data/linkage/live.sqlite. Wire it in here when it lands.
"""
import os
import re
import threading
from datetime import date as date_cls, timedelta
from typing import Optional

from fastapi import HTTPException, Query, Request

import codeshare

SOURCE = "bts_typical_schedule"
DEFUNCT = {"NK": "Spirit Airlines no longer operates, so there is no flight to look up. "
                 "You can still enter the details manually."}
# Hawaiian now flies under Alaska codes. Used only if the schedule has no HA rows of its own.
RENAMED = {"HA": "AS"}
_FLIGHT_RE = re.compile(r"^\s*([A-Z]{2}|[A-Z]\d|\d[A-Z])?\s*0*(\d{1,4})\s*$")
_AIRLINE_RE = re.compile(r"^[A-Z0-9]{2}$")

_schedule = None
_schedule_lock = threading.Lock()
_limiter = None


def get_schedule():
    global _schedule
    if _schedule is None:
        with _schedule_lock:
            if _schedule is None:
                from copilot.schedule import BtsTypicalSchedule
                _schedule = BtsTypicalSchedule()
    return _schedule


def parse_query(flight: str, airline: Optional[str] = None) -> tuple:
    """'UA648' / 'UA 648' / ('648', airline='UA') -> ('UA', '648'). Raises ValueError on bad input."""
    flight = (flight or "").upper()
    airline = (airline or "").strip().upper() or None
    if airline and not _AIRLINE_RE.match(airline):
        raise ValueError("airline must be a 2-character code such as UA")
    m = _FLIGHT_RE.match(flight)
    if not m:
        raise ValueError("flight must look like UA648 or UA 648")
    code = m.group(1) or airline
    if not code:
        raise ValueError("include the airline code, e.g. UA648")
    if airline and m.group(1) and airline != m.group(1):
        raise ValueError("airline does not match the flight number's airline code")
    if m.group(2) == "":
        raise ValueError("flight number is missing")
    return code, m.group(2)


def _rate_limit(request: Request) -> None:
    global _limiter
    try:
        per_min = max(0.0, float(os.getenv("FLYRELY_LOOKUP_RATE_PER_MIN") or 30))
    except ValueError:
        per_min = 30.0
    if per_min <= 0:
        return
    from copilot.ratelimit import RateLimiter, client_ip
    if _limiter is None:
        _limiter = RateLimiter()
    if not _limiter.allow(client_ip(request), per_min, max(1.0, per_min / 2)):
        raise HTTPException(status_code=429, detail="rate limit exceeded", headers={"Retry-After": "10"})


def _leg(f) -> dict:
    op = f.operating_airline
    return {
        "origin": f.origin,
        "dest": f.dest,
        "dep_time": f.dep_time,
        "arr_time": f.arr_time,
        "arr_next_day": f.arr_time < f.dep_time,
        "operating_airline": op,
        "operating_airline_name": codeshare.OPERATOR_NAMES.get(op, op),
        "operating_flight_number": f.operating_flight_number,
        "days_seen": f.days_seen,
    }


def lookup_flight(request: Request,
                  flight: str = Query(..., max_length=12, description="UA648, 'UA 648', or just the number with `airline`"),
                  date: str = Query(..., description="Local departure date, YYYY-MM-DD"),
                  airline: Optional[str] = Query(None, max_length=2)):
    _rate_limit(request)
    try:
        code, number = parse_query(flight, airline)
        day = date_cls.fromisoformat(date)
    except ValueError as e:
        raise HTTPException(status_code=422, detail={"reason": "invalid_input", "message": str(e)})
    today = date_cls.today()
    if not (today - timedelta(days=7) <= day <= today + timedelta(days=365)):
        raise HTTPException(status_code=422, detail={"reason": "invalid_input",
                                                     "message": "date must be within the next year (or the last week)"})

    if code in DEFUNCT:
        raise HTTPException(status_code=410, detail={"reason": "airline_defunct", "message": DEFUNCT[code]})

    sched = get_schedule()
    notes = []
    searched = code
    legs = sched.flights_by_number(code, number, day)
    if not legs and code in RENAMED:
        searched = RENAMED[code]
        legs = sched.flights_by_number(searched, number, day)
        notes.append(f"Hawaiian Airlines now flies under Alaska Airlines (AS) codes, so we looked up AS{number}. "
                     "Flight numbers can differ after the switch; check your booking.")
    if not legs:
        any_day = [f for d in range(7) for f in sched.flights_by_number(searched, number, day + timedelta(days=d))]
        why = (f"{searched}{number} is in the schedule, but not usually on that weekday." if any_day else
               f"We could not find {code}{number} in the typical schedule.")
        raise HTTPException(status_code=404, detail={
            "reason": "not_found",
            "message": f"{why} Check the flight number and date, or enter the details manually.",
            "schedule_as_of": sched.schedule_as_of()})

    as_of = sched.schedule_as_of()
    notes.append(f"Typical schedule from recent months (as of {as_of}). Airlines change schedules seasonally, "
                 "so confirm the times with your booking.")
    return {
        "flight": f"{searched}{number}",
        "airline": searched,
        "flight_number": number,
        "date": day.isoformat(),
        "matches": [_leg(f) for f in legs],
        "source": SOURCE,
        "confidence": "typical_schedule",
        "schedule_as_of": as_of,
        "note": " ".join(notes),
    }


def mount(app) -> None:
    app.add_api_route("/flights/lookup", lookup_flight, methods=["GET"], tags=["flights"])
