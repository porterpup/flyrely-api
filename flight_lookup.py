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
from datetime import date as date_cls, datetime, timedelta
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
    except ValueError as e:
        raise _invalid(str(e))
    day = _check_date(date)

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
        if any_day:
            message = (f"{searched}{number} is in the schedule, but not usually on that weekday. "
                       "Check the flight number and date, or enter the details manually.")
        else:
            message = NOT_FOUND_MESSAGE
        raise HTTPException(status_code=404, detail={
            "reason": "not_found",
            "message": message,
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


NOT_FOUND_MESSAGE = ("We couldn't find this flight. FlyRely currently covers US domestic flights only. "
                     "If it's a domestic flight, you can enter the details manually.")

_IATA_RE = re.compile(r"^[A-Z]{3}$")
_HHMM_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
MAX_SEARCH_RESULTS = 100


def _invalid(message: str):
    return HTTPException(status_code=422, detail={"reason": "invalid_input", "message": message})


def _check_date(date: str) -> date_cls:
    try:
        day = date_cls.fromisoformat(date)
    except ValueError:
        raise _invalid("date must look like 2026-11-10")
    today = date_cls.today()
    if not (today - timedelta(days=7) <= day <= today + timedelta(days=365)):
        raise _invalid("date must be within the next year (or the last week)")
    return day


def make_search(score_many, scored_airlines, unavailable, known_airports=None, factors_for=None):
    """Build the /flights/search handler. `score_many([(origin, dest, dep_dt, airline)]) -> [(prob, risk)]`.
    `known_airports`: US airport codes FlyRely covers (None = no check). `factors_for(origin, dest, dep_dt, airline)`
    returns plain-language reasons for a row."""

    def search_flights(request: Request,
                       origin: str = Query(..., max_length=3), dest: str = Query(..., max_length=3),
                       date: str = Query(...),
                       after: Optional[str] = Query(None, description="Earliest local departure, HH:MM"),
                       before: Optional[str] = Query(None, description="Latest local departure, HH:MM")):
        _rate_limit(request)
        o, d = origin.strip().upper(), dest.strip().upper()
        if not (_IATA_RE.match(o) and _IATA_RE.match(d)):
            raise _invalid("origin and dest must be 3-letter airport codes such as EWR")
        if o == d:
            raise _invalid("origin and dest must differ")
        for name, v in (("after", after), ("before", before)):
            if v and not _HHMM_RE.match(v):
                raise _invalid(f"{name} must look like 06:30")
        day = _check_date(date)
        if known_airports is not None:
            unknown = [c for c in (o, d) if c not in known_airports]
            if unknown:
                raise HTTPException(status_code=404, detail={
                    "reason": "unknown_airport",
                    "message": f"We don't recognise {' or '.join(unknown)}. FlyRely currently covers US "
                               "domestic flights only, so check the airport codes."})

        sched = get_schedule()
        legs = sched.flights_on(o, d, day)
        if after:
            legs = [f for f in legs if f.dep_time >= after]
        if before:
            legs = [f for f in legs if f.dep_time <= before]
        if not legs:
            raise HTTPException(status_code=404, detail={
                "reason": "not_found",
                "message": f"We found no flights from {o} to {d} on that date in the typical schedule. "
                           "Check the airports, or enter the details manually.",
                "schedule_as_of": sched.schedule_as_of()})
        legs = legs[:MAX_SEARCH_RESULTS]

        items = []
        for f in legs:
            model_airline = f.operating_airline if f.operating_airline in scored_airlines else f.marketing_airline
            items.append((o, d, datetime.fromisoformat(f"{day.isoformat()}T{f.dep_time}:00"), model_airline))
        try:
            scores = score_many(items)
        except unavailable as e:
            raise HTTPException(status_code=503, detail={"reason": getattr(e, "reason", "prediction_failed"),
                                                         "message": getattr(e, "message", str(e))})
        as_of = sched.schedule_as_of()
        flights = []
        for f, (prob, risk) in zip(legs, scores):
            row = _leg(f)
            row.update({"flight": f.flight_number, "airline": f.marketing_airline,
                        "airline_name": codeshare.OPERATOR_NAMES.get(f.marketing_airline, f.marketing_airline),
                        "flight_number": f.flight_number[len(f.marketing_airline):],
                        "delay_probability": round(prob, 3), "risk_level": risk})
            if factors_for is not None:
                model_airline = f.operating_airline if f.operating_airline in scored_airlines else f.marketing_airline
                row["risk_factors"] = factors_for(o, d, datetime.fromisoformat(f"{day.isoformat()}T{f.dep_time}:00"),
                                                  model_airline)
            flights.append(row)
        return {
            "origin": o, "dest": d, "date": day.isoformat(), "flights": flights,
            "source": SOURCE, "confidence": "typical_schedule", "schedule_as_of": as_of,
            "note": f"Based on recent schedules (as of {as_of}). Airlines change times seasonally, "
                    "so confirm the time on your booking.",
        }

    return search_flights


def mount(app, score_many=None, scored_airlines=(), unavailable=Exception, known_airports=None,
          factors_for=None) -> None:
    app.add_api_route("/flights/lookup", lookup_flight, methods=["GET"], tags=["flights"])
    if score_many is not None:
        app.add_api_route("/flights/search", make_search(score_many, set(scored_airlines), unavailable, known_airports, factors_for),
                          methods=["GET"],
                          tags=["flights"])
