"""Compose the POST /copilot/options response (plan §4.2). Pure functions: the scorer, schedule, clock and env
are all passed in, so tests need no model, no network and no real clock.

Privacy: `home_airport` is only used to compute `context`. This module never logs anything.
"""
import csv
import threading
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Callable, Mapping, Optional
from zoneinfo import ZoneInfo

from fastapi import HTTPException

from . import airlines, context, links
from .schedule import BtsTypicalSchedule, ScheduleSource
from .schemas import (
    AirlineNotice, AlternativeItem, Alternatives, Disclosure, Handoff, Hotel, OptionsRequest, OptionsResponse, RefundRights, Ride,
    RideLink,
)

AIRPORTS_REF = Path(__file__).resolve().parent.parent / "models" / "airports_ref.csv"
HOTEL_CHECKIN_CUTOFF = time(3, 0)

_default_schedule: Optional[BtsTypicalSchedule] = None
_default_lock = threading.Lock()


def default_schedule() -> BtsTypicalSchedule:
    """Process-wide lazy schedule (the CSV is read on the first call, not at import)."""
    global _default_schedule
    with _default_lock:
        if _default_schedule is None:
            _default_schedule = BtsTypicalSchedule()
        return _default_schedule


@dataclass(frozen=True)
class Airport:
    iata: str
    lat: float
    lon: float
    tz: str
    state: str


@lru_cache(maxsize=1)
def _airports() -> dict[str, Airport]:
    out: dict[str, Airport] = {}
    with open(AIRPORTS_REF, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            out[r["iata"].upper()] = Airport(r["iata"].upper(), float(r["lat"]), float(r["lon"]), r["tz"],
                                             (r.get("state") or "").strip())
    return out


def _airport(code: str) -> Airport:
    ap = _airports().get(code.strip().upper())
    if ap is None:
        raise HTTPException(status_code=422, detail={"reason": "unknown_airport", "field": "flight"})
    return ap


def _flight_number(airline: Optional[str], number: str) -> str:
    n = number.strip().upper().replace(" ", "")
    return airline.upper() + n if airline and n.isdigit() else n


def _airline_name(code: str) -> Optional[str]:
    info = airlines.lookup(code)
    return info.name if info else None


def _handoff(link: links.Link) -> Handoff:
    d = link.to_dict()
    d.pop("accepts_dropoff_address", None)
    return Handoff(**d)


def _combine(day: date, hhmm: str) -> datetime:
    h, m = hhmm.split(":")[:2]
    return datetime.combine(day, time(int(h), int(m)))


def compose(
    req: OptionsRequest,
    *,
    score: Optional[Callable],
    schedule: ScheduleSource,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    env: Mapping[str, str] = None,
) -> OptionsResponse:
    import os
    env = os.environ if env is None else env
    f = req.flight
    try:
        dep = datetime.fromisoformat(f.departure_time)
    except ValueError:
        raise HTTPException(status_code=422, detail={"reason": "invalid_departure_time"})
    if dep.tzinfo is not None:
        dep = dep.replace(tzinfo=None)
    if req.rebooked is not None and req.state != "rebooked":
        raise HTTPException(status_code=422, detail={"reason": "rebooked_requires_state_rebooked"})
    origin, dest = _airport(f.origin), _airport(f.destination)
    if origin.iata == dest.iata:
        raise HTTPException(status_code=422, detail={"reason": "same_airport"})

    flight_day = dep.date()
    warnings: list[str] = []

    # airline = the ticketing (marketing) carrier the user BOOKED with. Contact details (phone, manage-booking URL) are
    # always that carrier's, even when a regional partner operates the flight; operating_airline_name on alternatives
    # is informational only.
    code = (f.airline or "").strip().upper() or (f.flight_number.strip().upper()[:2] if f.flight_number[:2].isalpha() else "")
    airline_info = airlines.lookup(code)
    notice = None
    if airline_info is None:
        entry = airlines.get_entry(code)
        if entry and (entry.get("user_note") or entry.get("status_url")):
            notice = AirlineNotice(code=code, name=entry.get("name") or code, note=entry.get("user_note"),
                                   status_url=entry.get("status_url"))
        else:
            warnings.append("unknown_airline")
    own = _flight_number(code or None, f.flight_number)

    ctx = context.location(req.home_airport, origin.iata)
    # airports_ref is US-only: an empty `state` means a territory (SJU, GUM, ...), which counts as international here.
    international = not (origin.state and dest.state)
    refund = RefundRights(applies=bool(origin.state or dest.state), jurisdiction=None, content_key=None,
                          international=international, eu_possible=None)
    if refund.applies:
        refund = RefundRights(applies=True, jurisdiction="US_DOT", content_key="us_dot_part260_v2",
                              international=international, eu_possible=None)

    # --- alternatives ---
    search = [_handoff(lk) for lk in links.flight_links(origin.iata, dest.iata, flight_day, env)]
    alt_warnings: list[str] = []
    items: list[AlternativeItem] = []
    if not refund.applies:
        alt_warnings.append("non_us_route")
    else:
        now_local = now().astimezone(ZoneInfo(origin.tz)).replace(tzinfo=None)
        cands = []
        for sf in schedule.flights_on(origin.iata, dest.iata, flight_day):
            if own in (sf.flight_number, sf.operating_flight_number):
                continue
            dep_l = _combine(flight_day, sf.dep_time)
            if dep_l <= now_local:
                continue
            arr_l = _combine(flight_day, sf.arr_time)
            if arr_l < dep_l:
                arr_l += timedelta(days=1)
            cands.append((sf, dep_l, arr_l))

        scores: list[Optional[tuple[float, str]]] = [None] * len(cands)
        if cands:
            try:
                if score is None:
                    raise RuntimeError("no scorer")
                scores = [score(sf.origin, sf.dest, dep_l, sf.marketing_airline) for sf, dep_l, _ in cands]
            except Exception:
                scores = [None] * len(cands)
                warnings.append("risk_unavailable")
        rows = sorted(zip(cands, scores),
                      key=lambda r: (r[1][0] if r[1] else 0.0, r[0][1]))
        # one row per (carrier, flight number): the schedule can list the same flight at slightly different times;
        # keep the one seen on the most days.
        best: dict[tuple[str, str], int] = {}
        for i, ((sf, _, _), _sc) in enumerate(rows):
            k = (sf.marketing_airline, sf.flight_number)
            if k not in best or sf.days_seen > rows[best[k]][0][0].days_seen:
                best[k] = i
        rows = [r for i, r in enumerate(rows) if best[(r[0][0].marketing_airline, r[0][0].flight_number)] == i]
        for (sf, dep_l, arr_l), sc in rows[:req.max_alternatives]:
            items.append(AlternativeItem(
                marketing_airline=sf.marketing_airline, flight_number=sf.flight_number,
                operating_airline=sf.operating_airline,
                operating_airline_name=_airline_name(sf.operating_airline),
                origin=sf.origin, destination=sf.dest,
                dep_local=dep_l.isoformat(), arr_local=arr_l.isoformat(),
                delay_probability=round(float(sc[0]), 4) if sc else None,
                risk_level=sc[1] if sc else None,
                days_seen=sf.days_seen,
                handoffs=search,
            ))
        if not items:
            alt_warnings.append("no_alternatives_found")

    alternatives = Alternatives(
        source=schedule.source, schedule_as_of=schedule.schedule_as_of(), date=flight_day.isoformat(),
        items=items, search_handoffs=search, warnings=alt_warnings)

    # --- hotel and ride ---
    hotel = None
    needs_hotel = ctx.location == "away" and (req.state == "cancelled" or (req.state == "rebooked" and req.rebooked == "later"))
    if needs_hotel:
        now_o = now().astimezone(ZoneInfo(origin.tz)).replace(tzinfo=None)
        # "tonight" in the origin's local time: before 03:00 we are still on the previous night.
        tonight = now_o.date() - timedelta(days=1) if now_o.time() < HOTEL_CHECKIN_CUTOFF else now_o.date()
        check_in = max(tonight, flight_day)
        check_out = check_in + timedelta(days=1)
        hotel = Hotel(airport=origin.iata, check_in=check_in.isoformat(), check_out=check_out.isoformat(),
                      links=[_handoff(lk) for lk in links.hotel_links(origin.iata, origin.lat, origin.lon,
                                                                       check_in, check_out, env)])
    purpose = None
    if ctx.location == "away":
        purpose = "to_hotel"
    elif ctx.location == "home":
        purpose = "home"
    if req.state == "rebooked" and req.rebooked != "later":
        purpose = None   # rebooked today: no ride card
    ride = Ride(purpose=purpose, links=[])
    if purpose:
        ride = Ride(purpose=purpose, links=[RideLink(**lk.to_dict()) for lk in links.ride_links(env=env)])

    all_links = search + (hotel.links if hotel else []) + ride.links
    return OptionsResponse(
        state=req.state, context=ctx, airline=airline_info, airline_notice=notice, refund_rights=refund, alternatives=alternatives,
        hotel=hotel, ride=ride, disclosure=Disclosure(required=any(h.is_affiliate for h in all_links)),
        warnings=warnings)
