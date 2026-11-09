"""POST /predict/itinerary: connection risk for a 1-4 leg trip (plan: backtest/PLAN_ARRIVAL_AND_CONNECTIONS.md 2.3).

Mounted by main only when FLYRELY_FEATURE_ITINERARY=1 (route absent otherwise). `mount(app, main_module)` follows the
flight_lookup.mount pattern; the main module supplies the per-leg prediction and the loaded models, so each leg is a
full /predict response (with the `arrival` block) and is logged exactly like /predict.

No partial answers: if the ladder or the dependence file is missing, or any leg fails, the whole call is 503.
Nothing is invented: unvalidated connections get `miss_probability: null` and a reason.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

import codeshare
from flyrely_dist import mct as mct_mod
from flyrely_dist.connections import miss_probability, trip_summary

MAX_LEGS = 4
MAX_LAYOVER_MIN = 24 * 60
VALIDATED_MAX_LAYOVER_MIN = 240
ITIN_VERSION = "itin-v1"
MCT_PATH = Path(__file__).resolve().parent / "models" / "mct_defaults.csv"
# Marketing networks: Hawaiian now flies as Alaska (AS), so HA and AS count as one network.
NETWORK = {"HA": "AS"}
CROSS_AIRLINE_WARNING = ("Separate airlines: if your first flight is late, the second won't wait. "
                         "Allow at least 2–3 hours.")
_mct_table: Optional[dict] = None


class LegIn(BaseModel):
    flight_id: Optional[str] = None
    flight_number: Optional[str] = None
    airline: Optional[str] = None
    origin: str = Field(..., min_length=3, max_length=3)
    destination: str = Field(..., min_length=3, max_length=3)
    departure_time: datetime
    arrival_time: Optional[datetime] = None


class ConnectionIn(BaseModel):
    min_connect_minutes: Optional[int] = Field(None, ge=10, le=240)


class ItineraryRequest(BaseModel):
    itinerary_id: Optional[str] = None
    legs: list[LegIn] = Field(..., min_length=1)
    connections: list[ConnectionIn] = Field(default_factory=list)


def _err(status: int, reason: str, message: str, **extra) -> HTTPException:
    return HTTPException(status_code=status, detail={"reason": reason, "message": message, **extra})


def mct_table() -> dict:
    global _mct_table
    if _mct_table is None:
        try:
            _mct_table = mct_mod.load_table(MCT_PATH)
        except Exception:
            _mct_table = {}     # resolve_mct then uses the frozen lists
    return _mct_table


def network(airline: Optional[str]) -> Optional[str]:
    return NETWORK.get(airline, airline) if airline else None


def risk_band(p: float) -> str:
    """Display bands frozen in the protocol (rule C6): under 5%, 5-15%, 15% and over."""
    return "low" if p < 0.05 else "medium" if p < 0.15 else "high"


def _resolve_arrival(leg: LegIn, schedule) -> Optional[datetime]:
    """Scheduled local arrival from the typical schedule by flight number and date, or None.

    The schedule departure must be within 90 minutes of the leg's, measured on real datetimes, so a match can wrap
    across midnight (a 00:05 entry matches a 23:55 departure on the previous day, which then sets the arrival date).
    Next-day arrival is decided in UTC from both airports' zones (flight_lookup.arrives_next_day)."""
    import flight_lookup
    airline, number = codeshare.parse_flight(leg.airline.upper() if leg.airline else None, leg.flight_number)
    if not airline or not number:
        return None
    origin, dest = leg.origin.upper(), leg.destination.upper()
    want = leg.departure_time.replace(tzinfo=None)
    best = None
    for day in (want.date(), want.date() - timedelta(days=1), want.date() + timedelta(days=1)):
        for f in schedule.flights_by_number(airline, number, day):
            if f.origin != origin or f.dest != dest:
                continue
            dep = datetime.combine(day, datetime.min.time()) + timedelta(minutes=_hhmm(f.dep_time))
            gap = abs((dep - want).total_seconds()) / 60
            if gap <= 90 and (best is None or gap < best[0]):
                best = (gap, day, f)
    if best is None:
        return None
    _, day, f = best
    arr = datetime.combine(day, datetime.min.time()) + timedelta(minutes=_hhmm(f.arr_time))
    if flight_lookup.arrives_next_day(origin, dest, f.dep_time, f.arr_time, day):
        arr += timedelta(days=1)
    return arr


def _hhmm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def mount(app, m) -> None:
    """`m` is the main module: `_predict_core`, `FlightRequest`, `arrival_service`, `prediction_store`, `PredictionUnavailable`."""

    async def predict_itinerary(req: ItineraryRequest, background_tasks: BackgroundTasks):
        svc = m.arrival_service
        if not svc.loaded or svc.dependence is None:
            reason = svc.load_error if not svc.loaded else svc.dependence_error
            raise _err(503, "model_unavailable", f"Itinerary model not loaded: {reason}")

        n = len(req.legs)
        if n > MAX_LEGS:
            raise _err(422, "too_many_legs", f"A trip can have at most {MAX_LEGS} flights.")
        if req.connections and len(req.connections) != n - 1:
            raise _err(422, "invalid_connections", "connections must have one entry per connection (legs - 1).")
        conns_in = req.connections or [ConnectionIn() for _ in range(n - 1)]

        legs = [(l.origin.upper(), l.destination.upper(), l.departure_time.replace(tzinfo=None),
                 l.arrival_time.replace(tzinfo=None) if l.arrival_time else None) for l in req.legs]
        for i in range(n - 1):
            if legs[i][1] != legs[i + 1][0]:
                raise _err(422, "not_connected",
                           f"Flight {i + 2} must leave from {legs[i][1]}, where flight {i + 1} lands.", index=i)

        arrivals = [a for (_, _, _, a) in legs]
        for i in range(n - 1):               # only legs that feed a connection need an arrival time
            if arrivals[i] is None:
                import flight_lookup
                sched = await run_in_threadpool(flight_lookup.get_schedule)
                arrivals[i] = await run_in_threadpool(_resolve_arrival, req.legs[i], sched)
                if arrivals[i] is None:
                    raise _err(422, "schedule_unknown",
                               f"We don't know when flight {i + 1} lands. Add its arrival time.", index=i)
        layovers = []
        for i in range(n - 1):
            layover = round((legs[i + 1][2] - arrivals[i]).total_seconds() / 60)
            if layover <= 0 or layover > MAX_LAYOVER_MIN:
                raise _err(422, "invalid_layover",
                           "The connection must be more than 0 and at most 24 hours.", index=i)
            layovers.append(layover)

        itinerary_id = req.itinerary_id or f"itin-{uuid.uuid4().hex[:12]}"
        responses, forecasts = [], []
        for i, l in enumerate(req.legs):
            request = m.FlightRequest(flight_id=l.flight_id, origin=l.origin, destination=l.destination,
                                      departure_time=legs[i][2], airline=l.airline, flight_number=l.flight_number,
                                      arrival_time=arrivals[i])
            resp, fc = await m._predict_core(request, background_tasks, need_forecast=True)
            responses.append(resp)
            forecasts.append(fc)

        connections, misses, rows = [], [], []
        now = datetime.now(timezone.utc)
        model_version = f"{ITIN_VERSION} ({svc.version}, dep-copula-{svc.dependence.variant})"
        for i in range(n - 1):
            a, b = responses[i], responses[i + 1]
            hub, layover = legs[i][1], layovers[i]
            net_a, net_b = network(a.airline), network(b.airline)
            known = net_a is not None and net_b is not None
            cross = known and net_a != net_b
            minutes, source = mct_mod.resolve_mct(hub, mct_table(), conns_in[i].min_connect_minutes,
                                                  cross_network=cross)
            op_a = (a.operating_airline or a.airline, a.operating_flight_number or a.flight_number)
            op_b = (b.operating_airline or b.airline, b.operating_flight_number or b.flight_number)
            through = None not in op_a and None not in op_b and op_a == op_b

            warnings, reason = [], None
            if through:
                kind, reason = "through_flight", "through_flight"
                warnings.append("Same flight number continues. Usually the same plane.")
            else:
                kind = "connection"
                if not known:
                    reason = "airline_unknown"
                elif cross:
                    reason = "different_airlines"
                    warnings.append(CROSS_AIRLINE_WARNING)
                elif hub not in mct_mod.TOP30:
                    reason = "airport_not_tested"
                    warnings.append(f"We haven't tested connections at {hub}, so we don't show a number.")
                elif layover > VALIDATED_MAX_LAYOVER_MIN:
                    reason = "layover_over_4h"
                    warnings.append("Layovers over 4 hours are longer than our connection model covers, "
                                    "so we don't show a number.")
                elif layover < minutes:
                    reason = "layover_below_minimum"
                if layover < minutes:
                    warnings.append(f"Layover is shorter than the usual minimum at {hub}")

            validated = reason is None
            miss = None
            if validated:
                miss = round(svc.calibrate_miss(miss_probability(forecasts[i], forecasts[i + 1], layover, minutes,
                                                                 svc.dependence, hub)), 4)
            misses.append(0.0 if through else miss)
            connections.append({
                "at": hub, "kind": kind, "scheduled_layover_minutes": layover,
                "min_connect_minutes": minutes, "mct_source": source,
                "miss_probability": miss, "risk_level": risk_band(miss) if miss is not None else None,
                "validated": validated, "not_validated_reason": reason, "warnings": warnings,
            })
            rows.append({"itinerary_id": itinerary_id, "ts_utc": now, "connection_index": i,
                         "leg_flight_ids": [a.flight_id, b.flight_id], "at": hub, "kind": kind,
                         "layover_min": layover, "mct_min": minutes, "mct_source": source,
                         "miss_probability": miss, "validated": validated, "not_validated_reason": reason,
                         "model_version": model_version})

        trip = trip_summary(forecasts, misses)
        if trip["disruption_probability"] is not None:
            trip["disruption_probability"] = round(trip["disruption_probability"], 4)
        if rows:
            background_tasks.add_task(m.prediction_store.safe_write_itinerary, rows)
        return {
            "itinerary_id": itinerary_id,
            "legs": [r.model_dump() for r in responses],
            "connections": connections,
            "trip": trip,
            "warnings": [],
            "model_version": model_version,
        }

    app.add_api_route("/predict/itinerary", predict_itinerary, methods=["POST"], tags=["itinerary"])
