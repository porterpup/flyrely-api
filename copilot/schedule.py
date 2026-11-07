"""
Where the copilot gets "other flights on this route" from.

`ScheduleSource.flights_on(origin, dest, date)` returns the flights usually scheduled that weekday.
`BtsTypicalSchedule` reads models/route_schedule.csv.gz (built by scripts/build_route_schedule.py from the last
8 weeks of BTS data). The file is loaded on the first call, not at import, so API start-up is unchanged when the
copilot flag is off. It reloads when the file changes (scripts/refresh_schedule.py swaps in a new one monthly). `SwimScheduleSource` is a stub for the day's filed flight plans (roadmap R3).
"""

import csv
import gzip
import logging
import threading
from dataclasses import dataclass
from datetime import date as date_cls, datetime
from pathlib import Path
from typing import Optional, Protocol, Union

log = logging.getLogger("copilot.schedule")
SCHEDULE_PATH = Path(__file__).resolve().parent.parent / "models" / "route_schedule.csv.gz"


@dataclass(frozen=True)
class ScheduledFlight:
    marketing_airline: str
    flight_number: str            # marketing airline + number, e.g. "AA117"
    operating_airline: str
    operating_flight_number: str  # e.g. "OO5521"
    origin: str
    dest: str
    dep_time: str                 # local "HH:MM"
    arr_time: str                 # local "HH:MM", may be the next calendar day
    days_seen: int


class ScheduleSource(Protocol):
    source: str

    def flights_on(self, origin: str, dest: str, date: Union[date_cls, datetime, str]) -> list[ScheduledFlight]: ...

    def schedule_as_of(self) -> Optional[str]: ...


def _to_date(d) -> date_cls:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date_cls):
        return d
    return date_cls.fromisoformat(str(d)[:10])


class BtsTypicalSchedule:
    source = "bts_typical_schedule"

    def __init__(self, path: Path = SCHEDULE_PATH):
        self.path = Path(path)
        self._by_route: Optional[dict[tuple[str, str], list[tuple[int, ScheduledFlight]]]] = None
        self._as_of: Optional[str] = None
        self._by_number: Optional[tuple] = None   # (the _by_route table it was built from, index)
        self._sig = None
        self._lock = threading.Lock()

    def _signature(self):
        try:
            st = self.path.stat()
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size, st.st_ino)

    def _load(self):
        """Load on first use, and again whenever the file changes (mtime, size or inode), so a monthly refresh
        that swaps in a new file with os.replace is picked up without restarting the API. A file that goes missing
        or fails to parse after a good load keeps the last good table."""
        sig = self._signature()
        if self._by_route is not None and (sig == self._sig or sig is None):
            return
        with self._lock:
            sig = self._signature()      # taken before reading: a swap during the read triggers another reload
            if self._by_route is not None and (sig == self._sig or sig is None):
                return
            by_route: dict = {}
            as_of = None
            try:
                if sig is not None:
                    with gzip.open(self.path, "rt", newline="") as f:
                        for r in csv.DictReader(f):
                            flight = ScheduledFlight(
                                marketing_airline=r["marketing_airline"],
                                flight_number=r["marketing_airline"] + r["marketing_flight_number"],
                                operating_airline=r["operating_airline"],
                                operating_flight_number=r["operating_airline"] + r["operating_flight_number"],
                                origin=r["origin"], dest=r["dest"], dep_time=r["dep_time"], arr_time=r["arr_time"],
                                days_seen=int(r["days_seen"]))
                            by_route.setdefault((r["origin"], r["dest"]), []).append((int(r["dow_mask"]), flight))
                            as_of = max(as_of or "", r["schedule_as_of"])
            except (OSError, EOFError, ValueError, KeyError) as e:
                if self._by_route is None:
                    raise
                log.warning("schedule file unreadable (%s); keeping the previous table", type(e).__name__)
                self._sig = sig
                return
            for rows in by_route.values():
                rows.sort(key=lambda x: x[1].dep_time)
            self._as_of = as_of
            self._sig = sig
            self._by_route = by_route          # assigned last: readers see a complete table

    def flights_on(self, origin: str, dest: str, date) -> list[ScheduledFlight]:
        """Flights usually scheduled on this date's weekday, earliest departure first. [] for an unknown route."""
        self._load()
        bit = 1 << _to_date(date).weekday()     # Monday = bit 0, as in the CSV
        return [f for mask, f in self._by_route.get((origin.upper(), dest.upper()), []) if mask & bit]

    def flights_by_number(self, airline: str, number: str, date) -> list[ScheduledFlight]:
        """Every leg of marketing flight `airline``number` usually scheduled on this date's weekday (earliest first).

        One number can be several legs (through-flights) or several daily frequencies; all are returned.
        """
        self._load()
        by_route = self._by_route
        cached = self._by_number
        if cached is None or cached[0] is not by_route:      # rebuilt when the schedule file is reloaded
            idx: dict = {}
            for rows in by_route.values():
                for mask, f in rows:
                    idx.setdefault((f.marketing_airline, f.flight_number[len(f.marketing_airline):]), []).append((mask, f))
            cached = self._by_number = (by_route, idx)
        bit = 1 << _to_date(date).weekday()
        hits = [f for mask, f in cached[1].get((airline.upper(), str(number).lstrip("0")), []) if mask & bit]
        return sorted(hits, key=lambda f: f.dep_time)

    def schedule_as_of(self) -> Optional[str]:
        """Latest BTS month in the artifact (YYYY-MM), or None if the file is missing."""
        self._load()
        return self._as_of


class SwimScheduleSource:
    """Stub: will read the day's filed flight plans from the SWIM live-state store (roadmap R3)."""
    source = "swim_flight_plans"

    def flights_on(self, origin: str, dest: str, date) -> list[ScheduledFlight]:
        raise NotImplementedError("SWIM schedule source is not built yet (roadmap R3)")

    def schedule_as_of(self) -> Optional[str]:
        raise NotImplementedError("SWIM schedule source is not built yet (roadmap R3)")
