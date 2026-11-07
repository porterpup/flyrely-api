"""SQLite (WAL) live store for the L1 linkage. Local only (data/linkage/live.sqlite, gitignored).

Every fact carries `first_recv`, the archived receive time of the message that first showed it, so any later
reader can restrict itself to facts with first_recv <= as_of. Writers use MIN for first_recv, so re-processing
a file (restart, back-fill) can never move a fact later. Registrations and icao24 values are stored here and
nowhere else; nothing in this module logs them.
"""
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from linkage.carriers import is_us_normalised

SCHEMA = """
CREATE TABLE IF NOT EXISTS legs (
  leg_id TEXT PRIMARY KEY, callsign TEXT, carrier_icao TEXT, flight_no TEXT, origin TEXT, dest TEXT,
  sched_out_utc TEXT, sched_in_utc TEXT, aircraft_type TEXT, in_schedule INTEGER DEFAULT 0, flight_type TEXT,
  sfdps_gufi TEXT, eram_gufi TEXT, tfms_flight_ref TEXT,
  out_utc TEXT, out_recv TEXT, off_utc TEXT, off_recv TEXT, on_utc TEXT, on_recv TEXT, in_utc TEXT, in_recv TEXT,
  est_off_utc TEXT, est_on_utc TEXT, est_recv TEXT,
  cancelled INTEGER DEFAULT 0, cancelled_recv TEXT, cancelled_gufi TEXT,
  first_seen_recv TEXT, last_recv TEXT);
CREATE INDEX IF NOT EXISTS legs_triple ON legs(callsign, origin, dest);
CREATE TABLE IF NOT EXISTS leg_aircraft (
  leg_id TEXT, source TEXT, reg TEXT, icao24 TEXT, first_recv TEXT, last_recv TEXT,
  PRIMARY KEY (leg_id, source, reg, icao24));
CREATE INDEX IF NOT EXISTS leg_aircraft_reg ON leg_aircraft(reg);
CREATE TABLE IF NOT EXISTS surface_events (
  airport TEXT, callsign TEXT, icao24 TEXT, event TEXT, event_time TEXT, recv TEXT, sfdps_gufi TEXT, leg_id TEXT,
  PRIMARY KEY (airport, callsign, event, event_time));
CREATE TABLE IF NOT EXISTS sfdps_gufis (gufi TEXT PRIMARY KEY, leg_id TEXT, first_recv TEXT);
CREATE TABLE IF NOT EXISTS blocked_legs (leg_id TEXT PRIMARY KEY, reason TEXT);
CREATE TABLE IF NOT EXISTS ingest_progress (path TEXT PRIMARY KEY, lines_done INTEGER);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
-- created empty here; filled by the engine / snapshotter (task T5)
CREATE TABLE IF NOT EXISTS link_snapshots (
  leg_id TEXT, horizon TEXT, as_of TEXT, method TEXT, confidence REAL, tail_key TEXT, inbound_leg_id TEXT,
  leg_at_c_id TEXT, chain_json TEXT, engine_version TEXT, store_lag_s REAL);
CREATE TABLE IF NOT EXISTS pairing_prior (
  carrier TEXT, flight_no TEXT, origin TEXT, weekday INTEGER, inbound_flight_no TEXT, n_days INTEGER, share REAL,
  source TEXT, built_on TEXT);
"""

ACTUALS = ("out", "off", "on", "in")
SCHED_FIELDS = ("sched_out_utc", "sched_in_utc", "aircraft_type", "flight_type", "sfdps_gufi", "eram_gufi",
                "tfms_flight_ref")


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def iso_ms(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def norm_recv(s: str) -> str:
    """Archived recv ('2026-11-07T14:00:00.235537+00:00') -> fixed-width UTC string that sorts as text."""
    return iso_ms(parse_ts(s))


def norm_time(s: str) -> str:
    """Message time ('2026-11-07T13:58:00Z', optional fraction) -> 'YYYY-MM-DDTHH:MM:SSZ'."""
    return parse_ts(s).strftime("%Y-%m-%dT%H:%M:%SZ")


class Store:
    ANCHOR_TOL = timedelta(hours=12)
    DEFAULT_LEAD = timedelta(minutes=90)

    def __init__(self, path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, isolation_level=None, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)
        self._gufi_cache: dict = {}

    # ------------------------------------------------------------ transactions
    def begin(self):
        self.db.execute("BEGIN")

    def commit(self):
        if self.db.in_transaction:
            self.db.execute("COMMIT")

    def close(self):
        self.commit()
        self.db.close()

    # ------------------------------------------------------------ legs
    def get_leg(self, leg_id):
        return self.db.execute("SELECT * FROM legs WHERE leg_id=?", (leg_id,)).fetchone()

    def is_blocked(self, leg_id) -> bool:
        return bool(leg_id) and self.db.execute(
            "SELECT 1 FROM blocked_legs WHERE leg_id=?", (leg_id,)).fetchone() is not None

    def upsert_leg(self, leg_id, recv, *, callsign, carrier, flight_no, origin, dest, in_schedule=False,
                   sched=None, actuals=None, est=None, cancelled=None):
        """Create or update a leg. `recv` is the normalised archived receive time.
        sched: SCHED_FIELDS values (first value wins; they describe the original schedule / identity).
        actuals: {'out'|'off'|'on'|'in': normalised time} (first value wins, its recv is the first one seen).
        est: {'off'|'on': time} latest wins. cancelled: (True, gufi) or (False, None) to clear."""
        r = self.get_leg(leg_id)
        if r is None:
            self.db.execute(
                "INSERT INTO legs(leg_id, callsign, carrier_icao, flight_no, origin, dest, in_schedule, "
                "first_seen_recv, last_recv) VALUES (?,?,?,?,?,?,?,?,?)",
                (leg_id, callsign, carrier, flight_no, origin, dest, 1 if in_schedule else 0, recv, recv))
            r = self.get_leg(leg_id)
        sets, args = {}, []
        if recv < r["first_seen_recv"]:
            sets["first_seen_recv"] = recv
        if recv > r["last_recv"]:
            sets["last_recv"] = recv
        if in_schedule and not r["in_schedule"]:
            sets["in_schedule"] = 1
        for k, v in (sched or {}).items():
            if v and not r[k]:
                sets[k] = v
        for a, v in (actuals or {}).items():
            if not v:
                continue
            cur, cur_recv = r[f"{a}_utc"], r[f"{a}_recv"]
            if cur is None:
                sets[f"{a}_utc"], sets[f"{a}_recv"] = v, recv
            elif cur == v and recv < cur_recv:
                sets[f"{a}_recv"] = recv
        for a, v in (est or {}).items():
            if v:
                sets[f"est_{a}_utc"] = v
                sets["est_recv"] = recv
        if cancelled is not None:
            flag, gufi = cancelled
            if flag and (not r["cancelled"] or (r["cancelled_recv"] or "") > recv):
                sets["cancelled"], sets["cancelled_recv"], sets["cancelled_gufi"] = 1, recv, gufi or ""
            elif not flag and r["cancelled"] and recv > (r["cancelled_recv"] or ""):
                sets["cancelled"] = 0
        if sets:
            cols = ", ".join(f"{k}=?" for k in sets)
            self.db.execute(f"UPDATE legs SET {cols} WHERE leg_id=?", (*sets.values(), leg_id))
        return r

    def find_leg(self, callsign, origin, dest, target: datetime):
        """Nearest existing leg with this (callsign, origin, dest) whose anchor (scheduled out, else first
        seen + 90 min) lies within 12 h of `target`; None when there is none."""
        best, best_d = None, None
        for r in self.db.execute("SELECT leg_id, sched_out_utc, first_seen_recv FROM legs "
                                 "WHERE callsign=? AND origin=? AND dest=?", (callsign, origin, dest)):
            anchor = parse_ts(r["sched_out_utc"]) if r["sched_out_utc"] else parse_ts(r["first_seen_recv"]) + self.DEFAULT_LEAD
            d = abs(anchor - target)
            if d <= self.ANCHOR_TOL and (best_d is None or d < best_d):
                best, best_d = r["leg_id"], d
        return best

    # ------------------------------------------------------------ aircraft identity
    def add_aircraft(self, leg_id, source, reg, icao24, recv):
        reg, icao24 = (reg or "").upper(), (icao24 or "").upper()
        if not (reg or icao24):
            return
        self.db.execute(
            "INSERT INTO leg_aircraft(leg_id, source, reg, icao24, first_recv, last_recv) VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(leg_id, source, reg, icao24) DO UPDATE SET "
            "first_recv=MIN(first_recv, excluded.first_recv), last_recv=MAX(last_recv, excluded.last_recv)",
            (leg_id, source, reg, icao24, recv, recv))

    def add_surface_event(self, airport, callsign, icao24, event, event_time, recv, gufi, leg_id):
        self.db.execute(
            "INSERT INTO surface_events(airport, callsign, icao24, event, event_time, recv, sfdps_gufi, leg_id) "
            "VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(airport, callsign, event, event_time) DO UPDATE SET "
            "recv=MIN(recv, excluded.recv)",
            (airport, callsign, (icao24 or "").upper(), event, event_time, recv, gufi or "", leg_id))

    # ------------------------------------------------------------ GUFI map
    def gufi_leg(self, gufi):
        if not gufi:
            return None
        if gufi in self._gufi_cache:
            return self._gufi_cache[gufi]
        r = self.db.execute("SELECT leg_id FROM sfdps_gufis WHERE gufi=?", (gufi,)).fetchone()
        if r:
            self._gufi_cache[gufi] = r[0]
            if len(self._gufi_cache) > 200_000:
                self._gufi_cache.clear()
        return r[0] if r else None

    def map_gufi(self, gufi, leg_id, recv):
        if not gufi:
            return
        self.db.execute("INSERT INTO sfdps_gufis(gufi, leg_id, first_recv) VALUES (?,?,?) "
                        "ON CONFLICT(gufi) DO UPDATE SET leg_id=excluded.leg_id, "
                        "first_recv=MIN(first_recv, excluded.first_recv)", (gufi, leg_id, recv))
        self._gufi_cache[gufi] = leg_id

    # ------------------------------------------------------------ merging / privacy
    def merge_legs(self, src, dst):
        """Fold leg `src` (an SFDPS-only orphan) into `dst` (the scheduled leg) and delete `src`."""
        s, d = self.get_leg(src), self.get_leg(dst)
        if not s or not d or src == dst:
            return
        for r in self.db.execute("SELECT * FROM leg_aircraft WHERE leg_id=?", (src,)).fetchall():
            self.add_aircraft(dst, r["source"], r["reg"], r["icao24"], r["first_recv"])
            self.add_aircraft(dst, r["source"], r["reg"], r["icao24"], r["last_recv"])
        self.db.execute("DELETE FROM leg_aircraft WHERE leg_id=?", (src,))
        self.db.execute("UPDATE surface_events SET leg_id=? WHERE leg_id=?", (dst, src))
        self.db.execute("UPDATE sfdps_gufis SET leg_id=? WHERE leg_id=?", (dst, src))
        self._gufi_cache.clear()
        sets = {}
        for k in ("flight_type", "sfdps_gufi", "eram_gufi", "aircraft_type"):
            if s[k] and not d[k]:
                sets[k] = s[k]
        for a in ACTUALS:
            if s[f"{a}_utc"] and not d[f"{a}_utc"]:
                sets[f"{a}_utc"], sets[f"{a}_recv"] = s[f"{a}_utc"], s[f"{a}_recv"]
        if s["est_recv"] and (not d["est_recv"] or s["est_recv"] > d["est_recv"]):
            sets.update(est_off_utc=s["est_off_utc"], est_on_utc=s["est_on_utc"], est_recv=s["est_recv"])
        if s["cancelled"] and not d["cancelled"]:
            sets.update(cancelled=1, cancelled_recv=s["cancelled_recv"], cancelled_gufi=s["cancelled_gufi"])
        if s["first_seen_recv"] < d["first_seen_recv"]:
            sets["first_seen_recv"] = s["first_seen_recv"]
        if sets:
            self.db.execute(f"UPDATE legs SET {', '.join(k + '=?' for k in sets)} WHERE leg_id=?",
                            (*sets.values(), dst))
        self.db.execute("DELETE FROM legs WHERE leg_id=?", (src,))

    def adopt_orphans(self, leg_id):
        """After a scheduled leg appears, absorb SFDPS-only legs for the same flight."""
        d = self.get_leg(leg_id)
        if not d or not d["sched_out_utc"]:
            return 0
        target, n = parse_ts(d["sched_out_utc"]), 0
        for r in self.db.execute("SELECT leg_id, first_seen_recv FROM legs WHERE callsign=? AND origin=? AND dest=? "
                                 "AND sched_out_utc IS NULL", (d["callsign"], d["origin"], d["dest"])).fetchall():
            if abs(parse_ts(r["first_seen_recv"]) + self.DEFAULT_LEAD - target) <= self.ANCHOR_TOL:
                self.merge_legs(r["leg_id"], leg_id)
                n += 1
        return n

    def block_leg(self, leg_id, reason):
        """Privacy drop: delete the leg and every fact about it, and remember it so later messages
        for the same leg (TFMS, STDDS) are dropped too."""
        if not leg_id:
            return
        self.db.execute("INSERT OR REPLACE INTO blocked_legs(leg_id, reason) VALUES (?,?)", (leg_id, reason))
        for (g,) in self.db.execute("SELECT gufi FROM sfdps_gufis WHERE leg_id=?", (leg_id,)).fetchall():
            self.db.execute("INSERT OR REPLACE INTO blocked_legs(leg_id, reason) VALUES (?,?)", ("gufi:" + g, "gufi"))
        self.delete_leg(leg_id)

    def delete_leg(self, leg_id):
        self.db.execute("DELETE FROM leg_aircraft WHERE leg_id=?", (leg_id,))
        self.db.execute("DELETE FROM surface_events WHERE leg_id=?", (leg_id,))
        self.db.execute("DELETE FROM sfdps_gufis WHERE leg_id=?", (leg_id,))
        self.db.execute("DELETE FROM link_snapshots WHERE leg_id=?", (leg_id,))
        self.db.execute("DELETE FROM legs WHERE leg_id=?", (leg_id,))
        self._gufi_cache.clear()

    def scrub(self, ids) -> int:
        """Delete every leg / surface event that matches a block-list identifier (callsign, reg, icao24).
        Run whenever the list changes. Returns the number of legs removed."""
        ids = sorted({i.upper() for i in ids if i})
        if not ids:
            return 0
        n = 0
        for i in range(0, len(ids), 500):
            ph = ",".join("?" * len(ids[i:i + 500]))
            part = ids[i:i + 500]
            legs = {r[0] for r in self.db.execute(f"SELECT leg_id FROM legs WHERE callsign IN ({ph})", part)}
            legs |= {r[0] for r in self.db.execute(
                f"SELECT leg_id FROM leg_aircraft WHERE reg IN ({ph}) OR icao24 IN ({ph})", part + part)}
            legs |= {r[0] for r in self.db.execute(
                f"SELECT leg_id FROM surface_events WHERE (callsign IN ({ph}) OR icao24 IN ({ph})) "
                "AND leg_id IS NOT NULL", part + part)}
            for lg in legs:
                self.block_leg(lg, "ladd_scrub")
            n += len(legs)
            self.db.execute(f"DELETE FROM surface_events WHERE callsign IN ({ph}) OR icao24 IN ({ph})", part + part)
        return n

    # ------------------------------------------------------------ queries
    def chain_legs(self, carriers):
        """The chain universe (BTS parity): a leg with a scheduled out time, domestic at both ends, a reporting
        carrier or its regional partner, and scheduled service. "Scheduled" = the TFMS airline schedule showed it
        (FlightCreate / ScheduleActivate / OAG-sourced modify) or the FAA flight plan is flightType SCHEDULED
        (which covers flights whose schedule message predates the archive). A plan flagged NON_SCHEDULED /
        GENERAL / OTHER (ferry, charter, positioning) is out whatever else is true."""
        out = []
        for r in self.db.execute("SELECT * FROM legs WHERE sched_out_utc IS NOT NULL "
                                 "AND (in_schedule=1 OR flight_type='SCHEDULED') "
                                 "AND (flight_type IS NULL OR flight_type='SCHEDULED')"):
            if r["carrier_icao"] in carriers and is_us_normalised(r["origin"]) and is_us_normalised(r["dest"]):
                out.append(r)
        return out

    def counts(self) -> dict:
        q = lambda s: self.db.execute(s).fetchone()[0]
        return {
            "legs": q("SELECT COUNT(*) FROM legs"),
            "legs_in_schedule": q("SELECT COUNT(*) FROM legs WHERE in_schedule=1"),
            "legs_with_reg": q("SELECT COUNT(DISTINCT leg_id) FROM leg_aircraft WHERE reg<>''"),
            "legs_with_icao24": q("SELECT COUNT(DISTINCT leg_id) FROM leg_aircraft WHERE icao24<>''"),
            "surface_events": q("SELECT COUNT(*) FROM surface_events"),
            "blocked_legs": q("SELECT COUNT(*) FROM blocked_legs"),
        }

    def set_meta(self, k, v):
        self.db.execute("INSERT INTO meta(k, v) VALUES (?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, str(v)))

    def get_meta(self, k, default=None):
        r = self.db.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
        return r[0] if r else default
