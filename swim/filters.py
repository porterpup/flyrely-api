"""Receive-time filtering for SWIM messages (stdlib only; regex on the raw XML, no full parse).

What is kept, per source (all configurable, see deploy/filter.example.json):
  TFMS   airline flights only. Keep FlightTimes, FlightCreate/Modify/ScheduleActivate, flight plans,
         amendments, cancellations, departure/arrival information; ALL flow data and Status.
         Track (trackInformation) updates are thinned to 1 per flight per `track_interval_min`
         (0 = drop them). Other flight message types are dropped.
  SFDPS  airline flights only. Keep flight messages that carry aircraftDescription (flight plan data),
         the FIRST message that shows a given actual departure/arrival runway time for a flight,
         and Status. Position/track messages are thinned like TFMS tracks. Batch messages
         (many flights in one message) are split per flight and re-assembled from the kept flights.
  STDDS  pass-through (only the block list applies) until its messages have been profiled.

LADD hook: if the block-list file exists, flights on it are dropped first, whatever else applies
(see deploy/LADD_BLOCKLIST.md). Anything unparseable is KEPT (fail open) and counted.
"""
import json
import logging
import os
import re
import time
from collections import Counter
from itertools import islice
from pathlib import Path

log = logging.getLogger("swim")

# Passenger carriers that report to BTS plus their regional partners (ICAO 3-letter codes).
# Extend with `extra_carriers` in the config. Cargo, charter and foreign carriers are not included.
DEFAULT_CARRIERS = (
    "AAL UAL DAL SWA JBU ASA NKS FFT HAL AAY SCX VXP MXY "           # mainline / low-cost
    "SKW RPA ENY EDV PDT PSA JIA CPZ GJS ASH QXE AWI UCA MKU KAP"    # regional partners
).split()

TFMS_KEEP_TYPES = [
    "FlightTimes", "FlightCreate", "FlightModify", "FlightScheduleActivate",
    "flightPlanInformation", "flightPlanAmendmentInformation", "flightPlanCancellation",
    "departureInformation", "arrivalInformation",
]
TFMS_TRACK_TYPES = ["trackInformation"]

DEFAULTS = {
    "tfms": {"enabled": True, "airlines_only": True, "airline_mode": "carriers", "extra_carriers": [],
             "keep_types": TFMS_KEEP_TYPES, "track_types": TFMS_TRACK_TYPES, "track_interval_min": 5,
             "keep_flow": True, "keep_status": True},
    "sfdps": {"enabled": True, "airlines_only": True, "airline_mode": "carriers", "extra_carriers": [],
              "track_interval_min": 5, "keep_message_types": []},
    "stdds": {"enabled": False},
}

CALLSIGN_RE = re.compile(r"^([A-Z]{3})(\d{1,4})([A-Z]{0,2})$")
ATTR_RE = re.compile(r'(\w+)="([^"]*)"')
TFMS_HEAD_RE = re.compile(r"<(?:\w+:)?fltdMessage\b([^>]*)>")
SF_MESSAGE_RE = re.compile(r"<message\b.*?</message>", re.S)
SF_IDENT_RE = re.compile(r'aircraftIdentification="([^"]*)"')
SF_REG_RE = re.compile(r'\bregistration="([^"]*)"')
SF_ADDR_RE = re.compile(r'\baircraftAddress="([^"]*)"')
SF_GUFI_RE = re.compile(r"<gufi\b[^>]*>([^<]+)<")
SF_PLANID_RE = re.compile(r'<flightPlan\b[^>]*\bidentifier="([^"]*)"')
SF_DEP_RE = re.compile(r"<departure\b.*?</departure>", re.S)
SF_ARR_RE = re.compile(r"<arrival\b.*?</arrival>", re.S)
SF_ACTUAL_RE = re.compile(r'<actual\s+time="([^"]+)"')
MAX_STATE = 300_000


# ------------------------------------------------------------------ helpers
def is_airline(ident: str, mode: str, carriers: set) -> bool:
    m = CALLSIGN_RE.match((ident or "").upper())
    if not m:
        return False
    return True if mode == "pattern" else m.group(1) in carriers


def _trim(d: dict):
    if len(d) > MAX_STATE:
        for k in list(islice(d, len(d) // 2)):
            del d[k]


class Blocklist:
    """LADD hook. One identifier per line or comma/space separated: callsign, registration (N-number)
    or 24-bit hex address; '#' starts a comment. Missing file = empty list = no-op. The file is
    re-read (cheaply) when its mtime changes, checked at most every `check_s` seconds."""

    def __init__(self, path=None, check_s: float = 60.0):
        self.path = Path(path) if path else None
        self.check_s, self._next, self._mtime = check_s, 0.0, None
        self.ids: frozenset = frozenset()
        self.reload(force=True)

    def reload(self, force=False):
        now = time.monotonic()
        if not force and now < self._next:
            return
        self._next = now + self.check_s
        if not self.path:
            return
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            if self._mtime is not None:
                log.info("block list removed; no LADD filtering")
            self._mtime, self.ids = None, frozenset()
            return
        if mtime == self._mtime:
            return
        try:
            ids = set()
            for line in self.path.read_text(errors="replace").splitlines():
                ids.update(t.upper() for t in re.split(r"[\s,;]+", line.split("#", 1)[0]) if t)
            self.ids, self._mtime = frozenset(ids), mtime
            log.info("block list loaded: %d identifiers", len(ids))
        except OSError:
            log.warning("block list unreadable; keeping the previous one")

    def hit(self, *idents) -> bool:
        self.reload()
        return bool(self.ids) and any(i and i.upper() in self.ids for i in idents)


def load_config(path=None, env=None) -> dict:
    """DEFAULTS, overlaid by the JSON file (path or $SWIM_FILTER_CONFIG), then env overrides:
    SWIM_FILTER=off disables filtering; SWIM_TRACK_INTERVAL_MIN overrides the track interval."""
    env = os.environ if env is None else env
    cfg = json.loads(json.dumps(DEFAULTS))
    p = path or env.get("SWIM_FILTER_CONFIG")
    if p:
        user = json.loads(Path(p).read_text())
        for src, vals in user.items():
            if src not in cfg:
                raise ValueError(f"filter config: unknown source {src!r}")
            cfg[src].update(vals)
    if env.get("SWIM_FILTER", "").lower() in ("off", "0", "false", "no"):
        for c in cfg.values():
            c["enabled"] = False
    if env.get("SWIM_TRACK_INTERVAL_MIN"):
        for src in ("tfms", "sfdps"):
            cfg[src]["track_interval_min"] = float(env["SWIM_TRACK_INTERVAL_MIN"])
    return cfg


# ------------------------------------------------------------------- filter
class MessageFilter:
    """apply(props, body, now_epoch) -> (body_or_None, reason). None = drop the message."""

    def __init__(self, source: str, cfg: dict = None, blocklist: Blocklist = None):
        self.source = source
        self.cfg = {**DEFAULTS[source], **(cfg or {})}
        self.blocklist = blocklist or Blocklist(None)
        self.carriers = set(DEFAULT_CARRIERS) | {c.upper() for c in self.cfg.get("extra_carriers", [])}
        self.counts = Counter()
        self._last_track: dict = {}
        self._actuals: dict = {}

    # -- common
    def _is_airline(self, ident):
        return (not self.cfg.get("airlines_only", True)) or is_airline(
            ident, self.cfg.get("airline_mode", "carriers"), self.carriers)

    def _track_ok(self, key, now) -> bool:
        gap = float(self.cfg.get("track_interval_min", 5)) * 60
        if gap <= 0 or not key:
            return False
        last = self._last_track.get(key)
        if last is not None and now - last < gap:
            return False
        self._last_track[key] = now
        _trim(self._last_track)
        return True

    def apply(self, props: dict, body: str, now: float):
        try:
            if self.source == "tfms":
                out, reason = self._tfms(props, body, now)
            elif self.source == "sfdps":
                out, reason = self._sfdps(props, body, now)
            else:
                out, reason = self._passthrough(body)
        except Exception as e:  # never lose data because of a filter bug
            out, reason = body, f"keep:filter_error:{type(e).__name__}"
        self.counts[reason] += 1
        return out, reason

    def _passthrough(self, body):
        return body, "keep:passthrough"

    # -- TFMS
    def _tfms(self, props, body, now):
        c = self.cfg
        cls = str(props.get("TFMDataClass", ""))
        if not c.get("enabled", True):
            return body, "keep:filter_off"
        if cls == "FlowInformation":
            return (body, "keep:flow") if c["keep_flow"] else (None, "drop:flow")
        if cls == "Status":
            return (body, "keep:status") if c["keep_status"] else (None, "drop:status")
        if cls != "FlightData":
            return body, "keep:other_class"
        m = TFMS_HEAD_RE.search(body[:4000])
        if not m:
            return body, "keep:unparsed"
        a = dict(ATTR_RE.findall(m.group(1)))
        acid, mtype = a.get("acid", ""), a.get("msgType", "")
        if self.blocklist.hit(acid):
            return None, "drop:ladd"
        if not self._is_airline(acid):
            return None, "drop:not_airline"
        if mtype in c["keep_types"]:
            return body, "keep:" + mtype
        if mtype in c["track_types"]:
            if self._track_ok(a.get("flightRef") or acid, now):
                return body, "keep:track_sample"
            return None, "drop:track_thinned"
        return None, "drop:type:" + (mtype or "unknown")

    # -- SFDPS
    def _sfdps(self, props, body, now):
        c = self.cfg
        mtype = str(props.get("FDPS_MessageType", ""))
        if not c.get("enabled", True):
            return body, "keep:filter_off"
        if str(props.get("FDPS_DataType", "")).upper() == "STATUS" or mtype.lower() == "status":
            return body, "keep:status"
        if mtype in c["keep_message_types"]:
            return body, "keep:configured_type"
        blocks = list(SF_MESSAGE_RE.finditer(body))
        if not blocks:
            if "<flight " in body:
                return body, "keep:unparsed"
            return body, "keep:nonflight"
        kept, why = [], Counter()
        for b in blocks:
            r = self._sf_block(b.group(0), now)
            why[r] += 1
            if r.startswith("keep"):
                kept.append(b.group(0))
        best = next((r for r in why if r.startswith("keep")), None)
        if not kept:
            return None, "drop:" + next(iter(why.most_common(1)))[0].split(":", 1)[1]
        if len(kept) == len(blocks):
            return body, best
        out = body[:blocks[0].start()] + "\n    ".join(kept) + body[blocks[-1].end():]
        return out, best

    def _sf_block(self, blk, now):
        m = SF_IDENT_RE.search(blk)
        ident = m.group(1) if m else ""
        reg, addr = SF_REG_RE.search(blk), SF_ADDR_RE.search(blk)
        if self.blocklist.hit(ident, reg and reg.group(1), addr and addr.group(1)):
            return "drop:ladd"
        if not self._is_airline(ident):
            return "drop:not_airline"
        if "<aircraftDescription" in blk:
            return "keep:flight_plan"
        g = SF_GUFI_RE.search(blk) or SF_PLANID_RE.search(blk)
        key = g.group(1) if g else ident
        if self._new_actual(key, blk):
            return "keep:runway_time"
        if "<position" in blk:
            return "keep:track_sample" if self._track_ok(key, now) else "drop:track_thinned"
        return "drop:other"

    def _new_actual(self, key, blk) -> bool:
        dep = SF_DEP_RE.search(blk)
        arr = SF_ARR_RE.search(blk)
        d = SF_ACTUAL_RE.search(dep.group(0)) if dep else None
        r = SF_ACTUAL_RE.search(arr.group(0)) if arr else None
        cur = (d and d.group(1), r and r.group(1))
        if cur == (None, None) or not key:
            return False
        old = self._actuals.get(key, (None, None))
        new = (cur[0] and cur[0] != old[0]) or (cur[1] and cur[1] != old[1])
        if new:
            self._actuals[key] = (cur[0] or old[0], cur[1] or old[1])
            _trim(self._actuals)
        return bool(new)
