"""Receive-time filtering for SWIM messages (stdlib only; regex on the raw XML, no full parse).

What is kept, per source (all configurable, see deploy/filter.example.json):
  TFMS   messages are BATCHES: one message holds several <fltdMessage> flight elements (often different
         flights). Each flight element is filtered on its own; the message is re-assembled from the kept
         elements (kept elements are byte-identical to the original, the envelope and any text between
         elements is preserved, so it stays well-formed XML).
         Airline flights only. Keep FlightTimes, FlightCreate/Modify/ScheduleActivate, flight plans,
         amendments, cancellations, departure/arrival information; ALL flow data and Status.
         Track (trackInformation) updates are thinned to 1 per flight per `track_interval_min`
         (0 = drop them). Other flight message types are dropped.
  SFDPS  airline flights only. Keep flight messages that carry aircraftDescription (flight plan data),
         the FIRST message that shows a given actual departure/arrival runway time for a flight,
         and Status. Position/track messages are thinned like TFMS tracks. Batch messages
         (many flights in one message) are split per flight and re-assembled from the kept flights.
  STDDS  the high-volume surface feed is cut down to what linkage needs: kept are SMES
         SurfaceMovementEventMessage (runway on/off events), TDES TowerDepartureEventMessage and
         TDLSCSPMessage (departure / PDC data), status messages (ISMC, DATIS, asdexMsg system status) and
         anything unrecognised (fail open). Dropped: asdexMsg position reports (positionReport/adsbReport,
         about 90% of the bytes) and SafetyLogicHoldBar. Only the block list applies if disabled.

LADD hook: if the block-list file exists, flights on it are dropped first, whatever else applies, on every
path: per flight element in TFMS/SFDPS batches (also with SWIM_FILTER=off), and by a raw-text scan of the whole
message for STDDS, unparseable messages, binary bodies and other non-flight data (best effort: a hit drops the
whole message). Flow messages are kept whole (see deploy/LADD_BLOCKLIST.md). Anything else unparseable is KEPT
(fail open) and counted.
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
              "track_interval_min": 5, "keep_message_types": [], "keep_cancellations": True},
    "stdds": {"enabled": True, "drop_roots": ["asdexMsg", "SafetyLogicHoldBar"]},
}

CALLSIGN_RE = re.compile(r"^([A-Z]{3})(\d{1,4})([A-Z]{0,2})$")
STDDS_ROOT_RE = re.compile(r"<(?!\?)(?:\w+:)?(\w+)")
ATTR_RE = re.compile(r'(\w+)="([^"]*)"')
TFMS_HEAD_RE = re.compile(r"<(?:\w+:)?fltdMessage\b([^>]*)>")
SF_MESSAGE_RE = re.compile(r"<message\b.*?</message>", re.S)
TFMS_BLOCK_RE = re.compile(r"<((?:\w+:)?)fltdMessage\b[^>]*(?<!/)>.*?</\1fltdMessage>", re.S)
SF_CANCEL_RE = re.compile(r'fdpsFlightStatus="CANCELLED"')
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
        self._rx = None
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
            self._mtime, self.ids, self._rx = None, frozenset(), None
            return
        if mtime == self._mtime:
            return
        try:
            ids = set()
            for line in self.path.read_text(errors="replace").splitlines():
                ids.update(t.upper() for t in re.split(r"[\s,;]+", line.split("#", 1)[0]) if t)
            self.ids, self._mtime = frozenset(ids), mtime
            self._rx = (re.compile(r"(?<![A-Z0-9])(?:" + "|".join(re.escape(i) for i in sorted(ids, key=len, reverse=True))
                                   + r")(?![A-Z0-9])", re.I) if ids else None)
            log.info("block list loaded: %d identifiers", len(ids))
        except OSError:
            log.warning("block list unreadable; keeping the previous one")

    def scan(self, text: str) -> bool:
        """Best effort: does any listed identifier appear as a whole token anywhere in the raw text?"""
        self.reload()
        return bool(self._rx and self._rx.search(text))

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

    def _guard(self, text, keep_reason):
        """Keep `text` unless the block list matches anywhere in it (best effort, whole message)."""
        if self.blocklist.scan(text):
            return None, "drop:ladd_text"
        return text, keep_reason

    def check_binary(self, raw: bytes) -> bool:
        """True = keep. Used for non-UTF-8 bodies: best-effort block-list scan of the raw bytes."""
        keep = not self.blocklist.scan(raw.decode("latin-1"))
        self.counts["keep:binary" if keep else "drop:ladd_text"] += 1
        return keep

    def apply(self, props: dict, body: str, now: float):
        try:
            if self.source == "tfms":
                out, reason = self._tfms(props, body, now)
            elif self.source == "sfdps":
                out, reason = self._sfdps(props, body, now)
            elif self.source == "stdds":
                out, reason = self._stdds(body)
            else:
                out, reason = self._guard(body, "keep:passthrough")
        except Exception as e:  # never lose data because of a filter bug (the block list still gets a try)
            try:
                out, reason = self._guard(body, f"keep:filter_error:{type(e).__name__}")
            except Exception:
                out, reason = body, f"keep:filter_error:{type(e).__name__}"
        self.counts[reason] += 1
        return out, reason

    def _assemble(self, body, blocks, kept, why):
        """Re-serialise a batch from its kept elements (byte-identical elements, original envelope/separators)."""
        best = next((r for r in why if r.startswith("keep")), None)
        if not kept:
            return None, "drop:" + why.most_common(1)[0][0].split(":", 1)[1]
        if len(kept) == len(blocks):
            return body, best
        sep = body[blocks[0].end():blocks[1].start()] if len(blocks) > 1 else ""
        return body[:blocks[0].start()] + sep.join(kept) + body[blocks[-1].end():], best

    # -- TFMS
    def _tfms(self, props, body, now):
        c = self.cfg
        cls = str(props.get("TFMDataClass", ""))
        on = c.get("enabled", True)
        if cls == "FlightData":
            return self._tfms_flights(body, now, ladd_only=not on)
        if not on:
            return self._guard(body, "keep:filter_off")
        if cls == "FlowInformation":  # kept whole: see deploy/LADD_BLOCKLIST.md
            return (body, "keep:flow") if c["keep_flow"] else (None, "drop:flow")
        if cls == "Status":
            return (body, "keep:status") if c["keep_status"] else (None, "drop:status")
        return self._guard(body, "keep:other_class")

    def _tfms_flights(self, body, now, ladd_only):
        blocks = list(TFMS_BLOCK_RE.finditer(body))
        if not blocks:
            return self._guard(body, "keep:unparsed")
        kept, why = [], Counter()
        for b in blocks:
            r = self._tfms_block(b.group(0), now, ladd_only)
            why[r] += 1
            if r.startswith("keep"):
                kept.append(b.group(0))
        return self._assemble(body, blocks, kept, why)

    def _tfms_block(self, blk, now, ladd_only):
        c = self.cfg
        m = TFMS_HEAD_RE.search(blk[:4000])
        if not m:
            return "drop:ladd_text" if self.blocklist.scan(blk) else "keep:unparsed"
        a = dict(ATTR_RE.findall(m.group(1)))
        acid, mtype = a.get("acid", ""), a.get("msgType", "")
        if self.blocklist.hit(acid):
            return "drop:ladd"
        if ladd_only:
            return "keep:filter_off"
        if not self._is_airline(acid):
            return "drop:not_airline"
        if mtype in c["keep_types"]:
            return "keep:" + mtype
        if mtype in c["track_types"]:
            return "keep:track_sample" if self._track_ok(a.get("flightRef") or acid, now) else "drop:track_thinned"
        return "drop:type:" + (mtype or "unknown")

    # -- STDDS
    def _stdds(self, body):
        if not self.cfg.get("enabled", True):
            return self._guard(body, "keep:passthrough")
        m = STDDS_ROOT_RE.search(body[:3000])
        root = m.group(1) if m else ""
        if root in self.cfg.get("drop_roots", ()):
            # asdexMsg also carries system status ("degraded", ...): that is status, keep it
            if root == "asdexMsg" and "<systemStatus" in body and "<positionReport" not in body \
                    and "<adsbReport" not in body:
                return self._guard(body, "keep:status")
            return None, "drop:" + root
        return self._guard(body, "keep:" + (root or "unparsed"))

    # -- SFDPS
    def _sfdps(self, props, body, now):
        c = self.cfg
        mtype = str(props.get("FDPS_MessageType", ""))
        on = c.get("enabled", True)
        if str(props.get("FDPS_DataType", "")).upper() == "STATUS" or mtype.lower() == "status":
            return self._guard(body, "keep:status" if on else "keep:filter_off")
        if on and mtype in c["keep_message_types"]:
            return self._guard(body, "keep:configured_type")
        blocks = list(SF_MESSAGE_RE.finditer(body))
        if not blocks:
            return self._guard(body, "keep:unparsed" if "<flight " in body else "keep:nonflight")
        kept, why = [], Counter()
        for b in blocks:
            r = self._sf_block(b.group(0), now, ladd_only=not on)
            why[r] += 1
            if r.startswith("keep"):
                kept.append(b.group(0))
        return self._assemble(body, blocks, kept, why)

    def _sf_block(self, blk, now, ladd_only=False):
        m = SF_IDENT_RE.search(blk)
        ident = m.group(1) if m else ""
        reg, addr = SF_REG_RE.search(blk), SF_ADDR_RE.search(blk)
        if self.blocklist.hit(ident, reg and reg.group(1), addr and addr.group(1)):
            return "drop:ladd"
        if ladd_only:
            return "keep:filter_off"
        if not self._is_airline(ident):
            return "drop:not_airline"
        if "<aircraftDescription" in blk:
            return "keep:flight_plan"
        if self.cfg.get("keep_cancellations", True) and SF_CANCEL_RE.search(blk):
            return "keep:cancellation"
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
