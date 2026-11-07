"""L1 extractor: tails the archived SWIM hour files and keeps data/linkage/live.sqlite current.

  python -m linkage.extract                      # live: poll every 60 s (launchd com.flyrely.linkage.extract)
  python -m linkage.extract --backfill 2026-11-07   # re-extract one UTC day from the archive, then exit
  python -m linkage.extract --once               # one poll, then exit

Input: data/swim/<src>/<YYYY-MM-DD>/<src>_<YYYYmmddTHH>_<run>.jsonl.zst, one JSON record per line
({"recv", "props", "body"}); `recv` is the archived receive time and is the as-of time of every fact.
Tailing: each file keeps a streaming zstd decompressor and a byte offset; new bytes are fed to it every poll
(the consumer flushes to a decodable block boundary after every batch, see swim/archive.py). If the decoder
ever errors, the file is re-read from the start and records already processed (counted in `ingest_progress`)
are skipped. All writes are idempotent upserts, so a restart or an overlap is harmless.
Privacy: LADD (swim.filters.Blocklist: callsign, registration, icao24) and FDPS_BlockAtFAA / FDPS_Sensitive /
blockAtFAA drops are applied before anything is stored; the leg and all its facts are removed.
Logs and status files contain counts only, never registrations or icao24 values.
"""
import argparse
import json
import logging
import os
import re
import sys
import time
import zlib
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from linkage import carriers as cr
from linkage.store import Store, norm_recv, norm_time, parse_ts
from swim import archive, filters

log = logging.getLogger("linkage")

REPO = Path(__file__).resolve().parent.parent
DATA_SWIM = REPO / "data" / "swim"
DEFAULT_DB = REPO / "data" / "linkage" / "live.sqlite"
SOURCES = ("tfms", "sfdps", "stdds")
CARRIERS = frozenset(filters.DEFAULT_CARRIERS)

# ---------------------------------------------------------------- regexes (raw XML, no full parse)
_ATTR = re.compile(r'(\w+)="([^"]*)"')
TFMS_BLOCK = filters.TFMS_BLOCK_RE
TFMS_HEAD = filters.TFMS_HEAD_RE
_IGTD = re.compile(r"<(?:\w+:)?igtd>([^<]+)<")
_TGUFI = re.compile(r"<(?:\w+:)?gufi>([^<]+)<")
_FAC = re.compile(r"<(?:\w+:)?facilityIdentifier>([^<]+)<")
_FSTATUS = re.compile(r"<(?:\w+:)?flightStatus>([^<]+)<")
_MODEL = re.compile(r"<(?:\w+:)?aircraftModel>([^<]+)<")
_SPECS = re.compile(r"<(?:\w+:)?flightAircraftSpecs\b[^>]*>([^<]+)<")
_FTD = re.compile(r"<(?:\w+:)?flightTimeData\b([^>]*)>")
_ETD = re.compile(r"<(?:\w+:)?etd\b([^>]*)>")
_ETA = re.compile(r"<(?:\w+:)?eta\b([^>]*)>")
SF_MSG = filters.SF_MESSAGE_RE
_SF_FLIGHT = re.compile(r"<flight\b([^>]*)>")
_SF_DEP_PT = re.compile(r'<departure\b[^>]*\bdeparturePoint="([^"]*)"')
_SF_ARR_PT = re.compile(r'<arrival\b[^>]*\barrivalPoint="([^"]*)"')
_SF_RWY = re.compile(r'<runwayTime>\s*<(estimated|actual)\s+time="([^"]+)"', re.S)
_SF_STATUS = re.compile(r'fdpsFlightStatus="([^"]*)"')
_SF_FDPS_GUFI = re.compile(r'name="FDPS_GUFI"\s+value="([^"]*)"')
_SF_MODEL = re.compile(r"<icaoModelIdentifier>([^<]+)<")
_SF_BODY_BLOCK = re.compile(r'(?:BlockAtFAA|Sensitive)"?\s*(?:value=)?"?true', re.I)
_GUFI_TIME = re.compile(r"us\.fdps\.(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)")
_ST_ROOT = re.compile(r"<(?:\w+:)?(SurfaceMovementEventMessage|TDLSCSPMessage|TowerDepartureEventMessage)\b")
_ST_BLOCK = re.compile(r"blockAtFAA\W{0,4}true", re.I)
_PDC_AN = re.compile(r"/AN\s+([A-Z0-9]{2,7})")


def _tag(blk, name):
    m = re.search(rf"<{name}>([^<]*)<", blk)
    return m.group(1).strip() if m else ""


def _flag(v) -> bool:
    return str(v).strip().lower() == "true"


# ---------------------------------------------------------------- pure parsers
def parse_tfms_block(blk: str):
    """One <fltdMessage> element -> dict of facts, or None when it carries no usable identity."""
    m = TFMS_HEAD.search(blk[:4000])
    if not m:
        return None
    h = dict(_ATTR.findall(m.group(1)))
    cs, mtype = h.get("acid", "").upper(), h.get("msgType", "")
    parts = cr.split_callsign(cs)
    igtd = _IGTD.search(blk)
    if not parts or not igtd:
        return None
    fac = _FAC.search(blk)
    ftd = dict(_ATTR.findall(_FTD.search(blk).group(1))) if _FTD.search(blk) else {}
    f = {
        "callsign": cs, "carrier": parts[0], "flight_no": parts[1], "msg_type": mtype,
        "origin": cr.norm_airport(h.get("depArpt", "")), "dest": cr.norm_airport(h.get("arrArpt", "")),
        "igtd": norm_time(igtd.group(1)), "flight_ref": h.get("flightRef", ""),
        "eram_gufi": (_TGUFI.search(blk).group(1) if _TGUFI.search(blk) else ""),
        "in_schedule": mtype in ("FlightCreate", "FlightScheduleActivate") or (fac and fac.group(1) == "OAG"),
        "status": (_FSTATUS.search(blk).group(1) if _FSTATUS.search(blk) else ""),
    }
    model = _MODEL.search(blk) or _SPECS.search(blk)
    f["aircraft_type"] = model.group(1).strip() if model else ""
    f["sched_in"] = norm_time(ftd["airlineInTime"]) if ftd.get("airlineInTime") and f["in_schedule"] else ""
    act = {}
    for attr, key in (("gateDeparture", "out"), ("runwayDeparture", "off"), ("runwayArrival", "on"),
                      ("gateArrival", "in")):
        if ftd.get(attr):
            act[key] = norm_time(ftd[attr])
    for tag in _ETD.findall(blk):
        a = dict(_ATTR.findall(tag))
        if a.get("etdType") == "ACTUAL" and a.get("timeValue"):
            act.setdefault("off", norm_time(a["timeValue"]))
    for tag in _ETA.findall(blk):
        a = dict(_ATTR.findall(tag))
        if a.get("etaType") == "ACTUAL" and a.get("timeValue"):
            act.setdefault("on", norm_time(a["timeValue"]))
    f["actuals"] = act
    f["cancelled"] = mtype == "flightPlanCancellation" or f["status"].upper() == "CANCELLED"
    return f


def parse_sfdps_block(blk: str, props=None):
    """One SFDPS <message> flight element -> dict, or None when there is no airline-style identity."""
    props = props or {}
    m = filters.SF_IDENT_RE.search(blk)
    cs = (m.group(1) if m else "").upper()
    parts = cr.split_callsign(cs)
    if not parts:
        return None
    reg, addr = filters.SF_REG_RE.search(blk), filters.SF_ADDR_RE.search(blk)
    dep, arr = _SF_DEP_PT.search(blk), _SF_ARR_PT.search(blk)
    fl = dict(_ATTR.findall(_SF_FLIGHT.search(blk).group(1))) if _SF_FLIGHT.search(blk) else {}
    fg = _SF_FDPS_GUFI.search(blk)
    gufi = fg.group(1) if fg else ""
    if not gufi:
        g2 = filters.SF_GUFI_RE.search(blk)
        gufi = g2.group(1) if g2 else ""
    created = _GUFI_TIME.search(gufi)
    dep_blk, arr_blk = filters.SF_DEP_RE.search(blk), filters.SF_ARR_RE.search(blk)
    est, act = {}, {}
    for tag, key, section in (("dep", "off", dep_blk), ("arr", "on", arr_blk)):
        if section:
            for kind, t in _SF_RWY.findall(section.group(0)):
                (act if kind == "actual" else est)[key] = norm_time(t)
    st = _SF_STATUS.search(blk)
    ft = fl.get("flightType", "")
    model = _SF_MODEL.search(blk)
    return {
        "callsign": cs, "carrier": parts[0], "flight_no": parts[1],
        "origin": cr.norm_airport(dep.group(1) if dep else str(props.get("FDPS_Origin", ""))),
        "dest": cr.norm_airport(arr.group(1) if arr else str(props.get("FDPS_DestId", ""))),
        "reg": reg.group(1).upper() if reg else "", "icao24": addr.group(1).upper() if addr else "",
        "gufi": gufi, "plan_created": created.group(1) if created else "",
        "status": st.group(1) if st else "", "flight_type": ft, "source": fl.get("source", ""),
        "est": est, "actuals": act, "cancelled": bool(st and st.group(1) == "CANCELLED"),
        "aircraft_type": model.group(1).strip() if model else "",
        "has_plan": "<aircraftDescription" in blk,
    }


def parse_stdds(body: str):
    """STDDS message -> dict of facts for SMES SE / TDES PD / TDES DE, else None."""
    m = _ST_ROOT.search(body[:3000])
    if not m:
        return None
    kind = m.group(1)
    sf = _tag(body, "sfdpsGufi")
    dest = cr.norm_airport(_tag(body, "destinationAirport"))
    blocked = bool(_ST_BLOCK.search(body))
    if kind == "SurfaceMovementEventMessage":
        cs = _tag(body, "callsign").upper()
        return {"kind": "smes", "callsign": cs, "icao24": _tag(body, "acAddress").upper(),
                "airport": cr.norm_airport(_tag(body, "airport")), "event": _tag(body, "event"),
                "time": _tag(body, "time"), "gufi": sf, "origin": cr.norm_airport(_tag(body, "departureAirport")),
                "dest": dest, "blocked": blocked}
    if kind == "TDLSCSPMessage":
        an = _PDC_AN.search(body)
        return {"kind": "pdc", "callsign": _tag(body, "aircraftID").upper(), "reg": an.group(1).upper() if an else "",
                "airport": cr.norm_airport(_tag(body, "airportID")), "gufi": sf, "dest": dest,
                "origin": cr.norm_airport(_tag(body, "airportID")), "blocked": blocked}
    return {"kind": "tde", "callsign": _tag(body, "aircraftID").upper(),
            "airport": cr.norm_airport(_tag(body, "departureAirport")), "event": "pdc_clearance",
            "time": _tag(body, "clearanceDeliveryTime") or _tag(body, "eventTime"), "gufi": sf, "dest": dest,
            "origin": cr.norm_airport(_tag(body, "departureAirport")), "blocked": blocked}


# ---------------------------------------------------------------- extractor
class Extractor:
    def __init__(self, store: Store, blocklist: filters.Blocklist = None):
        self.store = store
        self.blocklist = blocklist or filters.Blocklist(None)
        self._ladd_ids = frozenset(self.blocklist.ids)
        self.counts = Counter()
        self.max_recv = {}

    # -- privacy
    def _ladd(self, *idents) -> bool:
        return self.blocklist.hit(*idents)

    def check_blocklist_change(self):
        """Scrub stored data when the LADD list changes (call once per poll)."""
        self.blocklist.reload()
        if self.blocklist.ids != self._ladd_ids:
            self._ladd_ids = frozenset(self.blocklist.ids)
            self.store.begin()
            n = self.store.scrub(self._ladd_ids)
            self.store.commit()
            self.counts["scrubbed_legs"] += n
            log.info("block list changed (%d identifiers): scrubbed %d legs", len(self._ladd_ids), n)

    # -- dispatch
    def process_record(self, source: str, rec: dict):
        recv = norm_recv(rec["recv"])
        if source not in self.max_recv or recv > self.max_recv[source]:
            self.max_recv[source] = recv
        body = rec.get("body")
        if not body:
            self.counts["skip_nobody"] += 1
            return
        props = rec.get("props") or {}
        self.counts[f"msgs_{source}"] += 1
        if source == "tfms":
            for b in TFMS_BLOCK.finditer(body):
                self._tfms(b.group(0), recv)
        elif source == "sfdps":
            blocked_msg = _flag(props.get("FDPS_BlockAtFAA")) or _flag(props.get("FDPS_Sensitive"))
            for b in SF_MSG.finditer(body):
                self._sfdps(b.group(0), props, recv, blocked_msg)
        elif source == "stdds":
            self._stdds(body, recv)

    def process_line(self, source: str, line: str):
        try:
            rec = json.loads(line)
        except ValueError:
            self.counts["skip_badjson"] += 1
            return
        try:
            self.process_record(source, rec)
        except Exception as e:  # one bad message must not stop the extractor
            self.counts["error_" + type(e).__name__] += 1

    # -- TFMS
    def _tfms(self, blk, recv):
        f = parse_tfms_block(blk)
        if not f:
            return
        if self._ladd(f["callsign"]):
            self.counts["drop_ladd"] += 1
            self._block_by_callsign(f)
            return
        if f["carrier"] not in CARRIERS:
            return
        leg_id = f"{f['callsign']}|{f['origin']}|{f['dest']}|{f['igtd'][:10]}"
        if self.store.is_blocked(leg_id):
            self.counts["drop_blocked_leg"] += 1
            return
        existed = self.store.get_leg(leg_id) is not None
        if f["msg_type"] in ("trackInformation",):
            return
        self.store.upsert_leg(
            leg_id, recv, callsign=f["callsign"], carrier=f["carrier"], flight_no=f["flight_no"],
            origin=f["origin"], dest=f["dest"], in_schedule=f["in_schedule"],
            sched={"sched_out_utc": f["igtd"], "sched_in_utc": f["sched_in"], "aircraft_type": f["aircraft_type"],
                   "eram_gufi": f["eram_gufi"], "tfms_flight_ref": f["flight_ref"]},
            actuals=f["actuals"], cancelled=(True, f["eram_gufi"]) if f["cancelled"] else None)
        if not existed:
            self.store.adopt_orphans(leg_id)
        self.counts["tfms_facts"] += 1

    def _block_by_callsign(self, f):
        leg_id = f"{f['callsign']}|{f['origin']}|{f['dest']}|{f['igtd'][:10]}"
        self.store.block_leg(leg_id, "ladd")

    # -- SFDPS
    def _sfdps(self, blk, props, recv, blocked_msg):
        m = filters.SF_IDENT_RE.search(blk)
        ident = m.group(1) if m else ""
        reg, addr = filters.SF_REG_RE.search(blk), filters.SF_ADDR_RE.search(blk)
        flagged = blocked_msg or bool(_SF_BODY_BLOCK.search(blk))
        ladd = self._ladd(ident, reg and reg.group(1), addr and addr.group(1))
        if not (flagged or ladd) and not (
                "<aircraftDescription" in blk or "CANCELLED" in blk or "<actual" in blk):
            return  # position-only update: nothing the store keeps
        f = parse_sfdps_block(blk, props)
        if not f or f["carrier"] not in CARRIERS:
            return
        target = self._sf_target(f, recv)
        leg_id = self.store.gufi_leg(f["gufi"]) or self.store.find_leg(f["callsign"], f["origin"], f["dest"], target)
        if flagged or ladd:
            self.counts["drop_blockatfaa" if flagged else "drop_ladd"] += 1
            self.store.block_leg(leg_id, "blockatfaa" if flagged else "ladd")
            self.store.block_leg(f"{f['callsign']}|{f['origin']}|{f['dest']}|{target.strftime('%Y-%m-%d')}",
                                 "blockatfaa" if flagged else "ladd")
            self.store.block_leg("gufi:" + f["gufi"] if f["gufi"] else None, "gufi")
            return
        if leg_id is None:
            leg_id = f"{f['callsign']}|{f['origin']}|{f['dest']}|sf{recv[:10]}"
        if self.store.is_blocked(leg_id) or self.store.is_blocked("gufi:" + f["gufi"]):
            self.counts["drop_blocked_leg"] += 1
            return
        leg = self.store.get_leg(leg_id)
        cancelled = None
        if f["cancelled"]:
            cancelled = (True, f["gufi"])
        elif leg is not None and leg["cancelled"] and f["gufi"] and leg["cancelled_gufi"] \
                and f["gufi"] != leg["cancelled_gufi"] and recv > (leg["cancelled_recv"] or ""):
            cancelled = (False, None)  # a different plan was filed after the cancellation: re-filed, not cancelled
        elif leg is not None and leg["cancelled"] and f["actuals"].get("off") and recv > (leg["cancelled_recv"] or ""):
            cancelled = (False, None)
        self.store.upsert_leg(
            leg_id, recv, callsign=f["callsign"], carrier=f["carrier"], flight_no=f["flight_no"],
            origin=f["origin"], dest=f["dest"],
            sched={"flight_type": f["flight_type"], "sfdps_gufi": f["gufi"], "aircraft_type": f["aircraft_type"]},
            actuals=f["actuals"], est=f["est"], cancelled=cancelled)
        self.store.map_gufi(f["gufi"], leg_id, recv)
        if f["reg"] or f["icao24"]:
            self.store.add_aircraft(leg_id, "sfdps", f["reg"], f["icao24"], recv)
        self.counts["sfdps_facts"] += 1

    @staticmethod
    def _sf_target(f, recv):
        if f["actuals"].get("off"):
            return parse_ts(f["actuals"]["off"])
        if f["est"].get("off"):
            return parse_ts(f["est"]["off"])
        return parse_ts(recv) + Store.DEFAULT_LEAD

    # -- STDDS
    def _stdds(self, body, recv):
        f = parse_stdds(body)
        if not f:
            return
        parts = cr.split_callsign(f["callsign"])
        if not parts or parts[0] not in CARRIERS:
            return
        flagged = f["blocked"]
        if flagged or self._ladd(f["callsign"], f.get("icao24"), f.get("reg")):
            self.counts["drop_blockatfaa" if flagged else "drop_ladd"] += 1
            lg = self.store.gufi_leg(f["gufi"])
            self.store.block_leg(lg, "blockatfaa" if flagged else "ladd")
            if flagged and f["gufi"]:
                self.store.block_leg("gufi:" + f["gufi"], "gufi")
            return
        if f["kind"] == "smes":
            if not f["event"]:
                return
            ev_time = norm_time(f["time"]) if f["time"] else recv
        elif f["kind"] == "tde":
            ev_time = norm_time(f["time"]) if f["time"] else recv
        else:
            ev_time = recv
        target = parse_ts(ev_time)
        if f["kind"] == "smes" and f["airport"] != f["origin"]:
            target -= timedelta(hours=2)  # event at the arrival end: the flight left earlier
        leg_id = self.store.gufi_leg(f["gufi"])
        if leg_id is None and f["origin"] and f["dest"]:
            leg_id = self.store.find_leg(f["callsign"], f["origin"], f["dest"], target)
        if self.store.is_blocked(leg_id) or self.store.is_blocked("gufi:" + f["gufi"]):
            self.counts["drop_blocked_leg"] += 1
            return
        if f["kind"] == "pdc":
            if leg_id and f["reg"]:
                self.store.add_aircraft(leg_id, "pdc", f["reg"], "", recv)
                self.counts["pdc_facts"] += 1
            return
        self.store.add_surface_event(f["airport"], f["callsign"], f.get("icao24", ""), f["event"], ev_time, recv,
                                     f["gufi"], leg_id)
        if f["kind"] == "smes" and f["icao24"] and leg_id:
            self.store.add_aircraft(leg_id, "smes", "", f["icao24"], recv)
        self.counts["surface_facts"] += 1


# ---------------------------------------------------------------- tailing
class FileTail:
    """Incremental reader for one growing .jsonl.zst / .jsonl.gz file."""

    CHUNK = 1 << 16  # compressed bytes per read: bounds the decompressed burst (XML packs 20-50x)

    def __init__(self, path: Path, skip_lines: int = 0):
        self.path, self.offset, self.buf = Path(path), 0, b""
        self.lines_done = 0
        self.skip = skip_lines
        self.reread = path.name.endswith(".gz")  # gzip fallback codec: always re-read with a line count
        self.dobj = None
        self._reset_decoder()

    def _reset_decoder(self):
        self.offset, self.buf, self.dobj = 0, b"", None
        if not self.reread:
            zstd = archive.zstd_module()
            if zstd is None:
                raise RuntimeError("zstandard module needed to read .zst archives")
            self.dobj = zstd.ZstdDecompressor().decompressobj(read_across_frames=True)

    def poll(self):
        """Generator of the new complete lines since the last poll (those already done before a restart are
        skipped). Streams chunk by chunk so a large file never sits in memory; `lines_done` counts a line only
        once it has been handed out."""
        try:
            size = self.path.stat().st_size
        except OSError:
            return
        if self.reread:
            yield from self._poll_reread(size)
            return
        if size <= self.offset:
            return
        zstd = archive.zstd_module()
        try:
            with open(self.path, "rb") as fh:
                fh.seek(self.offset)
                while self.offset < size:
                    chunk = fh.read(min(self.CHUNK, size - self.offset))
                    if not chunk:
                        break
                    self.offset += len(chunk)
                    data = self.dobj.decompress(chunk)
                    if data:
                        self.buf += data
                        *lines, self.buf = self.buf.split(b"\n")
                        yield from self._count(lines)
        except zstd.ZstdError:
            log.warning("%s: decoder error, falling back to re-read with skip", self.path.name)
            self.reread = True
            self.skip = self.lines_done  # lines consumed so far
            yield from self._poll_reread(size, force=True)

    def _poll_reread(self, size, force=False):
        """Re-read the whole file; `skip` is the number of lines already consumed."""
        if not force and size <= self.offset:
            return
        self.offset = size
        n = 0
        for n, line in enumerate(archive.read_lines(self.path), 1):  # stops cleanly at a torn tail
            if n > self.skip:
                self.skip = self.lines_done = n
                yield line

    def _count(self, raw_lines):
        for ln in raw_lines:
            if self.skip:
                self.skip -= 1
                self.lines_done += 1
                continue
            self.lines_done += 1
            yield ln.decode("utf-8", "replace")


def day_dirs(root: Path, source: str, days):
    return [root / source / d for d in days if (root / source / d).is_dir()]


def hour_files(root: Path, source: str, days):
    out = []
    for d in day_dirs(root, source, days):
        for ext in archive.EXTS:
            out += d.glob(f"*{ext}")
    return sorted(out, key=lambda p: (p.parent.name, p.name))


class Runner:
    def __init__(self, store: Store, ex: Extractor, root: Path = DATA_SWIM, status_path: Path = None):
        self.store, self.ex, self.root, self.status_path = store, ex, Path(root), status_path
        self.reset()

    def reset(self):
        """Drop in-memory tails; they restart from the persisted line counts (after a failed poll)."""
        self.tails = {}
        self.store._gufi_cache.clear()
        self.progress = {r[0]: r[1] for r in self.store.db.execute("SELECT path, lines_done FROM ingest_progress")}

    @staticmethod
    def recent_days(n=2, now=None):
        now = now or datetime.now(timezone.utc)
        return [(now - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n - 1, -1, -1)]

    def poll(self, days=None):
        days = days or self.recent_days()
        self.ex.check_blocklist_change()
        n = 0
        self.store.begin()
        for src in SOURCES:
            for p in hour_files(self.root, src, days):
                key = str(p)
                t = self.tails.get(key)
                if t is None:
                    t = self.tails[key] = FileTail(p, skip_lines=self.progress.get(key, 0))
                for line in t.poll():
                    self.ex.process_line(src, line)
                    n += 1
                self.store.db.execute("INSERT INTO ingest_progress(path, lines_done) VALUES (?,?) "
                                      "ON CONFLICT(path) DO UPDATE SET lines_done=excluded.lines_done",
                                      (key, t.lines_done))
        self.store.commit()
        # forget tails of days that rolled out of the window
        for k in [k for k in self.tails if Path(k).parent.name not in days]:
            del self.tails[k]
        self.write_status(n)
        return n

    def write_status(self, n_lines):
        now = datetime.now(timezone.utc)
        lag = {}
        for src, rv in self.ex.max_recv.items():
            lag[src] = round((now - parse_ts(rv)).total_seconds(), 1)
        st = {"updated": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "lines_last_poll": n_lines,
              "store_lag_s": lag, "counts": self.store.counts(), "extractor": dict(self.ex.counts)}
        self.store.set_meta("status", json.dumps(st))
        if self.status_path:
            tmp = Path(str(self.status_path) + ".tmp")
            tmp.write_text(json.dumps(st))
            os.replace(tmp, self.status_path)
        return st


def backfill(store: Store, ex: Extractor, root: Path, day: str) -> dict:
    """Re-extract one UTC day from the archive (all sources, TFMS first so schedule legs exist before SFDPS/STDDS
    facts are matched). Idempotent: running it again changes nothing."""
    ex.check_blocklist_change()
    for src in SOURCES:
        for p in hour_files(root, src, [day]):
            store.begin()
            n = 0
            for line in archive.read_lines(p):
                ex.process_line(src, line)
                n += 1
            store.commit()
            log.info("backfill %s %s: %d records", src, p.name, n)
    return store.counts()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backfill", metavar="YYYY-MM-DD", help="re-extract this UTC day from the archive, then exit")
    ap.add_argument("--once", action="store_true", help="one poll, then exit")
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--data-root", type=Path, default=DATA_SWIM)
    ap.add_argument("--blocklist", type=Path, default=None, help="default: SWIM_BLOCKLIST_FILE or data/ladd_blocklist.txt")
    ap.add_argument("--days", type=int, default=2, help="live mode: how many recent UTC days of files to follow")
    ap.add_argument("--log-file", type=Path, default=None)
    a = ap.parse_args(argv)
    handlers = None
    if a.log_file:
        from logging.handlers import RotatingFileHandler
        handlers = [RotatingFileHandler(a.log_file, maxBytes=10_000_000, backupCount=3)]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers)
    bl = a.blocklist or Path(os.environ.get("SWIM_BLOCKLIST_FILE", a.data_root.parent / "ladd_blocklist.txt"))
    store = Store(a.db)
    ex = Extractor(store, filters.Blocklist(bl))
    if a.backfill:
        t0 = time.time()
        c = backfill(store, ex, a.data_root, a.backfill)
        print(json.dumps({"backfill": a.backfill, "seconds": round(time.time() - t0, 1), **c,
                          "extractor": dict(ex.counts)}))
        return 0
    runner = Runner(store, ex, a.data_root, a.db.parent / "status.json")
    log.info("linkage extractor started; db=%s", a.db)
    while True:
        t0 = time.time()
        try:
            n = runner.poll(Runner.recent_days(a.days))
            log.info("poll: %d lines in %.1fs; %s", n, time.time() - t0, json.dumps(store.counts()))
        except Exception:
            log.exception("poll failed")
            if store.db.in_transaction:
                store.db.execute("ROLLBACK")
            runner.reset()
        if a.once:
            return 0
        time.sleep(max(1.0, a.interval - (time.time() - t0)))


if __name__ == "__main__":
    sys.exit(main())
