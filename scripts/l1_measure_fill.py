"""L1 plan, task T1: full-day tail-registration fill measurement over the raw SWIM archives.

    python scripts/l1_measure_fill.py --date 2026-11-08 [--root data/swim] [--out backtest/results/l1_fill_2026-11-08.json]

READ-ONLY: reads data/swim/{tfms,sfdps,stdds} archives, writes one JSON of aggregate numbers. It never
prints, stores or logs a registration, ICAO24 address or any per-flight identifier: the output has counts,
shares and lead-time deciles per carrier only.

Definitions (UTC day D):
  legs         TFMS FlightCreate / FlightScheduleActivate flights with igtd in D, both ends in the US
               (airports in models/airports_ref.csv; ICAO and IATA codes are normalised to IATA), carrier = first 3 letters of the callsign.
               Duplicates merged on (callsign, origin, dest, igtd).
  reg lead     igtd minus the FIRST archived receive time of an SFDPS message that carries a registration
               for the flight. An SFDPS message is assigned to the leg with the earliest igtd that is not
               more than 90 min before the message (same callsign, origin, dest). Negative = the registration
               first appeared after the scheduled gate time.
  Carriers     NK (Spirit) is defunct: reported as "not expected", not as missing. Hawaiian (HAL/HA) now
               operates under Alaska codes: HAL callsigns are counted inside ASA.
  known_at_60  share of ALL scheduled legs with reg lead >= 60 min (at T-1h the tail was already known);
  known_at_180 same for 180 min (T-3h).
  coverage     the JSON records per-source archive coverage. complete_day is true only if SFDPS and STDDS have
               data in all 24 hours of D and TFMS in all 24 hours of D-1 (legs are activated about 24 h
               before igtd); otherwise the numbers are partial.
  SMES         a SurfaceMovementEventMessage whose callsign and airport (origin or destination) match a leg
               (nearest igtd within 12 h).
  PDC          a TDLSCSPMessage whose aircraft id and airport (origin) match a leg; pdc_with_reg = its text
               carries a registration (/AN <reg>); only the presence is counted.
"""
import argparse
import csv
import json
import re
import sys
from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from swim import archive, filters  # noqa: E402

CARRIERS = {  # ICAO -> label, the 10 target carriers (plan section 1.3); every other carrier in the
    "AAL": "AA", "DAL": "DL", "UAL": "UA", "SWA": "WN", "JBU": "B6",  # SWIM filter's list is reported
    "ASA": "AS (incl. Hawaiian)", "NKS": "NK (defunct, not expected)", "FFT": "F9", "AAY": "G4",  # as a regional or "other".
}
REGIONALS = ("SKW", "RPA", "ENY", "EDV", "PDT", "PSA", "JIA", "CPZ", "GJS", "ASH", "QXE", "AWI", "UCA", "MXY", "MKU", "KAP")
ICAO_IATA_FIX = {"TJSJ": "SJU", "TJBQ": "BQN", "TJPS": "PSE", "TIST": "STT", "TISX": "STX", "PGUM": "GUM"}
IGTD_RE = re.compile(r"<(?:\w+:)?igtd>([^<]+)<")
SF_DEPPT_RE = re.compile(r'\bdeparturePoint="([^"]*)"')
SF_ARRPT_RE = re.compile(r'\barrivalPoint="([^"]*)"')
SE_CALL_RE = re.compile(r"<callsign>([^<]*)<")
SE_AIRPORT_RE = re.compile(r"<airport>([^<]*)<")
TD_ID_RE = re.compile(r"<aircraftID>([^<]*)<")
TD_AIRPORT_RE = re.compile(r"<airportID>([^<]*)<")
PDC_REG_RE = re.compile(r"/AN\s+[A-Z0-9-]{3,7}")
SF_WINDOW = timedelta(minutes=90)
MAX_LEAD = timedelta(hours=12)  # a plan message more than 12 h before igtd is yesterday's flight, not this leg
STDDS_WINDOW = timedelta(hours=12)
DECILES = [10, 20, 30, 40, 50, 60, 70, 80, 90]


def parse_ts(s):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except (ValueError, AttributeError):
        return None


_US_IATA = None


def us_airports():
    """IATA codes of the US airports in models/airports_ref.csv (BTS universe, territories included)."""
    global _US_IATA
    if _US_IATA is None:
        try:
            with open(ROOT / "models" / "airports_ref.csv", newline="") as f:
                _US_IATA = {r["iata"] for r in csv.DictReader(f)}
        except OSError:
            _US_IATA = set()
    return _US_IATA


def to_iata(code):
    """TFMS mixes IATA (EWR) and ICAO (KEWR); SFDPS and STDDS use ICAO. Normalise to IATA where possible."""
    code = (code or "").upper()
    if code in ICAO_IATA_FIX:
        return ICAO_IATA_FIX[code]
    return code[1:] if len(code) == 4 and code[0] in "KP" else code


def is_us(iata):
    return iata in us_airports()


# ------------------------------------------------------------------ message parsers (pure)
def tfms_legs(body):
    """[(callsign, origin, dest, igtd datetime)] from FlightCreate / FlightScheduleActivate elements."""
    out = []
    for b in filters.TFMS_BLOCK_RE.finditer(body):
        blk = b.group(0)
        m = filters.TFMS_HEAD_RE.search(blk[:4000])
        if not m:
            continue
        a = dict(filters.ATTR_RE.findall(m.group(1)))
        if a.get("msgType") not in ("FlightCreate", "FlightScheduleActivate"):
            continue
        g = IGTD_RE.search(blk)
        igtd = parse_ts(g.group(1)) if g else None
        acid, dep, arr = a.get("acid", "").upper(), to_iata(a.get("depArpt")), to_iata(a.get("arrArpt"))
        if igtd and acid and dep and arr:
            out.append((acid, dep, arr, igtd))
    return out


def sfdps_reg_flights(body):
    """[(callsign, origin, dest)] for each SFDPS message in `body` that carries a registration."""
    out = []
    for b in filters.SF_MESSAGE_RE.finditer(body):
        blk = b.group(0)
        reg = filters.SF_REG_RE.search(blk)
        ident = filters.SF_IDENT_RE.search(blk)
        dep, arr = SF_DEPPT_RE.search(blk), SF_ARRPT_RE.search(blk)
        if reg and reg.group(1) and ident and dep and arr:
            out.append((ident.group(1).upper(), to_iata(dep.group(1)), to_iata(arr.group(1))))
    return out


def sfdps_idents(body):
    return [m.group(1).upper() for m in filters.SF_IDENT_RE.finditer(body)]


def stdds_event(body):
    """('smes', callsign, airport) | ('pdc', callsign, airport, has_reg) | None."""
    head = body[:3000]
    if "SurfaceMovementEventMessage" in head:
        c, a = SE_CALL_RE.search(body), SE_AIRPORT_RE.search(body)
        return ("smes", c.group(1).upper(), to_iata(a.group(1))) if c and a else None
    if "TDLSCSPMessage" in head:
        c, a = TD_ID_RE.search(body), TD_AIRPORT_RE.search(body)
        return ("pdc", c.group(1).upper(), to_iata(a.group(1)), bool(PDC_REG_RE.search(body))) if c and a else None
    return None


# ------------------------------------------------------------------ analysis
class Leg:
    __slots__ = ("call", "dep", "arr", "igtd", "first_reg", "smes", "pdc", "pdc_reg")

    def __init__(self, call, dep, arr, igtd):
        self.call, self.dep, self.arr, self.igtd = call, dep, arr, igtd
        self.first_reg = None
        self.smes = self.pdc = self.pdc_reg = False


def build_legs(tfms_msgs):
    legs = {}
    for _recv, body in tfms_msgs:
        for call, dep, arr, igtd in tfms_legs(body):
            legs.setdefault((call, dep, arr, igtd), Leg(call, dep, arr, igtd))
    return sorted(legs.values(), key=lambda l: l.igtd)


def assign_registrations(legs, sfdps_msgs):
    """Set leg.first_reg = earliest recv of an SFDPS reg message assigned to the leg."""
    by_key = defaultdict(list)
    for l in legs:
        by_key[(l.call, l.dep, l.arr)].append(l)  # legs are igtd-sorted already
    for recv, body in sfdps_msgs:
        for key in sfdps_reg_flights(body):
            cands = by_key.get(key)
            if not cands:
                continue
            i = bisect_left([c.igtd for c in cands], recv - SF_WINDOW)
            if i < len(cands) and cands[i].igtd - recv <= MAX_LEAD and (cands[i].first_reg is None or recv < cands[i].first_reg):
                cands[i].first_reg = recv


def assign_stdds(legs, stdds_msgs):
    by_call = defaultdict(list)
    for l in legs:
        by_call[l.call].append(l)
    for recv, body in stdds_msgs:
        ev = stdds_event(body)
        if not ev:
            continue
        kind, call, airport = ev[:3]
        cands = [l for l in by_call.get(call, ()) if (l.dep == airport if kind == "pdc" else airport in (l.dep, l.arr))
                 and abs(l.igtd - recv) <= STDDS_WINDOW]
        if not cands:
            continue
        l = min(cands, key=lambda c: abs(c.igtd - recv))
        if kind == "smes":
            l.smes = True
        else:
            l.pdc = True
            l.pdc_reg = l.pdc_reg or ev[3]


def pct(sorted_vals, p):
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return round(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo), 1)


def share(n, d):
    return round(n / d, 4) if d else None


def summarise(legs):
    leads = sorted((l.igtd - l.first_reg).total_seconds() / 60 for l in legs if l.first_reg)
    n = len(legs)
    return {
        "scheduled_legs": n,
        "legs_with_reg": len(leads),
        "fill": share(len(leads), n),
        "reg_lead_min_deciles": {f"p{p}": pct(leads, p) for p in DECILES},
        "known_at_60": share(sum(1 for x in leads if x >= 60), n),
        "known_at_180": share(sum(1 for x in leads if x >= 180), n),
        "share_of_reg_legs_ge_60": share(sum(1 for x in leads if x >= 60), len(leads)),
        "share_of_reg_legs_ge_180": share(sum(1 for x in leads if x >= 180), len(leads)),
        "smes_legs": sum(l.smes for l in legs), "smes_share": share(sum(l.smes for l in legs), n),
        "pdc_legs": sum(l.pdc for l in legs), "pdc_share": share(sum(l.pdc for l in legs), n),
        "pdc_with_reg_legs": sum(l.pdc_reg for l in legs), "pdc_with_reg_share": share(sum(l.pdc_reg for l in legs), n),
    }


def analyse(day, tfms_msgs, sfdps_msgs, stdds_msgs, coverage=None):
    """The three message lists are lists of (recv datetime, body). Returns the JSON-able result."""
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    all_legs = build_legs(tfms_msgs)
    assign_registrations(all_legs, sfdps_msgs)
    assign_stdds(all_legs, stdds_msgs)
    legs = [l for l in all_legs if start <= l.igtd < end]
    domestic = [l for l in legs if is_us(l.dep) and is_us(l.arr)]
    by_carrier = defaultdict(list)
    for l in domestic:
        by_carrier["ASA" if l.call.startswith("HAL") else l.call[:3]].append(l)  # Hawaiian flies under Alaska codes
    carriers = {}
    for icao, label in CARRIERS.items():
        carriers[icao] = {"label": label, **summarise(by_carrier.get(icao, []))}
    regionals = {r: summarise(by_carrier[r]) for r in REGIONALS if r in by_carrier}
    other = [l for k, v in by_carrier.items() if k not in CARRIERS and k not in REGIONALS for l in v]
    sf_nks = len({i for _r, b in sfdps_msgs for i in sfdps_idents(b) if i.startswith("NKS")})
    st_nks = len({ev[1] for _r, b in stdds_msgs for ev in [stdds_event(b)] if ev and ev[1].startswith("NKS")})
    return {
        "date": day.isoformat(), "coverage": coverage or {},
        "legs_total_in_day": len(legs), "legs_domestic": len(domestic),
        "carriers": carriers, "regionals": regionals,
        "regionals_all": summarise([l for r in REGIONALS for l in by_carrier.get(r, [])]),
        "other_carriers": summarise(other),
        "all_carriers": summarise(domestic),
        "nks": {"status": "defunct, not expected (not a coverage gap)",
                "tfms_legs": sum(1 for l in legs if l.call.startswith("NKS")),
                "sfdps_distinct_callsigns": sf_nks, "stdds_distinct_callsigns": st_nks},
        "hal": {"status": "Hawaiian now operates under Alaska codes; HAL callsigns are counted in ASA",
                "tfms_legs_hal_callsign_merged_into_asa": sum(1 for l in legs if l.call.startswith("HAL"))},
    }


# ------------------------------------------------------------------ archive reading
def read_source(root, src, days):
    """[(recv, body)] from <root>/<src>/<day>/ for the given days (read-only)."""
    names = {d.isoformat() for d in days}
    out = []
    for f in archive.archive_files(Path(root) / src):
        if f.parent.name not in names:
            continue
        for line in archive.read_lines(f):
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            body, recv = rec.get("body"), parse_ts(rec.get("recv", ""))
            if body and recv:
                out.append((recv, body))
    return out


def coverage_of(msgs):
    if not msgs:
        return {"messages": 0, "first_recv": None, "last_recv": None}
    return {"messages": len(msgs), "first_recv": min(r for r, _ in msgs).isoformat(),
            "last_recv": max(r for r, _ in msgs).isoformat()}


def hours_covered(msgs, day_start):
    """How many of the 24 UTC hours from day_start have at least one archived message."""
    return len({int((r - day_start).total_seconds() // 3600) for r, _ in msgs if 0 <= (r - day_start).total_seconds() < 86400})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True, help="UTC day to measure, YYYY-MM-DD")
    ap.add_argument("--root", type=Path, default=ROOT / "data" / "swim")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    day = date.fromisoformat(args.date)
    d = lambda k: day + timedelta(days=k)
    tfms = read_source(args.root, "tfms", [d(-2), d(-1), d(0), d(1)])  # schedule messages arrive about a day ahead
    sfdps = read_source(args.root, "sfdps", [d(-1), d(0), d(1)])
    stdds = read_source(args.root, "stdds", [d(-1), d(0), d(1)])
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    cov = {"tfms": coverage_of(tfms), "sfdps": coverage_of(sfdps), "stdds": coverage_of(stdds)}
    # Legs are activated about 24 h before their igtd, so the schedule universe of day D is complete only if
    # TFMS has messages in every hour of D-1. SFDPS and STDDS must have messages in every hour of D.
    cov["tfms"]["hours_with_data_prev_day"] = hours_covered(tfms, start - timedelta(days=1))
    cov["sfdps"]["hours_with_data_day"] = hours_covered(sfdps, start)
    cov["stdds"]["hours_with_data_day"] = hours_covered(stdds, start)
    cov["schedule_universe_complete"] = cov["tfms"]["hours_with_data_prev_day"] == 24
    cov["complete_day"] = (cov["schedule_universe_complete"] and cov["sfdps"]["hours_with_data_day"] == 24
                           and cov["stdds"]["hours_with_data_day"] == 24)
    res = analyse(day, tfms, sfdps, stdds, cov)
    out = args.out or ROOT / "backtest" / "results" / f"l1_fill_{args.date}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(f"wrote {out}  (complete_day={cov['complete_day']}, domestic legs={res['legs_domestic']})")
    for icao, c in res["carriers"].items():
        print(f"  {icao} legs={c['scheduled_legs']} fill={c['fill']} known@60={c['known_at_60']} "
              f"known@180={c['known_at_180']} smes={c['smes_share']} pdc={c['pdc_share']}")
    r = res["regionals_all"]
    print(f"  regionals legs={r['scheduled_legs']} fill={r['fill']} known@60={r['known_at_60']} known@180={r['known_at_180']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
