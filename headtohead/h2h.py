"""
Live head-to-head: FlyRely vs Flighty on real flights.

Flighty has no API, so its numbers are typed in from the app at each checkpoint.
FlyRely's predictions are logged automatically at the same checkpoints, and the
two are scored against what actually happened.

    python headtohead/h2h.py add DL1234 2026-11-20T08:30 JFK LAX
    python headtohead/h2h.py due                          # checkpoints open now
    python headtohead/h2h.py snapshot                     # log FlyRely at due checkpoints (run every 10 min)
    python headtohead/h2h.py flighty 3 1d --ontime 82     # Flighty's on-time %, as shown
    python headtohead/h2h.py flighty 3 3h --delay 25      # Flighty's predicted/posted delay in minutes
    python headtohead/h2h.py outcome 3 --delay 22         # actual departure delay (or --cancelled)
    python headtohead/h2h.py outcomes-from-bts            # fill outcomes from BTS once published
    python headtohead/h2h.py score

Checkpoints: booking (when added), 7d, 1d, 3h, 1h before scheduled departure.
Set FLYRELY_API_URL to log what the deployed API returns; otherwise the model
is run in-process from main.py.
"""

import argparse
import csv
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DB = Path(os.getenv("H2H_DB", ROOT / "data" / "headtohead.sqlite"))
CHECKPOINTS = {"booking": None, "7d": timedelta(days=7), "1d": timedelta(days=1),
               "3h": timedelta(hours=3), "1h": timedelta(hours=1)}
WINDOW = timedelta(minutes=30)  # a checkpoint counts if logged within this long after its time

def _load_airport_tz() -> dict:
    """IANA time zone per airport from models/airports_ref.csv (scripts/build_airports_ref.py)."""
    with (ROOT / "models" / "airports_ref.csv").open(newline="") as f:
        return {r["iata"]: r["tz"] for r in csv.DictReader(f)}


AIRPORT_TZ = _load_airport_tz()

SCHEMA = """
create table if not exists flights (
  id integer primary key, carrier text, number text, origin text, dest text,
  operating_carrier text, operating_number text,
  sched_dep_local text, tz text, added_utc text,
  actual_delay real, cancelled integer, outcome_source text
);
create table if not exists predictions (
  flight_id integer, checkpoint text, source text, logged_utc text,
  p_delay real, delay_minutes real, ontime_pct real, model_version text,
  primary key (flight_id, checkpoint, source)
);
"""


def db() -> sqlite3.Connection:
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def now() -> datetime:
    return datetime.now(timezone.utc)


def sched_utc(f) -> datetime:
    return datetime.fromisoformat(f["sched_dep_local"]).replace(tzinfo=ZoneInfo(f["tz"])).astimezone(timezone.utc)


def checkpoint_time(f, cp: str) -> datetime:
    return datetime.fromisoformat(f["added_utc"]) if cp == "booking" else sched_utc(f) - CHECKPOINTS[cp]


def due(f, at: datetime) -> list[str]:
    """Checkpoints whose time has come and that are still inside their logging window."""
    out = []
    for cp in CHECKPOINTS:
        t = checkpoint_time(f, cp)
        if cp != "booking" and t < datetime.fromisoformat(f["added_utc"]):
            continue  # flight added after this checkpoint had passed
        if t <= at <= t + WINDOW or (cp == "booking" and at - t <= timedelta(hours=1)):
            out.append(cp)
    return out


def flyrely_predict(f) -> tuple[float, str]:
    body = {"flight_id": f"h2h-{f['id']}", "origin": f["origin"], "destination": f["dest"],
            "departure_time": f["sched_dep_local"], "airline": f["carrier"], "flight_number": f["number"]}
    url = os.getenv("FLYRELY_API_URL")
    if url:
        import httpx
        r = httpx.post(url.rstrip("/") + "/predict", json=body, timeout=30)
        r.raise_for_status()
        j = r.json()
        return j["delay_probability"], j.get("model_version", "")
    sys.path.insert(0, str(ROOT))
    import main
    airline = f["operating_carrier"] if f["operating_carrier"] in main.AIRLINE_DELAY_RATES else f["carrier"]
    features = main.build_features(f["origin"], f["dest"], datetime.fromisoformat(f["sched_dep_local"]), airline)
    p, _ = main.model_service.predict(features)
    return float(p), f"{main.MODEL_VERSION} (in-process)"


# =============================================================================
# Commands
# =============================================================================

def cmd_add(a):
    flight = a.flight.upper().replace(" ", "")
    carrier, number = flight[:2], flight[2:]
    origin, dest = a.origin.upper(), a.dest.upper()
    tz = a.tz or AIRPORT_TZ.get(origin)
    if not tz:
        sys.exit(f"don't know the time zone for {origin}; pass --tz (e.g. America/Chicago)")
    dep = datetime.fromisoformat(a.departure).replace(tzinfo=None)
    sys.path.insert(0, str(ROOT))
    import codeshare
    op = codeshare.resolve(carrier, number, origin, dest) or {}
    op_carrier, op_number = op.get("operating_airline", carrier), op.get("operating_flight_number", number)
    con = db()
    cur = con.execute("insert into flights (carrier, number, origin, dest, operating_carrier, operating_number, "
                      "sched_dep_local, tz, added_utc) values (?,?,?,?,?,?,?,?,?)",
                      (carrier, number, origin, dest, op_carrier, op_number, dep.isoformat(), tz, now().isoformat()))
    con.commit()
    fid = cur.lastrowid
    f = con.execute("select * from flights where id=?", (fid,)).fetchone()
    operated = f", operated by {op['operating_airline_name']} {op_carrier}{op_number}" if op_carrier != carrier else ""
    print(f"added #{fid}: {carrier}{number} {origin}->{dest} {dep:%Y-%m-%d %H:%M} ({tz}){operated}")
    snapshot_one(con, f, ["booking"])
    print(f"  now record what Flighty shows: h2h.py flighty {fid} booking --ontime <pct>")
    for cp in ("7d", "1d", "3h", "1h"):
        t = checkpoint_time(f, cp)
        if t > now():
            print(f"  {cp:>3} checkpoint: {t.astimezone(ZoneInfo(tz)):%a %b %d %H:%M} local")


def snapshot_one(con, f, cps):
    for cp in cps:
        if con.execute("select 1 from predictions where flight_id=? and checkpoint=? and source='flyrely'",
                       (f["id"], cp)).fetchone():
            continue
        p, version = flyrely_predict(f)
        con.execute("insert into predictions (flight_id, checkpoint, source, logged_utc, p_delay, model_version) "
                    "values (?,?,?,?,?,?)", (f["id"], cp, "flyrely", now().isoformat(), p, version))
        con.commit()
        print(f"#{f['id']} {cp}: FlyRely p(delay>=15) = {p:.3f}")


def cmd_snapshot(a):
    con = db()
    for f in con.execute("select * from flights where actual_delay is null and cancelled is null").fetchall():
        snapshot_one(con, f, [cp for cp in due(f, now()) if cp != "booking"])


def cmd_due(a):
    con = db()
    at = now()
    for f in con.execute("select * from flights where actual_delay is null and cancelled is null").fetchall():
        for cp in due(f, at):
            have = {r["source"] for r in con.execute(
                "select source from predictions where flight_id=? and checkpoint=?", (f["id"], cp))}
            missing = [s for s in ("flyrely", "flighty") if s not in have]
            if missing:
                print(f"#{f['id']} {f['carrier']}{f['number']} {f['origin']}->{f['dest']} {cp}: need {', '.join(missing)}")


def cmd_flighty(a):
    if a.ontime is None and a.delay is None:
        sys.exit("pass --ontime <pct shown> or --delay <minutes shown, 0 if on time>")
    con = db()
    p = None if a.ontime is None else 1 - a.ontime / 100
    con.execute("insert or replace into predictions (flight_id, checkpoint, source, logged_utc, p_delay, "
                "delay_minutes, ontime_pct) values (?,?,?,?,?,?,?)",
                (a.id, a.checkpoint, "flighty", now().isoformat(), p, a.delay, a.ontime))
    con.commit()
    print(f"#{a.id} {a.checkpoint}: Flighty logged")


def cmd_outcome(a):
    if a.delay is None and not a.cancelled:
        sys.exit("pass --delay <minutes> or --cancelled")
    con = db()
    con.execute("update flights set actual_delay=?, cancelled=?, outcome_source='manual' where id=?",
                (a.delay, int(a.cancelled), a.id))
    con.commit()
    print(f"#{a.id}: outcome recorded")


def cmd_outcomes_from_bts(a):
    import pandas as pd
    con = db()
    open_ = con.execute("select * from flights where actual_delay is null and cancelled is null").fetchall()
    filled = 0
    for f in open_:
        dep = datetime.fromisoformat(f["sched_dep_local"])
        p = ROOT / "data" / "bts" / "parquet" / f"{dep:%Y-%m}.parquet"
        if not p.exists():
            continue
        # BTS reports by operating carrier, so match the operator's flight
        df = pd.read_parquet(p, columns=["FlightDate", "Reporting_Airline", "Flight_Number_Reporting_Airline",
                                         "Origin", "DepDelay", "Cancelled"])
        m = df[(df["FlightDate"] == pd.Timestamp(dep.date())) & (df["Origin"].astype(str) == f["origin"])
               & (df["Flight_Number_Reporting_Airline"] == int(f["operating_number"]))
               & (df["Reporting_Airline"].astype(str) == f["operating_carrier"])]
        if len(m) == 1:
            r = m.iloc[0]
            con.execute("update flights set actual_delay=?, cancelled=?, outcome_source='bts' where id=?",
                        (None if r["Cancelled"] else float(r["DepDelay"]), int(r["Cancelled"]), f["id"]))
            filled += 1
    con.commit()
    print(f"filled {filled} of {len(open_)} open flights from BTS")


def cmd_score(a):
    con = db()
    rows = con.execute("""
        select p.checkpoint, f.id, f.actual_delay, f.cancelled,
               max(case when p.source='flyrely' then p.p_delay end) as fr_p,
               max(case when p.source='flighty' then p.p_delay end) as fl_p,
               max(case when p.source='flighty' then p.delay_minutes end) as fl_min
        from predictions p join flights f on f.id = p.flight_id
        where f.cancelled = 0 and f.actual_delay is not null
        group by p.checkpoint, f.id""").fetchall()
    rng = np.random.default_rng(42)
    print(f"{'checkpoint':<10}{'n':>5}  {'FlyRely Brier':>14}{'Flighty Brier':>14}  {'95% CI of diff':>20}"
          f"  {'FlyRely call acc':>17}{'Flighty call acc':>17}")
    for cp in CHECKPOINTS:
        r = [x for x in rows if x["checkpoint"] == cp and x["fr_p"] is not None]
        if not r:
            continue
        y = np.array([x["actual_delay"] >= 15 for x in r], dtype=float)
        fr = np.array([x["fr_p"] for x in r])
        # Flighty: its probability if it showed one, else its delay call (>=15 min shown = delayed)
        fl = np.array([x["fl_p"] if x["fl_p"] is not None else
                       (np.nan if x["fl_min"] is None else float(x["fl_min"] >= 15)) for x in r])
        ok = ~np.isnan(fl)
        n = int(ok.sum())
        line = f"{cp:<10}{n:>5}  "
        if n:
            d = (fr[ok] - y[ok]) ** 2 - (fl[ok] - y[ok]) ** 2
            boot = d[rng.integers(0, n, size=(2000, n))].mean(1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            line += (f"{((fr[ok] - y[ok]) ** 2).mean():>14.4f}{((fl[ok] - y[ok]) ** 2).mean():>14.4f}"
                     f"  {f'[{lo:+.4f}, {hi:+.4f}]':>20}"
                     f"  {((fr[ok] >= 0.5) == y[ok]).mean():>17.1%}{((fl[ok] >= 0.5) == y[ok]).mean():>17.1%}")
        print(line)
    print("Brier: lower is better. A CI entirely below zero means FlyRely is more accurate at that checkpoint.")


def main_cli():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("add"); s.add_argument("flight"); s.add_argument("departure", help="local, e.g. 2026-11-20T08:30")
    s.add_argument("origin"); s.add_argument("dest"); s.add_argument("--tz")
    sub.add_parser("due"); sub.add_parser("snapshot"); sub.add_parser("score"); sub.add_parser("outcomes-from-bts")
    s = sub.add_parser("flighty"); s.add_argument("id", type=int); s.add_argument("checkpoint", choices=list(CHECKPOINTS))
    s.add_argument("--ontime", type=float, help="on-time %% Flighty shows"); s.add_argument("--delay", type=float)
    s = sub.add_parser("outcome"); s.add_argument("id", type=int); s.add_argument("--delay", type=float)
    s.add_argument("--cancelled", action="store_true")
    a = ap.parse_args()
    {"add": cmd_add, "due": cmd_due, "snapshot": cmd_snapshot, "flighty": cmd_flighty, "outcome": cmd_outcome,
     "outcomes-from-bts": cmd_outcomes_from_bts, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main_cli()
