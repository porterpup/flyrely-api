#!/usr/bin/env python3
"""A5b: high-frequency hub-box polling of OpenSky /states/all (protocol: backtest/REPORT_A5B_OPENSKY_HUBS.md).

  python backtest/opensky_hubs.py poll       # background-safe poller; stops at the credit cap/floor; refuses to double-start
  python backtest/opensky_hubs.py flights    # optional /flights/departure comparison (30 credits each, capped)
  python backtest/opensky_hubs.py analyze    # no network; writes backtest/results/a5b_opensky_hubs.json
"""
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import opensky_client as oc  # noqa: E402

ROOT = oc.ROOT
HUBS = ["ATL", "ORD", "DFW", "DEN", "CLT", "LGA", "MSP", "DTW", "SEA", "MDW", "BNA", "RDU"]
INTERVAL_S = 120
HALF_LAT, HALF_LON = 0.5, 0.6          # ~1.2 sq deg (<= 25 sq deg => 1 credit)
CONUS_EVERY = 5                         # one CONUS call (4 credits) every 5th round, to compare with boxes
STATES_BEFORE_PHASE = 165               # /states credits spent before this phase (A5 ledger)
STATES_CAP = STATES_BEFORE_PHASE + 1500  # this phase may spend at most 1,500
CONUS = {"lamin": 24, "lamax": 50, "lomin": -125, "lomax": -66}
# (start ISO UTC or None for 'now', number of 2-minute rounds)
BLOCKS = [(None, 38), ("2026-11-04T15:00:00+00:00", 74)]
PIDFILE = oc.CACHE_DIR / "hubs.pid"
FLIGHTS_PHASE_CAP = 2250 + 120          # /flights credits: 4 optional calls only


def hub_coords():
    import pandas as pd
    a = pd.read_csv(ROOT / "data" / "ref" / "airports.csv", dtype=str)
    a = a[a.iata_code.isin(HUBS) & (a["type"] == "large_airport") | a.iata_code.isin(HUBS) & (a["type"] == "medium_airport")]
    a = a.drop_duplicates("iata_code").set_index("iata_code")
    return {h: (float(a.loc[h, "latitude_deg"]), float(a.loc[h, "longitude_deg"]), float(a.loc[h, "elevation_ft"]) * 0.3048) for h in HUBS}


def box(lat, lon):
    return {"lamin": round(lat - HALF_LAT, 3), "lamax": round(lat + HALF_LAT, 3),
            "lomin": round(lon - HALF_LON, 3), "lomax": round(lon + HALF_LON, 3)}


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def cmd_poll():
    if PIDFILE.exists():
        try:
            if _alive(int(PIDFILE.read_text())):
                print("poller already running; refusing to double-poll")
                return
        except ValueError:
            pass
    PIDFILE.write_text(str(os.getpid()))
    try:
        _poll_loop()
    finally:
        PIDFILE.unlink(missing_ok=True)


def _poll_loop():
    c = oc.OpenSky()
    coords = hub_coords()
    rnd = 0
    for start, n in BLOCKS:
        t0 = time.time() if start is None else dt.datetime.fromisoformat(start).timestamp()
        while time.time() < t0 - 1:
            time.sleep(min(30, t0 - time.time()))
        for k in range(n):
            tr = t0 + k * INTERVAL_S
            if time.time() > tr + INTERVAL_S:   # fell behind (e.g. machine slept): skip, never burst
                continue
            while time.time() < tr:
                time.sleep(0.5)
            rnd += 1
            try:
                for h in HUBS:
                    lat, lon, _ = coords[h]
                    c.get("/states/all", box(lat, lon), 1, cache_tag=f"hub-{int(tr)}-{h}", group_cap=STATES_CAP)
                if rnd % CONUS_EVERY == 0:
                    c.get("/states/all", CONUS, 4, cache_tag=f"hubconus-{int(tr)}", group_cap=STATES_CAP)
            except oc.BudgetStop as ex:
                print("STOP (budget):", ex, flush=True)
                return
            except Exception as ex:   # transient error: log and carry on at the next round, never tight-loop
                print("round error:", type(ex).__name__, flush=True)
                continue
            print(f"round {rnd} t={int(tr)} remaining={c.remaining} states_spent={oc.credits_spent('/states')}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    if cmd == "poll":
        cmd_poll()


# =========================================================================== flights comparison (optional, capped)
FLIGHTS_HUBS = ["ATL", "LGA", "CLT", "ORD"]


def cmd_flights():
    """near-live /flights/departure for the daytime window (30 credits each, 4 calls max, /flights floor 1,500)."""
    c = oc.OpenSky()
    b0, b1 = day_window()
    coords = hub_coords()
    for h in FLIGHTS_HUBS:
        icao = "K" + h
        try:
            body, cached = c.get("/flights/departure", {"airport": icao, "begin": int(b0), "end": int(b1)}, 30, group_cap=FLIGHTS_PHASE_CAP)
        except oc.BudgetStop as ex:
            print("STOP (budget):", ex)
            return
        print(h, "cached" if cached else "fetched", len(body), "remaining", c.remaining)


def day_window():
    """UTC span of the daytime block (from cached hub snapshots)."""
    ts = sorted({int(r["req"].split("hub-")[1].split("-")[0]) for r in oc.ledger_rows() if "#hub-" in r.get("req", "")})
    day = [t for t in ts if t >= dt.datetime.fromisoformat("2026-11-04T14:00:00+00:00").timestamp()]
    return (day[0], day[-1] + 600) if day else (None, None)


# =========================================================================== analysis
import re  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import opensky_coverage as oco  # noqa: E402

R_DEP, R_ARR_AIR, R_ARR_GND = 15.0, 25.0, 5.0
AGL_DEP, AGL_ARR, MAX_GAP = 1200.0, 600.0, 600.0
AL_RE = re.compile(r"^[A-Z]{3}\d{1,4}$")   # protocol: 3-letter ICAO carrier code + 1 to 4 digits


def load_hub_snapshots():
    """Long DataFrame of airline-callsign state vectors per hub box snapshot, plus per-round CONUS snapshots."""
    rows, conus = [], {}
    for r in oc.ledger_rows():
        req = r.get("req", "")
        if r.get("status") != 200 or not req.startswith("/states/all") or ("#hub-" not in req and "#hubconus-" not in req):
            continue
        tag = req.split("#", 1)[1]
        if tag.startswith("hubconus-"):
            key, _ = oc.cache_key("/states/all", CONUS, tag)
            f = oc.CACHE_DIR / f"{key}.json"
            if f.exists():
                conus[int(tag.split("-")[1])] = json.loads(f.read_text())["body"]
            continue
        _, tr, hub = tag.split("-")
        lat, lon, _e = HUB_COORDS[hub]
        key, _ = oc.cache_key("/states/all", box(lat, lon), tag)
        f = oc.CACHE_DIR / f"{key}.json"
        if not f.exists():
            continue
        body = json.loads(f.read_text())["body"]
        t = body["time"]
        for x in body.get("states") or []:
            cs = oco.norm_callsign(x[1])
            if cs and AL_RE.match(cs) and cs[:3] in oco.ICAO_TO_IATA and x[5] is not None and x[6] is not None:
                rows.append((hub, int(tr), t, x[0], cs, x[3] if x[3] else t, x[5], x[6], x[7], x[13], bool(x[8]), x[11]))
    d = pd.DataFrame(rows, columns=["hub", "rt", "t", "icao24", "cs", "te", "lon", "lat", "balt", "galt", "ground", "vr"])
    return d, conus


HUB_COORDS = {}


def detect_events(d, hub, airports):
    """Departure/arrival events at `hub` from its box snapshots (see PROTOCOL for definitions)."""
    lat0, lon0, elev0 = HUB_COORDS[hub]
    g = d[d.hub == hub].sort_values(["icao24", "te"]).reset_index(drop=True)
    if g.empty:
        return pd.DataFrame(columns=["kind", "icao24", "cs", "t", "bracket_s", "bracketed"])
    la, lo = g.lat.values, g.lon.values
    dist = oco.haversine_km(la, lo, lat0, lon0)
    # nearest other airport (cross-talk guard: JFK/EWR near LGA, MDW near ORD)
    other_d = np.full(len(g), 1e9)
    for (alat, alon, _e) in airports:
        if abs(alat - lat0) < 0.01 and abs(alon - lon0) < 0.01:
            continue
        other_d = np.minimum(other_d, oco.haversine_km(la, lo, alat, alon))
    mine = dist <= other_d
    alt = np.where(pd.notna(g.galt.values), g.galt.values, g.balt.values).astype(float)
    agl = alt - elev0
    ev = []
    ic = g.icao24.values
    te = g.te.values.astype(float)
    ground = g.ground.values
    vr = g.vr.values.astype(float)
    cs = g.cs.values
    last_ev = {}
    for i in range(len(g)):
        if not mine[i]:
            continue
        has_prev = i > 0 and ic[i - 1] == ic[i] and te[i] - te[i - 1] <= MAX_GAP
        # ---- departure
        if (not ground[i]) and dist[i] <= R_DEP and not np.isnan(agl[i]) and agl[i] <= AGL_DEP:
            if has_prev and ground[i - 1] and dist[i - 1] <= R_ARR_GND:
                lift = te[i] - min(max(agl[i], 0) / vr[i], 120) if vr[i] > 1 else te[i]
                ev.append(("dep", ic[i], cs[i], max(lift, te[i - 1]), te[i] - te[i - 1], True))
            elif (not has_prev) and vr[i] > 2:
                ev.append(("dep", ic[i], cs[i], te[i] - min(max(agl[i], 0) / vr[i], 120), np.nan, False))
        # ---- arrival (bracketed: airborne -> ground)
        if ground[i] and dist[i] <= R_ARR_GND and has_prev and (not ground[i - 1]) and dist[i - 1] <= R_ARR_AIR:
            ev.append(("arr", ic[i], cs[i], te[i], te[i] - te[i - 1], True))
        # ---- arrival (unbracketed: last sighting low and descending near the field)
        last_of_track = (i == len(g) - 1) or ic[i + 1] != ic[i] or te[i + 1] - te[i] > MAX_GAP
        if (not ground[i]) and last_of_track and dist[i] <= 12 and vr[i] < -1 and not np.isnan(agl[i]) and agl[i] < AGL_ARR:
            ev.append(("arr", ic[i], cs[i], te[i] + min(max(agl[i], 0) / abs(vr[i]), 120), np.nan, False))
    e = pd.DataFrame(ev, columns=["kind", "icao24", "cs", "t", "bracket_s", "bracketed"])
    if e.empty:
        return e
    e = e.sort_values(["kind", "icao24", "t"])
    keep = []
    prev = {}
    for k, r in zip(e.index, e.itertuples()):   # one event per aircraft per kind within 30 minutes
        key = (r.kind, r.icao24)
        if key in prev and r.t - prev[key] < 1800:
            continue
        prev[key] = r.t
        keep.append(k)
    return e.loc[keep].assign(hub=hub)


def blocks_from(snap_times, gap=3600):
    ts = sorted(set(int(x) for x in snap_times))
    out, cur = [], [ts[0]]
    for t in ts[1:]:
        if t - cur[-1] > gap:
            out.append((cur[0], cur[-1]))
            cur = []
        cur.append(t)
    out.append((cur[0], cur[-1]))
    return out


def expected_and_recurring(ref):
    full, ok = oco.bts_schedule_for_nov(ref)
    fl = oco.retime(full, ref)
    return full, fl


def ratio_for(events, fl, blocks_iv, hubs, step_label, numerator="all", sched_any=None):
    """Aggregate detected/expected per hub, for the given (already thinned) events. blocks_iv = interior windows.
    numerator: "all" = every airline-pattern callsign (protocol-literal, primary);
               "any_schedule" = only callsigns in sched_any (every August BTS flight, any day and airport) (drops international/charter/ferry);
               "hub_schedule" = only callsigns BTS lists for this hub and direction (strictest; a floor)."""
    sched_all = sched_any if sched_any is not None else set(fl.cs_op) | set(fl.cs_mk)
    out = {}
    for h in hubs:
        for kind, col_h, col_t in (("dep", "Origin", "woff"), ("arr", "Dest", "won")):
            sub = fl[fl[col_h] == h]
            for g, name in ((None, "all"), ("mainline", "mainline"), ("regional", "regional")):
                ss = sub if g is None else sub[sub.op.isin(oco.MAINLINE) == (g == "mainline")]
                ee = events[(events.hub == h) & (events.kind == kind)]
                if numerator == "any_schedule":
                    ee = ee[ee.cs.isin(sched_all)]
                elif numerator == "hub_schedule":
                    ee = ee[ee.cs.isin(set(ss.cs_op) | set(ss.cs_mk))]
                if g is not None:
                    ee = ee[ee.cs.str[:3].map(oco.ICAO_TO_IATA).isin(oco.MAINLINE) == (g == "mainline")]
                for bi, (w0, w1) in enumerate(blocks_iv):
                    exp = ss[(ss[col_t] >= w0) & (ss[col_t] <= w1)]
                    # expected count = flights per Aug day: rows / n_src per L (each L has its own n_src)
                    n_exp = sum(len(gg) / gg.n_src.iloc[0] for _, gg in exp.groupby("L")) if len(exp) else 0.0
                    n_det = int(((ee.t >= w0) & (ee.t <= w1)).sum())
                    out.setdefault(h, {}).setdefault(kind, {}).setdefault(name, {})[f"block{bi}"] = {"detected": n_det, "expected": round(float(n_exp), 1)}
    return out


def summarize_ratio(raw):
    """raw[h][kind][group][block] -> ratios, plus totals across hubs."""
    res = {"by_hub": {}, "total": {}}
    for h, kinds in raw.items():
        for kind, groups in kinds.items():
            for g, blocks in groups.items():
                for b, v in blocks.items():
                    res["by_hub"].setdefault(h, {}).setdefault(kind, {}).setdefault(g, {})[b] = {**v, "ratio": (v["detected"] / v["expected"]) if v["expected"] else None}
                    t = res["total"].setdefault(kind, {}).setdefault(g, {}).setdefault(b, {"detected": 0, "expected": 0.0})
                    t["detected"] += v["detected"]
                    t["expected"] += v["expected"]
    for kind in res["total"]:
        for g in res["total"][kind]:
            for b, t in res["total"][kind][g].items():
                t["ratio"] = t["detected"] / t["expected"] if t["expected"] else None
                t["expected"] = round(t["expected"], 1)
    return res


def cost_table():
    t = {}
    for m in (1, 2, 5, 10):
        t[f"every_{m}min"] = {"credits_per_hub_24h": int(1440 / m), "credits_per_hub_16h_daytime": int(960 / m),
                              "hubs_fitting_in_4000_24h": int(4000 // (1440 / m)), "hubs_fitting_in_4000_16h": int(4000 // (960 / m)),
                              "conus_one_call_24h": int(1440 / m * 4), "conus_one_call_16h": int(960 / m * 4),
                              "boxes_cheaper_than_one_conus_call_when_hubs_below": 4}
    return t


def classify_hub(dep_ratio, arr_ratio, dep_bracket, arr_bracket, min_bracket_n=10):
    """Pre-registered rule, daytime block, 2-minute polling. dep_bracket/arr_bracket = {"n", "median"} (seconds) or None.
    reliable: both ratios >= 0.85 and median bracket <= 180 s in both directions (a median from fewer than min_bracket_n
    bracketed events, or none, cannot be verified and does not pass); not usable: either ratio < 0.7; otherwise partial."""
    if dep_ratio is None or arr_ratio is None:
        return "no data"
    if min(dep_ratio, arr_ratio) < 0.7:
        return "not usable"
    br_ok = all(b is not None and b["n"] >= min_bracket_n and b["median"] <= 180 for b in (dep_bracket, arr_bracket))
    return "reliable" if (min(dep_ratio, arr_ratio) >= 0.85 and br_ok) else "partial"


def analyze():
    global HUB_COORDS
    HUB_COORDS = hub_coords()
    d, conus = load_hub_snapshots()
    if d.empty:
        raise SystemExit("no hub snapshots cached")
    ref = oco.airport_ref(set(pd.read_csv(ROOT / "data" / "ref" / "airports.csv", dtype=str, usecols=["iata_code"]).iata_code.dropna()))
    a = pd.read_csv(ROOT / "data" / "ref" / "airports.csv", dtype=str)
    a = a[a["type"].isin(["large_airport", "medium_airport"]) & (a.scheduled_service == "yes") & a.iata_code.notna()]
    airports = [(float(r.latitude_deg), float(r.longitude_deg), float(r.elevation_ft) * .3048 if pd.notna(r.elevation_ft) else 0.0) for r in a.itertuples()]
    full, fl = expected_and_recurring(ref)
    known_any = set(full.cs_op) | set(full.cs_mk)   # callsigns of every August BTS flight, any day and airport
    known_all = set(fl.cs_op) | set(fl.cs_mk)   # used by the optional /flights comparison

    blocks = blocks_from(d.t)
    interior = [(a_ + 1800, b_ - 1800) for a_, b_ in blocks if b_ - a_ > 3600]
    res = {"hubs": HUBS, "blocks_utc": [[pd.Timestamp(a_, unit="s").isoformat(), pd.Timestamp(b_, unit="s").isoformat()] for a_, b_ in blocks],
           "interior_windows_utc": [[pd.Timestamp(a_, unit="s").isoformat(), pd.Timestamp(b_, unit="s").isoformat()] for a_, b_ in interior],
           "rounds_per_block": [int(d[(d.t >= a_) & (d.t <= b_)].rt.nunique()) for a_, b_ in blocks],
           "airline_state_rows": int(len(d))}
    # ---- events at full cadence and thinned cadences
    rt_rank = {}
    for bi, (a_, b_) in enumerate(blocks):
        rts = sorted(d[(d.t >= a_) & (d.t <= b_)].rt.unique())
        for i, r in enumerate(rts):
            rt_rank[r] = i
    d["rk"] = d.rt.map(rt_rank)
    res["cadence_sensitivity"] = {}
    events_by_step = {}
    for step in (1, 2, 3):
        dd = d[d.rk % step == 0]
        ev = pd.concat([detect_events(dd, h, airports) for h in HUBS], ignore_index=True)
        events_by_step[step] = ev
        raw = ratio_for(ev, fl, interior, HUBS, f"step{step}")
        res["cadence_sensitivity"][f"{2 * step}min"] = summarize_ratio(raw)["total"]
    ev = events_by_step[1]
    sm = summarize_ratio(ratio_for(ev, fl, interior, HUBS, "step1"))
    res["detection_ratio_2min"] = sm
    res["events_total"] = {k: int(v) for k, v in ev.kind.value_counts().items()}
    # ---- timing (bracket widths), overall and per hub, for all windows and for the daytime block only
    br = ev[ev.bracketed]
    day0, day1 = interior[-1]   # same trimmed interior daytime window as the ratios

    def bw(x):
        return {"n": int(len(x)), "median": float(x.bracket_s.median()), "p90": float(x.bracket_s.quantile(.9))} if len(x) else None

    res["timing"] = {
        "share_bracketed": {k: float(ev[ev.kind == k].bracketed.mean()) for k in ("dep", "arr") if (ev.kind == k).any()},
        "bracket_width_s": {k: bw(br[br.kind == k]) for k in ("dep", "arr")},
        "bracket_width_s_daytime_interior": {k: bw(br[(br.kind == k) & br.t.between(day0, day1)]) for k in ("dep", "arr")},
        "bracket_width_s_by_hub_daytime_interior": {h: {k: bw(br[(br.hub == h) & (br.kind == k) & br.t.between(day0, day1)]) for k in ("dep", "arr")} for h in HUBS},
        "bracket_width_s_by_hub_all": {h: {k: bw(br[(br.hub == h) & (br.kind == k)]) for k in ("dep", "arr")} for h in HUBS}}
    # ---- primary metric per hub and per hour (interior windows cut into hourly bins)
    hourly, hlabels = [], []
    for wi, (w0, w1) in enumerate(interior):
        t0 = w0
        while t0 < w1:
            hourly.append((t0, min(t0 + 3600, w1)))
            hlabels.append(pd.Timestamp(t0, unit="s").isoformat() + "Z")
            t0 += 3600
    hr = summarize_ratio(ratio_for(ev, fl, hourly, HUBS, "hourly"))["by_hub"]
    res["detection_ratio_by_hour"] = {"bins_utc": hlabels, "by_hub": hr}
    # ---- sensitivity: numerator restricted to callsigns BTS lists for that hub and direction (a lower bound: identity-level)
    res["detection_ratio_2min_schedule_callsign_only"] = summarize_ratio(ratio_for(ev, fl, interior, HUBS, "step1", numerator="hub_schedule"))
    res["detection_ratio_2min_any_schedule_callsign"] = summarize_ratio(ratio_for(ev, fl, interior, HUBS, "step1", numerator="any_schedule", sched_any=known_any))
    dwin = ev[ev.t.between(*interior[-1])]
    res["daytime_interior_events_with_callsign_in_no_august_schedule"] = {
        "events": int(len(dwin)), "not_in_any_schedule": int((~dwin.cs.isin(known_any)).sum()),
        "by_hub": {h: [int((~dwin[dwin.hub == h].cs.isin(known_any)).sum()), int((dwin.hub == h).sum())] for h in HUBS}}
    # ---- diagnostic: why departures fall at 4-minute spacing (departures found at 2 min but not at 4 min)
    e2 = events_by_step[2]
    d2 = d[d.rk % 2 == 0].sort_values(["icao24", "te"])
    n_lost = n_pair = 0
    for r in ev[(ev.kind == "dep") & ev.t.between(*interior[-1])].itertuples():
        if len(e2[(e2.hub == r.hub) & (e2.kind == "dep") & (e2.icao24 == r.icao24) & ((e2.t - r.t).abs() < 1800)]):
            continue
        n_lost += 1
        g = d2[(d2.hub == r.hub) & (d2.icao24 == r.icao24) & ((d2.te - r.t).abs() < 1800)]
        gv, tv = g.ground.values, g.te.values
        if any(gv[i] and not gv[i + 1] and tv[i + 1] - tv[i] <= MAX_GAP for i in range(len(g) - 1)):
            n_pair += 1
    res["departure_cadence_diagnostic_2_to_4min"] = {"departures_lost": n_lost, "of_which_ground_then_airborne_pair_within_600s_in_thinned_data": n_pair}
    # ---- identity-level (recurring flights, secondary metric), tails, takeoff-timing plausibility
    fl["sarr"] = fl.won - fl.ArrDelay.fillna(0) * 60      # approx. scheduled arrival (wheels-on minus arrival delay; ignores taxi-in)
    key = ["op", "cs_op", "cs_mk", "Origin", "Dest", "dep_min", "L"]
    rec = fl.groupby(key).agg(n=("sdep", "size"), sdep=("sdep", "median"), sarr=("sarr", "median"), won=("won", "median"), woff=("woff", "median"),
                              n_src=("n_src", "first"),
                              tail=("Tail_Number", lambda s: s.mode().iloc[0] if s.notna().any() else None),
                              tail_share=("Tail_Number", lambda s: s.value_counts(normalize=True).iloc[0] if s.notna().any() else np.nan)).reset_index()
    rec = rec[rec.n >= 0.75 * rec.n_src]
    rec["icao24_pred"] = rec["tail"].map(oco.n_number_to_icao24)
    ident, lag_rows = {}, []
    for h in HUBS:
        for kind, col, tcol, scol in (("dep", "Origin", "woff", "sdep"), ("arr", "Dest", "won", "sarr")):
            e = ev[(ev.hub == h) & (ev.kind == kind)]
            for bi, (w0, w1) in enumerate(interior):
                r = rec[(rec[col] == h) & rec[tcol].between(w0, w1)]    # flights whose expected event falls in the interior window
                if r.empty:
                    continue
                hit, same_tail = 0, 0
                for rr in r.itertuples():
                    sch = getattr(rr, scol)
                    cand = e[e.cs.isin([rr.cs_op, rr.cs_mk]) & ((e.t - sch).abs() <= 3600)]   # within 60 min of SCHEDULED time
                    if len(cand):
                        hit += 1
                        best = cand.iloc[(cand.t - sch).abs().argmin()]
                        same_tail += int(rr.icao24_pred is not None and best.icao24 == rr.icao24_pred)
                        if kind == "dep":
                            lag_rows.append((h, bi, float(best.t - sch), float(getattr(rr, tcol) - sch)))
                ident.setdefault(h, {}).setdefault(kind, {})[f"block{bi}"] = {
                    "recurring_flights": int(len(r)), "matched_by_callsign_within_60min_of_scheduled": hit, "rate": hit / len(r),
                    "modal_tail_equals_observed_icao24": (same_tail / hit if hit else None),
                    "median_tail_stability_in_august": float(r.tail_share.median())}
    res["identity_level_lower_bound"] = ident
    lg = pd.DataFrame(lag_rows, columns=["hub", "block", "state_takeoff_minus_sched_s", "aug_wheelsoff_minus_sched_s"])

    def q(x):
        return {"n": int(len(x)), "p10": float(x.quantile(.1)), "p50": float(x.quantile(.5)), "p90": float(x.quantile(.9))} if len(x) else None

    res["takeoff_vs_schedule_plausibility"] = {
        "note": "matched recurring departures only; August column = wheels-off minus scheduled for the same matched flights (seconds)",
        "by_block": {f"block{b}": {"detected": q(lg[lg.block == b].state_takeoff_minus_sched_s), "august_bts": q(lg[lg.block == b].aug_wheelsoff_minus_sched_s)} for b in sorted(lg.block.unique())},
        "by_hub_median_s": {h: {"detected": float(lg[lg.hub == h].state_takeoff_minus_sched_s.median()), "august_bts": float(lg[lg.hub == h].aug_wheelsoff_minus_sched_s.median()), "n": int((lg.hub == h).sum())}
                            for h in HUBS if (lg.hub == h).any()}}
    # ---- observed turnarounds: departure whose aircraft arrived at same hub earlier in the block
    ta = {}
    for h in HUBS:
        dep = ev[(ev.hub == h) & (ev.kind == "dep")]
        arr = ev[(ev.hub == h) & (ev.kind == "arr")]
        if len(dep):
            ta[h] = float(np.mean([(arr[(arr.icao24 == r.icao24) & (arr.t < r.t) & (r.t - arr.t < 6 * 3600)].shape[0] > 0) for r in dep.itertuples()]))
    res["observed_turnaround_share_of_detected_departures"] = ta
    # ---- CONUS vs box equivalence
    eq = []
    for rt, body in conus.items():
        st = body.get("states") or []
        for h in HUBS:
            lat, lon, _ = HUB_COORDS[h]
            b = box(lat, lon)
            in_conus = {x[0] for x in st if x[6] is not None and b["lamin"] <= x[6] <= b["lamax"] and b["lomin"] <= x[5] <= b["lomax"]}
            key_ = oc.cache_key("/states/all", b, f"hub-{rt}-{h}")[0]
            f = oc.CACHE_DIR / f"{key_}.json"
            if f.exists():
                bx = {x[0] for x in (json.loads(f.read_text())["body"].get("states") or [])}
                if bx or in_conus:
                    eq.append((h, len(bx), len(in_conus), len(bx & in_conus) / max(1, len(bx | in_conus))))
    if eq:
        e2 = pd.DataFrame(eq, columns=["hub", "box_n", "conus_n", "jaccard"])
        res["box_vs_conus"] = {"comparisons": int(len(e2)), "mean_box_count": float(e2.box_n.mean()), "mean_conus_count_in_box": float(e2.conus_n.mean()),
                               "mean_jaccard_icao24": float(e2.jaccard.mean())}
    # ---- /flights/departure comparison (optional)
    fcmp = {}
    b0, b1 = day_window()
    for h in FLIGHTS_HUBS:
        key_ = oc.cache_key("/flights/departure", {"airport": "K" + h, "begin": int(b0 or 0), "end": int(b1 or 0)})[0]
        f = oc.CACHE_DIR / f"{key_}.json"
        if not f.exists() or b0 is None:
            continue
        body = json.loads(f.read_text())["body"]
        fd = pd.DataFrame(body)
        if fd.empty:
            fcmp[h] = {"flights_records": 0}
            continue
        fd["cs"] = fd.callsign.map(oco.norm_callsign)
        fd = fd[fd.cs.isin(known_all) & fd.firstSeen.between(interior[-1][0], interior[-1][1])]
        e = ev[(ev.hub == h) & (ev.kind == "dep")]
        diffs, found = [], 0
        for r in fd.itertuples():
            c = e[(e.icao24 == r.icao24) & ((e.t - r.firstSeen).abs() <= 1800)]
            if len(c):
                found += 1
                diffs.append(float(c.iloc[((c.t - r.firstSeen).abs()).argmin()].t - r.firstSeen))
        n_state = int(((e.t >= interior[-1][0]) & (e.t <= interior[-1][1])).sum())
        in_fl = set(fd.icao24)
        fcmp[h] = {"flights_records_known_callsigns": int(len(fd)), "state_detected_departures": n_state,
                   "flights_records_also_found_by_states": found, "share_of_flights_records_found_by_states": found / len(fd) if len(fd) else None,
                   "share_of_state_departures_present_in_flights": (len({x for x in e[(e.t >= interior[-1][0]) & (e.t <= interior[-1][1])].icao24} & in_fl) /
                                                                     max(1, e[(e.t >= interior[-1][0]) & (e.t <= interior[-1][1])].icao24.nunique())),
                   "takeoff_state_minus_flights_firstSeen_s": ({"median": float(np.median(diffs)), "p90_abs": float(np.percentile(np.abs(diffs), 90))} if diffs else None)}
    if fcmp:
        res["flights_departure_comparison"] = fcmp
    res["credits"] = {"states_spent_total": oc.credits_spent("/states"), "states_spent_this_phase": oc.credits_spent("/states") - STATES_BEFORE_PHASE,
                      "flights_spent_total": oc.credits_spent("/flights"), "states_remaining": oc.last_remaining("/states"),
                      "flights_remaining": oc.last_remaining("/flights")}
    res["cost_table"] = cost_table()
    last = f"block{len(interior) - 1}"
    cls = {}
    for h in HUBS:
        bt = res["timing"]["bracket_width_s_by_hub_daytime_interior"][h]
        cls[h] = {"dep_bracket_daytime": bt["dep"], "arr_bracket_daytime": bt["arr"]}
        for label, key in (("primary", "detection_ratio_2min"), ("middle_any_schedule", "detection_ratio_2min_any_schedule_callsign"),
                           ("strict_hub_schedule", "detection_ratio_2min_schedule_callsign_only")):
            bh = res[key]["by_hub"][h]
            dr, ar = bh["dep"]["all"][last]["ratio"], bh["arr"]["all"][last]["ratio"]
            cls[h][label] = {"dep_ratio_daytime": dr, "arr_ratio_daytime": ar, "class": classify_hub(dr, ar, bt["dep"], bt["arr"])}
        cls[h]["class"] = cls[h]["primary"]["class"]
    res["classification"] = cls
    out = ROOT / "backtest" / "results" / "a5b_opensky_hubs.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: res[k] for k in ("blocks_utc", "rounds_per_block", "events_total", "timing", "credits")}, indent=1, default=float))
    print(json.dumps(res["detection_ratio_2min"]["total"], indent=1, default=float)[:2500])
    return res


if __name__ == "__main__" and sys.argv[1:2] in (["analyze"], ["flights"]):
    {"analyze": analyze, "flights": cmd_flights}[sys.argv[1]]()
