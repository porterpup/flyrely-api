#!/usr/bin/env python3
"""A5: how faithfully does the free OpenSky REST API reproduce BTS actuals? (2026-08 only; holdout >= 2026-09 untouched)

Budgeted, cached and resumable:
  python backtest/opensky_coverage.py fetch     # pull /flights/all 2h windows (30 credits each), stops at the budget floor
  python backtest/opensky_coverage.py states    # one CONUS /states/all probe (4 credits)
  python backtest/opensky_coverage.py analyze   # no network; writes backtest/results/a5_opensky_coverage.json
Raw responses are cached in data/opensky_cache/ (gitignored); reruns cost nothing.
"""
import datetime as dt
import json
import sys
import zipfile
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import opensky_client as oc  # noqa: E402

ROOT = oc.ROOT
RESULT = ROOT / "backtest" / "results" / "a5_opensky_coverage.json"
MONTH = "2026-08"
# UTC days sampled: a mix of weekdays/weekends. 12 two-hour windows per day = 360 credits/day.
SAMPLE_DAYS = ["2026-08-04", "2026-08-08", "2026-08-12", "2026-08-16", "2026-08-20", "2026-08-25"]
WINDOW_S = 7200
FLIGHTS_COST = 30   # observed: every /flights/* call costs 30 credits (also for <24h-old data)

# --------------------------------------------------------------------------- pure helpers
IATA_TO_ICAO = {
    "AA": "AAL", "DL": "DAL", "UA": "UAL", "WN": "SWA", "AS": "ASA", "B6": "JBU", "F9": "FFT", "G4": "AAY",
    "NK": "NKS", "HA": "HAL", "9E": "EDV", "OO": "SKW", "YX": "RPA", "MQ": "ENY", "OH": "JIA", "YV": "ASH",
    "PT": "PDT", "QX": "QXE", "G7": "GJS", "C5": "UCA",
}
REGIONALS = ["9E", "OO", "YX", "MQ", "OH", "YV", "PT", "QX", "G7", "C5"]


def callsign_candidates(operating_iata, op_flight_num, marketing_iata=None, mkt_flight_num=None):
    """Plausible ADS-B callsigns for a flight: operating carrier + its number, marketing carrier + marketing number."""
    out = []
    for iata, num in ((operating_iata, op_flight_num), (marketing_iata, mkt_flight_num)):
        icao = IATA_TO_ICAO.get(str(iata))
        if icao is None or num is None or pd.isna(num):
            continue
        cs = f"{icao}{int(num)}"
        if cs not in out:
            out.append(cs)
    return out


def norm_callsign(cs):
    return None if cs is None or (isinstance(cs, float) and np.isnan(cs)) else str(cs).strip().upper() or None


_CHARSET = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # 24 letters, no I or O
_SUFFIX = 1 + len(_CHARSET) * (1 + len(_CHARSET))   # 601
_B4 = 1 + len(_CHARSET) + 10                         # 35
_B3 = 10 * _B4 + _SUFFIX                             # 951
_B2 = 10 * _B3 + _SUFFIX                             # 10111
_B1 = 10 * _B2 + _SUFFIX                             # 101711


def _suffix_offset(s):
    n = (len(_CHARSET) + 1) * _CHARSET.index(s[0]) + 1
    if len(s) == 2:
        n += _CHARSET.index(s[1]) + 1
    return n


def n_number_to_icao24(tail):
    """US N-number -> 6-hex-digit Mode S code (lowercase), or None if not a valid N-number."""
    if not tail or not isinstance(tail, str):
        return None
    t = tail.strip().upper()
    if len(t) < 2 or len(t) > 6 or t[0] != "N":
        return None
    t = t[1:]
    if not t[0].isdigit() or t[0] == "0":
        return None
    # letters only allowed as a 1-2 letter suffix after the digits (max 5 chars after N)
    if any(c in "IO" for c in t) or any(not (c.isdigit() or c in _CHARSET) for c in t):
        return None
    out = 0xA00001 + (int(t[0]) - 1) * _B1
    sizes = {1: _B2, 2: _B3, 3: _B4}
    for i in range(1, len(t)):
        c = t[i]
        if i == 4:
            out += _CHARSET.index(c) + 1 if c in _CHARSET else len(_CHARSET) + 1 + int(c)
        elif c in _CHARSET:
            rest = t[i:]
            if len(rest) > 2 or any(x.isdigit() for x in rest):
                return None
            out += _suffix_offset(rest)
            break
        else:
            out += _SUFFIX + int(c) * sizes[i]
    return f"{out:06x}"


def hhmm_to_minutes(v):
    v = int(v)
    return (v // 100) * 60 + v % 100


def pick_nearest_instant(base_date, hhmm, tz, target_utc):
    """Local date+hhmm in tz, choosing the day (-1/0/+1) whose UTC instant is nearest target_utc (handles midnight wraps)."""
    best = None
    mins = hhmm_to_minutes(hhmm)
    for off in (-1, 0, 1):
        loc = (pd.Timestamp(base_date) + pd.Timedelta(days=off) + pd.Timedelta(minutes=mins)).to_pydatetime().replace(tzinfo=tz)
        u = loc.astimezone(dt.timezone.utc)
        if best is None or abs((u - target_utc).total_seconds()) < abs((best - target_utc).total_seconds()):
            best = u
    return best


def prf(pred, truth):
    pred, truth = np.asarray(pred, bool), np.asarray(truth, bool)
    tp = int((pred & truth).sum())
    p = tp / pred.sum() if pred.sum() else None
    r = tp / truth.sum() if truth.sum() else None
    f = 2 * p * r / (p + r) if p and r else None
    return {"n": int(len(truth)), "truth_rate": float(truth.mean()) if len(truth) else None,
            "pred_rate": float(pred.mean()) if len(pred) else None, "precision": p, "recall": r, "f1": f}


def abs_stats(err_min):
    e = np.asarray(err_min, float)
    e = e[~np.isnan(e)]
    if not len(e):
        return {"n": 0}
    return {"n": int(len(e)), "bias_mean_min": float(e.mean()), "bias_median_min": float(np.median(e)),
            "median_abs_min": float(np.median(np.abs(e))), "p90_abs_min": float(np.percentile(np.abs(e), 90))}


# --------------------------------------------------------------------------- fetch
def window_list():
    out = []
    for d in SAMPLE_DAYS:
        t0 = int(dt.datetime.fromisoformat(d).replace(tzinfo=dt.timezone.utc).timestamp())
        out += [(t0 + i * WINDOW_S, t0 + (i + 1) * WINDOW_S) for i in range(12)]
    return out


def cmd_fetch():
    c = oc.OpenSky()
    done = new = 0
    for b, e in window_list():
        try:
            _, cached = c.get("/flights/all", {"begin": b, "end": e}, FLIGHTS_COST)
        except oc.BudgetStop as ex:
            print("STOP (budget):", ex)
            break
        done += 1
        new += 0 if cached else 1
        if not cached:
            print(f"fetched {b}-{e} remaining={c.remaining} spent_total={oc.credits_spent()}")
    print(f"windows available: {done}/{len(window_list())}, newly fetched: {new}, spent_total={oc.credits_spent()}")


def cmd_states():
    c = oc.OpenSky()
    body, cached = c.get("/states/all", {"lamin": 24, "lamax": 50, "lomin": -125, "lomax": -66}, 4)
    print("states cached" if cached else "states fetched", len(body.get("states") or []), "remaining", c.remaining)


# --------------------------------------------------------------------------- load
def load_opensky():
    rows = []
    for b, e in window_list():
        key, _ = oc.cache_key("/flights/all", {"begin": b, "end": e})
        f = oc.CACHE_DIR / f"{key}.json"
        if f.exists():
            rows += json.loads(f.read_text())["body"]
    if not rows:
        raise SystemExit("no cached OpenSky windows; run fetch first")
    d = pd.DataFrame(rows).drop_duplicates(["icao24", "firstSeen", "lastSeen"])
    d["cs"] = d["callsign"].map(norm_callsign)
    return d


def airport_ref(iatas):
    from timezonefinder import TimezoneFinder
    a = pd.read_csv(ROOT / "data" / "ref" / "airports.csv", dtype=str)
    a = a[a.iata_code.isin(iatas)].copy()
    rank = {"large_airport": 0, "medium_airport": 1, "small_airport": 2}
    a["r"] = a["type"].map(rank).fillna(9)
    a = a.sort_values(["iata_code", "r"]).drop_duplicates("iata_code")
    tf = TimezoneFinder()
    ref = {}
    for _, r in a.iterrows():
        icao = r.icao_code if isinstance(r.icao_code, str) else (r.gps_code if isinstance(r.gps_code, str) else r.ident)
        tzn = tf.timezone_at(lat=float(r.latitude_deg), lng=float(r.longitude_deg))
        ref[r.iata_code] = {"icao": icao, "tz": ZoneInfo(tzn) if tzn else None}
    return ref


def load_bts():
    z = zipfile.ZipFile(ROOT / "data" / "bts" / "marketing" /
                        "On_Time_Marketing_Carrier_On_Time_Performance_Beginning_January_2018_2026_8.zip")
    name = [n for n in z.namelist() if n.endswith(".csv")][0]
    cols = ["FlightDate", "Marketing_Airline_Network", "Flight_Number_Marketing_Airline", "IATA_Code_Operating_Airline",
            "Tail_Number", "Flight_Number_Operating_Airline", "Origin", "Dest", "CRSDepTime", "DepTime", "DepDelay",
            "TaxiOut", "WheelsOff", "WheelsOn", "CRSArrTime", "ArrTime", "ArrDelay", "Cancelled", "Diverted", "AirTime"]
    d = pd.read_csv(z.open(name), usecols=cols, low_memory=False)
    d["FlightDate"] = pd.to_datetime(d["FlightDate"])
    days = [pd.Timestamp(x) for x in SAMPLE_DAYS]
    keep = pd.Series(False, index=d.index)
    for x in days:
        keep |= d.FlightDate.between(x - pd.Timedelta(days=1), x + pd.Timedelta(days=1))
    return d, d[keep].copy()


def main_truth():
    full, sub = load_bts()
    codes = set(sub.Origin) | set(sub.Dest)
    ref = airport_ref(codes)
    # airport size tiers by August traffic (origin+dest departures+arrivals)
    traffic = pd.concat([full.Origin, full.Dest]).value_counts()
    tier = {a: ("T1_top30" if i < 30 else "T2_31to100" if i < 100 else "T3_other") for i, a in enumerate(traffic.index)}
    sub = sub[(sub.Cancelled == 0) & (sub.Diverted == 0)].dropna(
        subset=["DepTime", "WheelsOff", "WheelsOn", "ArrTime", "AirTime", "TaxiOut", "ArrDelay", "DepDelay"]).copy()
    sub = sub[sub.Origin.isin(ref) & sub.Dest.isin(ref)]
    sub = sub[[ref[o]["tz"] is not None and ref[d]["tz"] is not None for o, d in zip(sub.Origin, sub.Dest)]]
    out = {k: [] for k in ("sdep", "dep", "woff", "won", "sarr", "arr")}
    U = dt.timezone.utc
    for r in sub.itertuples(index=False):
        otz, dtz = ref[r.Origin]["tz"], ref[r.Dest]["tz"]
        base = pd.Timestamp(r.FlightDate)
        sdep = (base + pd.Timedelta(minutes=hhmm_to_minutes(int(r.CRSDepTime) % 2400))).to_pydatetime().replace(tzinfo=otz).astimezone(U)
        dep = sdep + dt.timedelta(minutes=float(r.DepDelay))
        woff = pick_nearest_instant(base, int(r.WheelsOff) % 2400, otz, dep + dt.timedelta(minutes=float(r.TaxiOut)))
        won = pick_nearest_instant(base, int(r.WheelsOn) % 2400, dtz, woff + dt.timedelta(minutes=float(r.AirTime)))
        arr = pick_nearest_instant(base, int(r.ArrTime) % 2400, dtz, won + dt.timedelta(minutes=8))
        sarr = arr - dt.timedelta(minutes=float(r.ArrDelay))
        for k, v in zip(out, (sdep, dep, woff, won, sarr, arr)):
            out[k].append(v.timestamp())
    for k, v in out.items():
        sub[k] = v
    sub["o_icao"] = sub.Origin.map(lambda x: ref[x]["icao"])
    sub["d_icao"] = sub.Dest.map(lambda x: ref[x]["icao"])
    sub["op"] = sub.IATA_Code_Operating_Airline
    sub["o_tier"] = sub.Origin.map(tier)
    sub["d_tier"] = sub.Dest.map(tier)
    sub["hr_local"] = [int(((pd.Timestamp(t, unit="s", tz="UTC").tz_convert(ref[o]["tz"])).hour)) for t, o in zip(sub.woff, sub.Origin)]
    # scope: inside a sampled UTC day, 1h away from the day edges so the 2h windows fully cover the flight's departure and arrival
    blk = np.full(len(sub), -1)
    for i, d in enumerate(SAMPLE_DAYS):
        t0 = pd.Timestamp(d, tz="UTC").timestamp()
        blk = np.where((sub.woff >= t0 + 3600) & (sub.won <= t0 + 86400 - 3600), i, blk)
    sub["blk"] = blk
    sub = sub[sub.blk >= 0].copy()
    sub["cands"] = [callsign_candidates(o, n, m, mn) for o, n, m, mn in
                    zip(sub.op, sub.Flight_Number_Operating_Airline, sub.Marketing_Airline_Network, sub.Flight_Number_Marketing_Airline)]
    sub["icao24_calc"] = sub.Tail_Number.map(n_number_to_icao24)
    return sub.reset_index(drop=True)


# --------------------------------------------------------------------------- matching
def best_match(bts, osf, kind):
    """For each BTS flight pick the nearest OpenSky flight by callsign candidate + conditions. Returns df indexed like bts."""
    ex = bts[["woff", "won", "o_icao", "d_icao", "cands"]].explode("cands").dropna(subset=["cands"])
    ex = ex.rename(columns={"cands": "cs"})
    ex["bi"] = ex.index
    m = ex.merge(osf[["cs", "icao24", "firstSeen", "lastSeen", "estDepartureAirport", "estArrivalAirport"]], on="cs", how="inner")
    if kind == "callsign_time":
        ok = (m.firstSeen - m.woff).abs() <= 5400
        dist = (m.firstSeen - m.woff).abs()
    elif kind == "dep":
        ok = (m.estDepartureAirport == m.o_icao) & ((m.firstSeen - m.woff).abs() <= 3600)
        dist = (m.firstSeen - m.woff).abs()
    elif kind == "arr":
        ok = (m.estArrivalAirport == m.d_icao) & ((m.lastSeen - m.won).abs() <= 3600)
        dist = (m.lastSeen - m.won).abs()
    elif kind == "full":
        ok = ((m.estDepartureAirport == m.o_icao) & ((m.firstSeen - m.woff).abs() <= 3600) &
              (m.estArrivalAirport == m.d_icao) & ((m.lastSeen - m.won).abs() <= 3600))
        dist = (m.firstSeen - m.woff).abs() + (m.lastSeen - m.won).abs()
    m = m[ok].assign(dist=dist[ok]).sort_values("dist").drop_duplicates("bi")
    return m.set_index("bi")


def route_only_match(bts, osf):
    """No callsign: OS flight with same dep+arr airport and firstSeen within 30 min of wheels-off (any callsign)."""
    o = osf.dropna(subset=["estDepartureAirport", "estArrivalAirport"])
    m = bts[["woff", "o_icao", "d_icao"]].assign(bi=bts.index).merge(
        o[["estDepartureAirport", "estArrivalAirport", "firstSeen"]], left_on=["o_icao", "d_icao"],
        right_on=["estDepartureAirport", "estArrivalAirport"])
    m = m[(m.firstSeen - m.woff).abs() <= 1800]
    return set(m.bi)


def rate_table(bts, flag, by):
    g = bts.assign(flag=flag).groupby(by, observed=True)["flag"].agg(["size", "mean"])
    return {str(k): {"n": int(r["size"]), "rate": float(r["mean"])} for k, r in g.iterrows()}


def analyze():
    osf = load_opensky()
    bts = main_truth()
    res = {"month": MONTH, "sample_days_utc": SAMPLE_DAYS, "windows_cached": int(sum(
        (oc.CACHE_DIR / f"{oc.cache_key('/flights/all', {'begin': b, 'end': e})[0]}.json").exists() for b, e in window_list())),
        "opensky_flights_loaded": int(len(osf)), "bts_flights_in_scope": int(len(bts))}
    for name, kind in (("callsign_time", "callsign_time"), ("dep", "dep"), ("arr", "arr"), ("full", "full")):
        res.setdefault("_m", {})[name] = best_match(bts, osf, kind)
    M = res.pop("_m")
    flag = {k: bts.index.isin(v.index) for k, v in M.items()}
    flag["route_only_no_callsign"] = bts.index.isin(list(route_only_match(bts, osf)))
    flag["has_callsign_candidate"] = bts.cands.map(len).gt(0).values
    res["match_rate_overall"] = {k: float(v.mean()) for k, v in flag.items()}
    res["definitions"] = {
        "callsign_time": "callsign (operating or marketing carrier ICAO+number) equal and OS firstSeen within 90 min of BTS wheels-off",
        "dep": "callsign + estDepartureAirport == origin + firstSeen within 60 min of wheels-off",
        "arr": "callsign + estArrivalAirport == dest + lastSeen within 60 min of wheels-on",
        "full": "dep and arr both on the same OS flight",
        "route_only_no_callsign": "any OS flight with same dep/arr airports and firstSeen within 30 min of wheels-off"}
    bts["op_group"] = bts.op.where(bts.op.isin(REGIONALS), np.where(bts.op.isin(["AA", "DL", "UA", "WN", "AS", "B6", "F9", "G4"]), bts.op, "other"))
    for k in ("dep", "arr", "full"):
        res.setdefault("match_by_carrier", {})[k] = rate_table(bts, flag[k], "op")
        res.setdefault("match_by_origin_tier", {})[k] = rate_table(bts, flag[k], "o_tier")
        res.setdefault("match_by_local_dep_hour", {})[k] = rate_table(bts, flag[k], "hr_local")
    res["match_by_dest_tier_arr"] = rate_table(bts, flag["arr"], "d_tier")
    top = bts.o_icao.value_counts().head(15).index
    res["match_dep_top15_origin_airports"] = rate_table(bts[bts.o_icao.isin(top)], flag["dep"][bts.o_icao.isin(top).values], "o_icao")

    # ---- time accuracy
    d = M["dep"]
    b = bts.loc[d.index]
    res["time_accuracy"] = {
        "dep_firstSeen_minus_wheels_off": abs_stats((d.firstSeen - b.woff) / 60),
        "dep_firstSeen_minus_offblock": abs_stats((d.firstSeen - b.dep) / 60)}
    a = M["arr"]
    ba = bts.loc[a.index]
    res["time_accuracy"]["arr_lastSeen_minus_wheels_on"] = abs_stats((a.lastSeen - ba.won) / 60)
    res["time_accuracy"]["arr_lastSeen_minus_gate_in"] = abs_stats((a.lastSeen - ba.arr) / 60)
    # by tier for departures
    res["time_accuracy_dep_by_origin_tier"] = {t: abs_stats(((d.firstSeen - b.woff) / 60)[b.o_tier == t]) for t in sorted(bts.o_tier.unique())}

    # ---- delay flags (train bias on first half of days, evaluate on second half)
    train_blk = {0, 1, 2}
    out = {}
    for lab, mm, bb, osk, sched, truthcol, ref_t in (
            ("departed_15_late", d, b, "firstSeen", "sdep", "DepDelay", "dep"),
            ("arrived_15_late", a, ba, "lastSeen", "sarr", "ArrDelay", "arr")):
        raw = (mm[osk] - bb[sched]) / 60            # OS-derived delay in minutes
        truth = bb[truthcol] >= 15
        tr = bb.blk.isin(train_blk)
        bias = float(np.median((mm[osk] - bb[ref_t if lab == "departed_15_late" else "arr"])[tr] / 60))  # vs off-block / gate-in
        calib = raw - bias
        te = ~tr
        out[lab] = {
            "calibration_offset_min_from_train_days": bias,
            "matched_only_raw": prf((raw >= 15).values, truth.values),
            "matched_only_bias_corrected_test_days": prf((calib[te] >= 15).values, truth[te].values),
            "delay_error_min_bias_corrected": abs_stats((calib - bb[truthcol]).values),
            "pearson_r_delay": float(np.corrcoef(raw, bb[truthcol])[0, 1]),
        }
        # recall over ALL scoped flights (unmatched count as 'not flagged')
        allp = pd.Series(False, index=bts.index)
        allp.loc[mm.index] = (calib >= 15).values
        allt = bts[truthcol] >= 15
        out[lab]["all_flights_bias_corrected_unmatched_negative"] = prf(allp.values, allt.values)
    res["delay_flag_fidelity"] = out

    # ---- inbound linkage
    ok_tail = bts.icao24_calc.notna()
    res["linkage"] = {"tail_to_icao24_computable": float(ok_tail.mean())}
    dm = M["dep"]
    chk = bts.loc[dm.index]
    res["linkage"]["icao24_formula_agrees_with_opensky_on_matched"] = float((chk.icao24_calc == dm.icao24).mean())
    res["linkage"]["icao24_formula_checked_n"] = int(len(chk))
    # rotations
    s = bts.sort_values(["Tail_Number", "sdep"]).copy()
    s["prev"] = s.groupby("Tail_Number")["won"].shift(1)
    s["prev_dest"] = s.groupby("Tail_Number")["Dest"].shift(1)
    s["prev_blk"] = s.groupby("Tail_Number")["blk"].shift(1)
    s["prev_sarr"] = s.groupby("Tail_Number")["sarr"].shift(1)
    s["prev_arrdelay"] = s.groupby("Tail_Number")["ArrDelay"].shift(1)
    s["prev_d_icao"] = s.groupby("Tail_Number")["d_icao"].shift(1)
    rot = s[(s.prev_dest == s.Origin) & (s.prev_blk == s.blk) & (s.sdep - s.prev > 0) & (s.sdep - s.prev <= 12 * 3600) & s.icao24_calc.notna()].copy()
    res["linkage"]["rotations_in_sample"] = int(len(rot))
    res["linkage"]["share_of_scoped_flights_with_same_day_inbound_leg"] = float(len(rot) / len(bts))
    # prev leg observed by icao24 (aircraft-based, no callsign needed)
    a24 = osf.dropna(subset=["estArrivalAirport"])[["icao24", "lastSeen", "estArrivalAirport"]]
    r = rot[["icao24_calc", "prev_d_icao", "prev", "prev_sarr", "prev_arrdelay", "sdep", "woff"]].assign(ri=rot.index).merge(
        a24, left_on=["icao24_calc", "prev_d_icao"], right_on=["icao24", "estArrivalAirport"])
    r = r[(r.lastSeen - r.prev).abs() <= 3600].assign(d=lambda x: (x.lastSeen - x.prev).abs()).sort_values("d").drop_duplicates("ri").set_index("ri")
    res["linkage"]["inbound_arrival_observed_via_icao24"] = float(len(r) / len(rot)) if len(rot) else None
    cur_dep = set(M["dep"].index)
    both = [i for i in r.index if i in cur_dep]
    res["linkage"]["inbound_arrival_and_this_departure_both_matched"] = float(len(both) / len(rot)) if len(rot) else None
    inb_os_late = ((r.lastSeen - r.prev_sarr) / 60 >= 15)
    inb_truth = r.prev_arrdelay >= 15
    res["linkage"]["inbound_late15_flag_vs_bts_raw_uncalibrated"] = prf(inb_os_late.values, inb_truth.values)
    res["linkage"]["inbound_arrival_time_error_min"] = abs_stats(((r.lastSeen - r.prev) / 60).values)
    # inbound also needs to be known by T-1h / T-3h: share of rotations whose inbound landed >=1h / >=3h before sched dep
    gap = (rot.sdep - rot.prev) / 3600
    res["linkage"]["rotations_inbound_landed_ge_1h_before_sched_dep"] = float((gap >= 1).mean()) if len(rot) else None
    res["linkage"]["rotations_inbound_landed_ge_3h_before_sched_dep"] = float((gap >= 3).mean()) if len(rot) else None
    res["linkage"]["note"] = "same-UTC-day rotations only; overnight aircraft are not linked in this sample"

    # ---- credits & live probes
    rows = oc.ledger_rows()
    res["credits"] = {"spent_total_this_experiment": oc.credits_spent(),
                      "flights_bucket_last_remaining": oc.last_remaining("/flights"),
                      "states_bucket_last_remaining": oc.last_remaining("/states"),
                      "calls_live": sum(1 for x in rows if x.get("status") in (200, 404)),
                      "observed_cost_per_call": sorted({(x["group"], x["req"].split("?")[0], x["cost_observed"]) for x in rows
                                                        if x.get("cost_observed") is not None})}
    res["credits"]["observed_cost_per_call"] = [list(t) for t in res["credits"]["observed_cost_per_call"]]
    res["live_probes"] = live_probe_summary()
    res["projection"] = projection()
    RESULT.parent.mkdir(exist_ok=True)
    RESULT.write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: res[k] for k in ("bts_flights_in_scope", "opensky_flights_loaded", "match_rate_overall", "time_accuracy", "credits")}, indent=1, default=float))


def live_probe_summary():
    out = {}
    for f in oc.CACHE_DIR.glob("*.json"):
        if f.name == "ledger.jsonl":
            continue
        j = json.loads(f.read_text())
        req = j.get("req", "")
        body = j["body"]
        if req.startswith("/flights/departure") or req.startswith("/flights/arrival"):
            kind = req.split("?")[0].split("/")[-1]
            ts = pd.DataFrame(body) if body else pd.DataFrame()
            out[f"{kind}_KATL_recent_10h_window"] = {"n_flights": int(len(ts)), "share_missing_arrival_airport": (float(ts.estArrivalAirport.isna().mean()) if len(ts) else None)}
        elif req.startswith("/states/all"):
            st = body.get("states") or []
            tag = "states_conus" if "lamin=24" in req else "states_ATL_2x2deg"
            cs = [x[1].strip() for x in st if x[1] and x[1].strip()]
            out[tag] = {"n_aircraft": len(st), "n_with_callsign": len(cs), "n_on_ground": int(sum(1 for x in st if x[8])),
                        "n_airline_callsign_3letter_plus_digits": int(sum(1 for c in cs if len(c) >= 4 and c[:3].isalpha() and c[3:].isdigit() and c[:3] in IATA_TO_ICAO.values()))}
    return out


def projection():
    sq = lambda lat, lon: lat * lon
    return {
        "flights_endpoints": "30 credits per call regardless of window (<=2h /all, <=2 days per airport/aircraft); 4000/30 = 133 calls/day",
        "states_cost_by_bbox_area_sq_deg": {"<=25": 1, "25-100": 2, "100-400": 3, ">400": 4},
        "states_buckets_separate_from_flights": True,
        "conus_states_poll_credits_per_day": {f"every_{m}min_24h": int(1440 / m * 4) for m in (1, 2, 5)},
        "per_airport_2x2deg_poll_credits_per_day": {f"every_{m}min_16h_per_airport": int(16 * 60 / m) for m in (2, 5, 10)},
        "per_airport_5x5deg_note": "<=25 sq deg still costs 1 credit, so a metro box up to 5x5 deg is 1 credit per poll",
        "flights_departure_per_airport_poll": {"airports_polled_every_15min_16h": 64 * 30, "note": "one airport polled every 15 min over 16h = 64 calls = 1920 credits; not scalable"},
    }


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    {"fetch": cmd_fetch, "states": cmd_states, "analyze": analyze}[cmd]()
