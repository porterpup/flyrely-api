"""CHAIN measurement: how much of the departure board can be linked to its aircraft by M1 / M4 anchor walk / M2 / M3.

Protocol: linkage/PROTOCOL_CHAIN.md (pre-registered). Offline, read-only on the live store, aggregates only: the output
JSON and the printed summary carry counts and shares, never a registration, icao24 or leg id.

    python -m linkage.studies.chain_measure --store <live.sqlite> --start 2026-11-08 --end 2026-11-12 \
        --horizons 180,60 --out linkage/results/chain_2026-11-08_2026-11-12.json

Cutoffs are rounded down to a --tick (default 5 min) boundary, so every fact used was received at or before the
true cutoff (never later); this lets one as-of state serve every departure of a tick.
"""
import argparse
import json
import sqlite3
import sys
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from linkage import chain_rules as cr
from linkage import priors_offline as po
from linkage.chain_rules import AsOfState, Leg
from linkage.store import Store, parse_ts
from linkage.top100 import TOP100

MAINLINE = ("AAL", "DAL", "UAL", "SWA", "JBU", "ASA", "FFT", "AAY")
REGIONAL = ("SKW", "RPA", "ENY", "EDV", "PDT", "JIA", "GJS", "ASH", "QXE", "UCA", "AWI", "CPZ")
CARRIERS = frozenset(MAINLINE) | frozenset(REGIONAL)

A7 = {  # linkage -> Brier gain; (0, 0) added as the origin, all else from the protocol
    180: [(0.0, 0.0), (0.29, 0.009), (0.50, 0.015), (0.75, 0.023), (1.0, 0.031)],
    60: [(0.0, 0.0), (0.29, 0.012), (0.50, 0.022), (0.75, 0.033), (1.0, 0.044)],
}
KNOWN_OUTAGES = [("2026-11-13T09:49:00Z", "2026-11-13T21:50:00Z")]
GAP_S = 15 * 60
DAY_LO_H, DAY_HI_H = 13, 27          # a day's departures: 13:00 UTC to 03:00 UTC next day
H18 = 18 * 3600

# decision bar (protocol)
BAR = {180: (0.50, 0.80), 60: (0.70, 0.90)}   # horizon -> (effective linkage, precision)


def group_of(carrier: str) -> str:
    return carrier if carrier in MAINLINE else "REG"


def epoch(d: date) -> float:
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp()


def interp(curve, x: float) -> float:
    x = max(0.0, min(1.0, x))
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return curve[-1][1]


# ------------------------------------------------------------------ loading
def open_ro(path: str):
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
    db.row_factory = sqlite3.Row
    return db


def load(db):
    """-> (legs: list[Leg], cancelled: set of leg ids, obs: list for AsOfState, truth: dict). Uses Store.chain_legs."""
    st = Store.__new__(Store)       # read-only: skip Store.__init__ (mkdir, WAL pragma, schema script)
    st.db = db
    rows = st.chain_legs(CARRIERS)
    legs, cancelled = [], set()
    for r in rows:
        so = parse_ts(r["sched_out_utc"]).timestamp()
        si = parse_ts(r["sched_in_utc"]).timestamp() if r["sched_in_utc"] else so + 5400
        legs.append(Leg(r["leg_id"], r["carrier_icao"], str(r["flight_no"]), r["origin"], r["dest"], so, si,
                        cr.type_family(r["aircraft_type"])))
        if r["cancelled"]:
            cancelled.add(r["leg_id"])
    ids = {lg.id for lg in legs}
    ac = [r for r in db.execute("SELECT leg_id, source, reg, icao24, first_recv, last_recv FROM leg_aircraft "
                                "WHERE source IN ('sfdps','smes')") if r["leg_id"] in ids]
    reg2hex = {r["reg"]: r["icao24"] for r in ac if r["source"] == "sfdps" and r["reg"] and r["icao24"]}

    def key(r):
        return r["icao24"] or reg2hex.get(r["reg"]) or ("r:" + r["reg"])

    obs, last = [], {}
    for r in ac:
        k = key(r)
        obs.append((parse_ts(r["first_recv"]).timestamp(), r["leg_id"], k, r["source"]))
        lr = (r["source"] == "smes", r["last_recv"])      # SMES beats SFDPS; then the latest last_recv
        if r["leg_id"] not in last or lr > last[r["leg_id"]][0]:
            last[r["leg_id"]] = (lr, k)
    truth = {lid: v[1] for lid, v in last.items()}
    return legs, cancelled, obs, truth


def gap_report(db, day: date) -> dict:
    """Largest gap (minutes) in SFDPS and in TFMS receive times between 11:00 and 03:00 UTC, from archive receive
    stamps in the store (SFDPS: aircraft facts; TFMS: leg update stamps)."""
    lo, hi = epoch(day) + 11 * 3600, epoch(day) + 27 * 3600
    iso = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    a, b = iso(lo), iso(hi)
    out = {}
    for name, sql in (
            ("sfdps", "SELECT first_recv FROM leg_aircraft WHERE source='sfdps' AND first_recv BETWEEN ?1 AND ?2 "
                      "UNION ALL SELECT last_recv FROM leg_aircraft WHERE source='sfdps' AND last_recv BETWEEN ?1 AND ?2"),
            ("tfms", "SELECT last_recv FROM legs WHERE last_recv BETWEEN ?1 AND ?2 "
                     "UNION ALL SELECT est_recv FROM legs WHERE est_recv BETWEEN ?1 AND ?2 "
                     "UNION ALL SELECT first_seen_recv FROM legs WHERE first_seen_recv BETWEEN ?1 AND ?2")):
        ts = sorted(parse_ts(r[0]).timestamp() for r in db.execute(sql, (a, b)))
        w = [lo] + ts + [hi]
        out[name] = round(max(y - x for x, y in zip(w, w[1:])) / 60, 1)
    out["outage"] = any(parse_ts(s).timestamp() < hi and parse_ts(e).timestamp() > lo for s, e in KNOWN_OUTAGES)
    out["complete"] = (not out["outage"]) and out["sfdps"] <= GAP_S / 60 and out["tfms"] <= GAP_S / 60
    return out


# ------------------------------------------------------------------ measurement
class Net:
    """Indexes over the leg universe."""

    def __init__(self, legs, truth):
        self.legs = legs
        self.by_id = {lg.id: lg for lg in legs}
        self.dep, self.arr, self.by_truth = defaultdict(list), defaultdict(list), defaultdict(list)
        for lg in sorted(legs, key=lambda g: g.so):
            self.dep[(lg.carrier, lg.origin)].append(lg)
            self.arr[(lg.carrier, lg.dest)].append(lg)
            if lg.id in truth:
                self.by_truth[truth[lg.id]].append(lg)
        self._so = {}

    def departures(self, carrier, airport):
        return self.dep.get((carrier, airport), ())


def new_acc():
    return defaultdict(int)


def resolve_claims(cl):
    """Best claim of an M4 walk set: high confidence first, then fewest steps; None if the best are ambiguous."""
    if not cl:
        return None
    best = min(cl, key=lambda c: (not c[2], c[1]))
    top = [c for c in cl if (not c[2], c[1]) == (not best[2], best[1])]
    return best if len({c[0] for c in top}) == 1 else None


def claimed(a: Leg, b: Leg, state: AsOfState, net: Net) -> bool:
    """Inbound candidate `a` is already used: its aircraft is filed on another departure from a.dest."""
    y = state.ident(a.id)
    if y is None:
        return False
    for d in state.legs_of.get(y, ()):
        if d != b.id and d != a.id:
            g = net.by_id[d]
            if g.origin == a.dest and g.so >= a.si:
                return True
    return False


def predict_b(b, state, net, priors, claims):
    """-> (method, aircraft_key or None, inbound Leg or None, high) for departure b at the state's cutoff."""
    k = state.ident_sfdps(b.id)
    if k is not None:
        return "M1", k, None, True, 0, ""
    c = resolve_claims(claims.get(b.id))
    if c is not None:
        return "M4", c[0], None, c[2], c[1], c[3]
    day = po.op_day(b.so)
    cands = [a for a in net.arr.get((b.carrier, b.origin), ()) if 0 < b.so - a.so <= H18 and a.si <= b.so]
    if b.carrier in cr.M2_ALLOW:
        m = [a for a in cands if a.fn == b.fn and cr.family_ok(a.fam, b.fam) and not claimed(a, b, state, net)]
        if m:
            return "M2", None, m[-1], True, 0, ""
    pf = priors.prev_fn(b.carrier, b.fn, b.origin, day)
    if pf is not None:
        m = [a for a in cands if a.fn == pf and cr.family_ok(a.fam, b.fam) and not claimed(a, b, state, net)]
        if m:
            return "M3", None, m[-1], False, 0, ""
    return "none", None, None, False, 0, ""


def anchor_exists(b, x, state, net) -> bool:
    """Upper bound: the aircraft was identified at the cutoff on one of its (up to 6) latest legs in the 18 h before b."""
    seq = net.by_truth.get(x, ())
    so = [g.so for g in seq]
    j = bisect_left(so, b.so)
    for g in seq[max(0, j - cr.MAX_STEPS):j]:
        if b.so - g.so <= H18 and state.ident(g.id) == x:
            return True
    return state.ident_sfdps(b.id) == x


def run_day(day: date, legs, net, cancelled, obs, truth, horizons, tick, acc, notes):
    base = epoch(day)
    lo, hi = base + DAY_LO_H * 3600, base + DAY_HI_H * 3600
    dnum = int(base // 86400)
    priors = po.build(legs, truth, dnum)
    pop = [b for b in legs if lo <= b.so < hi and b.origin in TOP100 and b.id not in cancelled]
    notes["pop_all"] += len(pop)
    have = [b for b in pop if b.id in truth]
    notes["pop_no_truth"] += len(pop) - len(have)
    ticks = defaultdict(list)       # tick start -> [(H, b)]
    for b in have:
        for h in horizons:
            t = (b.so - h * 60) // tick * tick
            ticks[t].append((h, b))
    state = AsOfState(obs)
    pn_memo = {}

    def prior_next(cur):
        if cur.id not in pn_memo:
            pn_memo[cur.id] = priors.next_fn(cur.carrier, cur.fn, cur.dest, po.op_day(cur.so))
        return pn_memo[cur.id]

    for t in sorted(ticks):
        state.advance(t)
        want = {b.id for _, b in ticks[t] if state.ident_sfdps(b.id) is None}
        claims = defaultdict(list)
        if want:
            ident = state.any.get
            for x, ids in state.legs_of.items():
                ls = sorted((net.by_id[i] for i in ids), key=lambda g: g.so)
                past = [g for g in ls if g.so <= t and t - g.so <= H18]
                if past:
                    anchor = past[-1]
                else:
                    fut = [g for g in ls if g.so > t]
                    if not fut:
                        continue
                    anchor = fut[0]
                prevs = [g for g in ls if g.so < anchor.so and g.dest == anchor.origin and g.carrier == anchor.carrier]
                pm = bool(prevs) and prior_next(prevs[-1]) == anchor.fn
                steps = cr.walk(anchor, x, net.departures, ident, prior_next, pm)
                high = True
                for n, s in enumerate(steps, 1):
                    high = high and s.high
                    if s.leg.id in want:
                        claims[s.leg.id].append((x, n, high, s.method))
        for h, b in ticks[t]:
            method, key, inbound, high, nsteps, last = predict_b(b, state, net, priors, claims)
            tx = truth[b.id]
            for g in (group_of(b.carrier), "ALL"):
                a = acc[(h, g)]
                a["n"] += 1
                a[f"method_{method}_linked"] += method != "none"
                if method == "none":
                    a["none"] += 1
                    ok = None
                else:
                    if key is not None:
                        ok = key == tx
                    else:
                        ok = truth.get(inbound.id) == tx
                        a["unverifiable_inbound"] += truth.get(inbound.id) is None
                    a["linked"] += 1
                    a["correct"] += ok
                    a[f"method_{method}_correct"] += ok
                    if method == "M4":
                        for tag in (f"last_{last}", f"steps_{nsteps}", f"high_{high}"):
                            a[f"m4_{tag}_linked"] += 1
                            a[f"m4_{tag}_correct"] += ok
                    if high:
                        a["high_linked"] += 1
                        a["high_correct"] += ok
                if anchor_exists(b, tx, state, net):
                    a["anchor"] += 1


def share(a, b):
    return round(a / b, 4) if b else None


def summarise(a):
    n = a["n"]
    linked, correct = a["linked"], a["correct"]
    out = {
        "n": n,
        "coverage": share(linked, n),
        "precision": share(correct, linked),
        "effective": share(correct, n),
        "wrong_share": share(linked - correct, n),
        "high_share": share(a["high_linked"], n),
        "high_precision": share(a["high_correct"], a["high_linked"]),
        "anchor_upper_bound": share(a["anchor"], n),
        "unverifiable_inbound_share_of_linked": share(a["unverifiable_inbound"], linked),
        "m4_diagnostics": {k[3:-7]: {"linked": v, "precision": share(a[k[:-7] + "_correct"], v)}
                           for k, v in sorted(a.items()) if k.startswith("m4_") and k.endswith("_linked")},
        "by_method": {m: {"linked_share": share(a[f"method_{m}_linked"], n),
                          "precision": share(a[f"method_{m}_correct"], a[f"method_{m}_linked"])}
                      for m in ("M1", "M4", "M2", "M3")},
    }
    return out


def brier(sm, h):
    """Brier gain read off the A7 curve: raw at the effective (correct) share, discounted at correct - wrong."""
    if not sm["n"]:
        return None
    eff, wrong = sm["effective"], sm["wrong_share"]
    return {"raw": round(interp(A7[h], eff), 4), "wrong_discounted": round(interp(A7[h], max(0.0, eff - wrong)), 4)}


def decide(res, horizons):
    """Protocol decision rule on the pooled (ALL) rows."""
    ok = {}
    for h, (eff, prec) in BAR.items():
        r = res.get(str(h), {}).get("ALL")
        ok[h] = bool(r and r["effective"] is not None and r["effective"] >= eff and (r["precision"] or 0) >= prec)
    if 180 in horizons and 60 in horizons:
        verdict = "GO" if ok[180] and ok[60] else ("T-1h-only GO" if ok[60] else "STOP")
    else:
        verdict = "not decidable (needs horizons 180 and 60)"
    return {"verdict": verdict, "t3h_conditions_met": ok[180], "t1h_conditions_met": ok[60],
            "bar": {"t3h": {"effective_min": 0.50, "precision_min": 0.80},
                    "t1h": {"effective_min": 0.70, "precision_min": 0.90}}}


def measure(legs, cancelled, obs, truth, days, horizons, tick=300):
    net = Net(legs, truth)
    acc, notes = defaultdict(new_acc), defaultdict(int)
    for d in days:
        run_day(d, legs, net, cancelled, obs, truth, horizons, tick, acc, notes)
    res = {}
    for (h, g), a in sorted(acc.items()):
        sm = summarise(a)
        sm["brier_gain_a7"] = brier(sm, h)
        res.setdefault(str(h), {})[g] = sm
    pop = notes["pop_all"]
    return {
        "population": {"legs": pop, "no_truth": notes["pop_no_truth"],
                       "no_truth_share": share(notes["pop_no_truth"], pop)},
        "results": res,
        "decision": decide(res, horizons),
    }


def valid_days(db, start: date, end: date, enforce=True, max_extra=7):
    """Days to measure: start..end, minus incomplete days, replaced by following complete days."""
    want = (end - start).days + 1
    days, excluded, d = [], {}, start
    last = end + timedelta(days=max_extra)
    store_end = db.execute("SELECT MAX(last_recv) FROM legs").fetchone()[0]
    store_end = parse_ts(store_end).timestamp() if store_end else 0
    while d <= last and len(days) < want:
        g = gap_report(db, d)
        covered = store_end >= epoch(d) + 27 * 3600
        if (g["complete"] or not enforce) and (covered or d <= end):
            days.append(d)
        else:
            excluded[d.isoformat()] = g if covered or d <= end else {"reason": "day not yet complete in store"}
            if d > end and not covered:
                break
        d += timedelta(days=1)
    return days, excluded


def md_summary(out) -> str:
    lines = [f"CHAIN measurement {out['start']} to {out['end']} (days used: {', '.join(out['days_used'])}); "
             f"no-truth share {out['population']['no_truth_share']}", "",
             "| horizon | group | n | coverage | precision | effective | high share | high prec | anchor UB | Brier eff. |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for h, grp in out["results"].items():
        for g, s in grp.items():
            br = s["brier_gain_a7"]
            lines.append(f"| {'T-3h' if h == '180' else 'T-1h' if h == '60' else h + 'm'} | {g} | {s['n']} | {s['coverage']} | "
                         f"{s['precision']} | {s['effective']} | {s['high_share']} | {s['high_precision']} | "
                         f"{s['anchor_upper_bound']} | {br['wrong_discounted'] if br else None} |")
    lines += ["", f"Decision (pooled): {out['decision']['verdict']}"]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--store", required=True, help="path to live.sqlite (opened read-only, never copied)")
    ap.add_argument("--start", required=True, type=date.fromisoformat)
    ap.add_argument("--end", required=True, type=date.fromisoformat)
    ap.add_argument("--horizons", default="180,60")
    ap.add_argument("--tick", type=int, default=300, help="cutoff rounding in seconds (rounds down)")
    ap.add_argument("--no-gap-check", action="store_true", help="keep days that fail the 15-minute gap check")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    horizons = [int(x) for x in a.horizons.split(",")]
    db = open_ro(a.store)
    days, excluded = valid_days(db, a.start, a.end, enforce=not a.no_gap_check)
    legs, cancelled, obs, truth = load(db)
    out = {"protocol": "linkage/PROTOCOL_CHAIN.md", "start": a.start.isoformat(), "end": a.end.isoformat(),
           "horizons_min": horizons, "tick_s": a.tick,
           "days_used": [d.isoformat() for d in days], "days_excluded": excluded,
           "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d")}
    out.update(measure(legs, cancelled, obs, truth, days, horizons, a.tick))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, sort_keys=False) + "\n")
    print(md_summary(out))


if __name__ == "__main__":
    sys.exit(main())
