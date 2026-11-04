"""
Fetch FAA ATCSCC advisories for ground stops (GS) and ground delay programs (GDP) from the public
Advisories Database (https://www.fly.faa.gov/adv/advADB, no key).

    python backtest/fetch_faa_adv.py lists   [--start 2024-01-01] [--end 2026-08-31] [--pause 1.5]
    python backtest/fetch_faa_adv.py details [--airports-top 100] [--pause 1.5]
    python backtest/fetch_faa_adv.py build                      # -> data/faa_adv/programs.parquet

Phase 1 (`lists`): one request per day, listing every GS/GDP advisory of that day (number, control element,
title, send time in UTC). Cached as data/faa_adv/list/<date>.html.
Phase 2 (`details`): one request per non-cancel advisory at the chosen airports, to read the program period
("GROUND STOP PERIOD", "CUMULATIVE PROGRAM PERIOD"). Cached as data/faa_adv/detail/<date>_<n>.txt.
Phase 3 (`build`): parse into data/faa_adv/programs.parquet (one row per advisory).

Polite: strictly sequential, a pause between requests, back-off on errors, everything cached, resumable.
Holdout lock: dates after 2026-08-31 are refused unless FLYRELY_HOLDOUT_UNLOCK=1.
"""

import argparse
import html
import os
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = ROOT / "data" / "faa_adv"
BASE = "http://www.fly.faa.gov/adv"
UA = "flyrely-research/1.0 (sequential, cached; seanthomasbutton@gmail.com)"
HOLDOUT_LAST_ALLOWED = "2026-08-31"
LIST_PARAMS = {"whichAdvisories": "ATCSCC", "advisoryCategory": "NotAll", "gStop": "true", "_gStop": "on",
               "gDelay": "true", "_gDelay": "on", "_airflow": "on", "_ctop": "on", "_route": "on", "_other": "on"}


def check_holdout(end: str) -> None:
    if date.fromisoformat(end) > date.fromisoformat(HOLDOUT_LAST_ALLOWED) and \
            os.environ.get("FLYRELY_HOLDOUT_UNLOCK") != "1":
        raise SystemExit(f"refusing {end}: dates after {HOLDOUT_LAST_ALLOWED} are the untouched holdout "
                         "(set FLYRELY_HOLDOUT_UNLOCK=1 to override)")


# =============================================================================
# Parsing (pure functions, unit-tested)
# =============================================================================

_ROW2 = re.compile(r'adv_otherdis\?adv_date=(\d{8})&(?:amp;)?advn=(\d+)"[^>]*>(.*?)(?=<a\s[^>]*adv_otherdis\?adv_date|</table>|$)', re.S | re.I)


def _text(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]*>", " ", s))).strip()


def classify(title: str):
    """(kind, status) from a brief title; None for anything that is not a GS or a GDP program advisory.
    kind: GS | GDP. status: active | proposed | cancel."""
    t = title.upper()
    if "CNX" in t.split() or t.endswith(" CNX"):
        status = "cancel"
    elif "PROPOSED" in t:
        status = "proposed"
    else:
        status = "active"
    if "GROUND STOP" in t or re.search(r"\bGS\b", t):
        return "GS", status
    if "GROUND DELAY PROGRAM" in t:
        return "GDP", status
    return None


def parse_list(page: str) -> list[dict]:
    """One dict per GS/GDP advisory on a daily list page."""
    out = []
    body = page.replace("&amp;", "&")
    for m in _ROW2.finditer(body):
        d, advn, rest = m.group(1), int(m.group(2)), _text(m.group(3))
        mm = re.match(r"^\d+\s+(\S+)/(\S+)\s+(\d\d/\d\d/\d\d)\s+(.*?)\s+(\d\d/\d\d/\d\d)\s+(\d\d:\d\d)$", rest)
        if not mm:
            continue
        element, center, _, title, sdate, stime = mm.groups()
        c = classify(title)
        if c is None:
            continue
        sent = datetime.strptime(f"{sdate} {stime}", "%m/%d/%y %H:%M").replace(tzinfo=timezone.utc)
        out.append({"adv_date": d, "advn": advn, "element": element, "center": center, "title": title,
                    "kind": c[0], "status": c[1], "sent_utc": sent})
    return out


_PERIOD = re.compile(r"(GROUND STOP PERIOD|CUMULATIVE PROGRAM PERIOD|GDP CNX PERIOD)\s*:\s*"
                     r"(\d{2})/(\d{4})Z\s*-\s*(\d{2})/(\d{4})Z")


def _stamp(anchor: datetime, dd: int, hhmm: str) -> datetime:
    """Resolve a DD/HHMMZ stamp to the datetime closest to `anchor` (handles month ends)."""
    best = None
    for k in (-1, 0, 1):
        y, mth = anchor.year, anchor.month + k
        y, mth = y + (mth - 1) // 12, (mth - 1) % 12 + 1
        try:
            cand = datetime(y, mth, dd, int(hhmm[:2]) % 24, int(hhmm[2:]), tzinfo=timezone.utc)
        except ValueError:
            continue
        if int(hhmm[:2]) == 24:
            cand += timedelta(hours=0)
        if best is None or abs(cand - anchor) < abs(best - anchor):
            best = cand
    return best


def parse_detail(text: str, sent: datetime) -> dict:
    """Program period (UTC) from the body of an advisory; NaT when absent."""
    flat = html.unescape(re.sub(r"<[^>]*>", " ", text))
    m = _PERIOD.search(flat)
    if not m:
        return {"period_start": pd.NaT, "period_end": pd.NaT}
    _, d1, t1, d2, t2 = m.groups()
    start = _stamp(sent, int(d1), t1)
    end = _stamp(start, int(d2), t2)
    if end < start:
        end += timedelta(days=31)
    return {"period_start": pd.Timestamp(start), "period_end": pd.Timestamp(end)}


# =============================================================================
# Fetching
# =============================================================================

def get(client: httpx.Client, url: str, params=None, tries: int = 6) -> str:
    for k in range(tries):
        try:
            r = client.get(url, params=params, timeout=120)
            if r.status_code == 200 and len(r.text) > 500:
                return r.text
            err = f"HTTP {r.status_code} len {len(r.text)}"
        except httpx.HTTPError as e:
            err = repr(e)
        wait = min(300, 10 * 2 ** k)
        print(f"  {url} {err}; back off {wait}s", flush=True)
        time.sleep(wait)
    raise SystemExit(f"giving up on {url}")


def cmd_lists(a):
    check_holdout(a.end)
    (OUT / "list").mkdir(parents=True, exist_ok=True)
    days = [d.date() for d in pd.date_range(a.start, a.end)]
    todo = [d for d in days if not (OUT / "list" / f"{d}.html").exists()]
    print(f"{len(todo)} of {len(days)} daily lists to fetch", flush=True)
    with httpx.Client(headers={"User-Agent": UA}, follow_redirects=True) as c:
        for i, d in enumerate(todo, 1):
            page = get(c, f"{BASE}/adv_list", dict(LIST_PARAMS, date=str(d)))
            tmp = OUT / "list" / f"{d}.html.part"
            tmp.write_text(page)
            tmp.rename(OUT / "list" / f"{d}.html")
            if i % 25 == 0 or i == len(todo):
                print(f"[{i}/{len(todo)}] {d}", flush=True)
            time.sleep(a.pause)


def list_frame() -> pd.DataFrame:
    rows = []
    for p in sorted((OUT / "list").glob("*.html")):
        for r in parse_list(p.read_text()):
            r["list_date"] = p.stem
            rows.append(r)
    return pd.DataFrame(rows)


def top_airports(n: int) -> set:
    from fetch_metar import busiest_airports
    return set(busiest_airports(n))


def cmd_details(a):
    lst = list_frame()
    keep = lst[(lst["status"] != "cancel") & lst["element"].isin(top_airports(a.airports_top))]
    keep = keep.drop_duplicates(["adv_date", "advn"])
    (OUT / "detail").mkdir(parents=True, exist_ok=True)
    todo = [r for r in keep.itertuples() if not (OUT / "detail" / f"{r.adv_date}_{r.advn}.txt").exists()]
    print(f"{len(todo)} of {len(keep)} detail pages to fetch", flush=True)
    with httpx.Client(headers={"User-Agent": UA}, follow_redirects=True) as c:
        for i, r in enumerate(todo, 1):
            page = get(c, f"{BASE}/adv_otherdis", {"adv_date": r.adv_date, "advn": r.advn})
            body = re.findall(r"<PRE>(.*?)</PRE>", page, re.S | re.I)
            text = html.unescape(body[0]) if body else ""
            tmp = OUT / "detail" / f"{r.adv_date}_{r.advn}.part"
            tmp.write_text(text or "(no body)")
            tmp.rename(OUT / "detail" / f"{r.adv_date}_{r.advn}.txt")
            if i % 100 == 0 or i == len(todo):
                print(f"[{i}/{len(todo)}] {r.adv_date} #{r.advn}", flush=True)
            time.sleep(a.pause)


def cmd_build(a):
    lst = list_frame()
    ends = []
    for r in lst.itertuples():
        f = OUT / "detail" / f"{r.adv_date}_{r.advn}.txt"
        d = parse_detail(f.read_text(), r.sent_utc.to_pydatetime()) if f.exists() else \
            {"period_start": pd.NaT, "period_end": pd.NaT}
        ends.append(d)
    out = pd.concat([lst.reset_index(drop=True), pd.DataFrame(ends)], axis=1)
    out["sent_utc"] = pd.to_datetime(out["sent_utc"], utc=True).dt.tz_localize(None)
    for c in ("period_start", "period_end"):
        out[c] = pd.to_datetime(out[c], utc=True).dt.tz_localize(None)
    out.to_parquet(OUT / "programs.parquet")
    print(out.groupby(["kind", "status"]).size())
    print("wrote", OUT / "programs.parquet", len(out))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("lists", "details", "build"):
        s = sub.add_parser(name)
        s.add_argument("--start", default="2024-01-01")
        s.add_argument("--end", default=HOLDOUT_LAST_ALLOWED)
        s.add_argument("--pause", type=float, default=1.5)
        s.add_argument("--airports-top", type=int, default=100)
    a = ap.parse_args()
    {"lists": cmd_lists, "details": cmd_details, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
