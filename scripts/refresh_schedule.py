#!/usr/bin/env python3
"""Monthly refresh of models/route_schedule.csv.gz (the "typical schedule" behind flight lookup and route search).

    venv/bin/python scripts/refresh_schedule.py            # what the weekly launchd job runs
    venv/bin/python scripts/refresh_schedule.py --dry-run  # gate + probe only, downloads and swaps nothing

HOLDOUT GATE. The 2026-09 BTS month is the sealed B5.1 holdout (backtest/holdout.py). Until the one-time `score`
stage has written backtest/results/h1_b5_1_2026-09.json, this job is a logged no-op: no network, no file read.
It never sets FLYRELY_HOLDOUT_UNLOCK (it removes it from its own environment) and never imports the frozen
backtest modules except `holdout` and `prepare` read-only. Months from 2026-09 on are kept apart from the
backtest data, in data/schedule/bts/, so nothing in backtest/ can load them.

Once the gate is open: probe the official BTS download (the same transtats.bts.gov/PREZIP files that
backtest/fetch_bts.sh uses) for the month after the newest one on disk; download each new month; rebuild with
scripts/build_route_schedule.py; sanity-check the result against the file in use; swap it in atomically (the old
file stays as route_schedule.csv.gz.prev). copilot/schedule.py reloads the file when it changes.
Optional ntfy message (FLYRELY_NTFY_TOPIC) on success and on failure.
"""
import argparse
import csv
import gzip
import logging
import os
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backtest"))
sys.path.insert(0, str(ROOT / "scripts"))

import build_route_schedule as brs  # noqa: E402
import holdout  # noqa: E402  read-only: month/lock helpers
import prepare  # noqa: E402  read-only: the zip reader and column map

log = logging.getLogger("refresh_schedule")

SCORE_RESULT = ROOT / "backtest" / "results" / f"h1_b5_1_{holdout.LOCKED_FROM}.json"   # written once by `score`
SCHEDULE = ROOT / "models" / "route_schedule.csv.gz"
EXTRA_DIR = ROOT / "data" / "schedule" / "bts"          # zips + parquets for months from the holdout on
BASE_URL = "https://transtats.bts.gov/PREZIP/"
ZIP_NAME = "On_Time_Marketing_Carrier_On_Time_Performance_Beginning_January_2018_{y}_{m}.zip"
WEEKS = 8
MAX_NEW_MONTHS = 6
MAJOR_CARRIERS = ("AA", "DL", "UA", "WN", "AS", "B6")
HAWAII = {"HNL", "OGG", "KOA", "LIH"}
MIN_AS_HAWAII_ROWS = 20
ROW_TOLERANCE = 0.30


class SanityError(Exception):
    pass


# ---------------------------------------------------------------- gate
def gate_open(result: Path = None) -> bool:
    """True only once the B5.1 holdout `score` stage has written its result file."""
    return (result or SCORE_RESULT).is_file()


# ---------------------------------------------------------------- months
def next_month(ym: str) -> str:
    y, m = int(ym[:4]), int(ym[5:])
    return f"{y + m // 12}-{m % 12 + 1:02d}"


def local_months(main_dir: Path = None, extra_dir: Path = None) -> list[str]:
    main_dir = main_dir or brs.BTS_PARQUET
    extra_dir = extra_dir or EXTRA_DIR
    found = {p.stem for p in main_dir.glob("????-??.parquet")} | {p.stem for p in extra_dir.glob("????-??.parquet")}
    return sorted(m for m in found if holdout.is_month(m))


def schedule_as_of(path: Path):
    """schedule_as_of of the file in use (first data row), or None if there is no file."""
    try:
        with gzip.open(path, "rt", newline="") as f:
            row = next(csv.DictReader(f), None)
        return row["schedule_as_of"] if row else None
    except (OSError, EOFError):
        return None


def _url(ym: str) -> str:
    return BASE_URL + ZIP_NAME.format(y=int(ym[:4]), m=int(ym[5:]))


def month_published(ym: str) -> bool:
    """HEAD the official BTS file. 404/403 = not published yet; other errors propagate (job fails and logs)."""
    req = urllib.request.Request(_url(ym), method="HEAD", headers={"User-Agent": "flyrely-schedule-refresh"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status == 200
    except urllib.error.HTTPError as e:
        if e.code in (403, 404):
            return False
        raise


def download_month(ym: str, dest: Path) -> None:
    """Download to a .part file, verify the zip, then rename."""
    tmp = dest.with_suffix(".part")
    req = urllib.request.Request(_url(ym), headers={"User-Agent": "flyrely-schedule-refresh"})
    with urllib.request.urlopen(req, timeout=900) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    with zipfile.ZipFile(tmp) as z:
        if z.testzip() is not None:
            raise RuntimeError(f"corrupt zip for {ym}")
    os.replace(tmp, dest)


def convert_month(zip_path: Path, ym: str, out_dir: Path) -> Path:
    """Marketing Carrier zip -> parquet, same columns as prepare.py's marketing output."""
    import pandas as pd
    out = out_dir / f"{ym}.parquet"
    df = prepare._read(zip_path, prepare.MARKETING_COLUMNS).rename(columns=prepare.MARKETING_RENAME)
    df["FlightDate"] = pd.to_datetime(df["FlightDate"])
    first, last = holdout.month_bounds(ym)
    if len(df) == 0 or df["FlightDate"].min() < pd.Timestamp(first) or df["FlightDate"].max() > pd.Timestamp(last):
        raise RuntimeError(f"BTS file for {ym} is empty or holds dates outside the month")
    tmp = out.with_suffix(".parquet.part")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out)
    return out


def fetch_new_months(extra_dir: Path = None, main_dir: Path = None) -> list[str]:
    """Download and convert every published month after the newest on disk. Returns the months added."""
    extra_dir = extra_dir or EXTRA_DIR
    extra_dir.mkdir(parents=True, exist_ok=True)
    added = []
    have = local_months(main_dir, extra_dir)
    if not have:
        raise RuntimeError("no BTS months on disk")
    ym = have[-1]
    for _ in range(MAX_NEW_MONTHS):
        ym = next_month(ym)
        if not month_published(ym):
            log.info("BTS %s not published yet", ym)
            break
        log.info("BTS %s is published: downloading", ym)
        zp = extra_dir / ZIP_NAME.format(y=int(ym[:4]), m=int(ym[5:]))
        download_month(ym, zp)
        convert_month(zp, ym, extra_dir)
        added.append(ym)
    return added


# ---------------------------------------------------------------- build, check, swap
def sanity_check(new, old_rows: int = None, old_as_of: str = None) -> dict:
    """`new` is the rebuilt schedule DataFrame. Raises SanityError on a bad file; returns the stats."""
    n = len(new)
    if n == 0:
        raise SanityError("new schedule is empty")
    if old_rows:
        lo, hi = old_rows * (1 - ROW_TOLERANCE), old_rows * (1 + ROW_TOLERANCE)
        if not lo <= n <= hi:
            raise SanityError(f"row count {n:,} is outside +/-{ROW_TOLERANCE:.0%} of the old {old_rows:,}")
    carriers = set(new["marketing_airline"])
    missing = [c for c in MAJOR_CARRIERS if c not in carriers]
    if missing:
        raise SanityError(f"major carriers missing: {missing}")
    if "NK" in carriers:
        raise SanityError("NK (Spirit, defunct) rows are present")
    a = new[new["marketing_airline"] == "AS"]
    hi_rows = int((a["origin"].isin(HAWAII) | a["dest"].isin(HAWAII)).sum())
    if hi_rows < MIN_AS_HAWAII_ROWS:
        raise SanityError(f"only {hi_rows} AS rows touch Hawaii (need {MIN_AS_HAWAII_ROWS}); former Hawaiian flying missing")
    as_of = str(new["schedule_as_of"].iloc[0])
    if old_as_of and as_of < old_as_of:
        raise SanityError(f"new schedule_as_of {as_of} is older than the current {old_as_of}")
    return {"rows": n, "as_of": as_of, "AS_hawaii_rows": hi_rows}


def build_new(tmp_path: Path, extra_dir: Path = None, main_dir: Path = None):
    """Rebuild with the existing builder. Writes tmp_path, returns the DataFrame."""
    df, as_of = brs.load_window(WEEKS, main_dir or brs.BTS_PARQUET, extra_src=extra_dir or EXTRA_DIR)
    out = brs.build(df, as_of)
    brs._write(out, tmp_path)
    if tmp_path.stat().st_size > brs.MAX_BYTES:
        raise SanityError(f"new schedule is {tmp_path.stat().st_size:,} bytes, over the limit")
    return out


def swap_in(new_path: Path, current: Path) -> None:
    """Keep the old file as <name>.prev, then atomically replace the current one (same filesystem)."""
    if current.exists():
        prev = current.with_name(current.name + ".prev")
        tmp = current.with_name(current.name + ".prev.tmp")
        shutil.copy2(current, tmp)
        os.replace(tmp, prev)
    os.replace(new_path, current)


# ---------------------------------------------------------------- main
def refresh(dry_run: bool = False, schedule: Path = None, extra_dir: Path = None, main_dir: Path = None,
            result: Path = None) -> str:
    """Returns a short status: 'gate-closed', 'up-to-date', 'dry-run' or 'refreshed <month>'."""
    schedule = schedule or SCHEDULE
    if not gate_open(result):
        log.info("holdout not yet scored, skipping (waiting for %s)", (result or SCORE_RESULT).name)
        return "gate-closed"
    log.info("holdout gate open (score result present)")
    if dry_run:
        log.info("dry run: would probe BTS for the month after %s", (local_months(main_dir, extra_dir) or ["?"])[-1])
        return "dry-run"
    added = fetch_new_months(extra_dir, main_dir)
    newest = local_months(main_dir, extra_dir)[-1]
    old_as_of = schedule_as_of(schedule)
    if not added and old_as_of and old_as_of >= newest:
        log.info("up to date: schedule_as_of %s, newest BTS month %s, nothing new", old_as_of, newest)
        return "up-to-date"
    old_rows = None
    if schedule.exists():
        with gzip.open(schedule, "rt") as f:
            old_rows = max(sum(1 for _ in f) - 1, 0)
    tmp = schedule.with_name(schedule.name + ".new")
    try:
        new = build_new(tmp, extra_dir, main_dir)
        stats = sanity_check(new, old_rows, old_as_of)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    swap_in(tmp, schedule)
    log.info("refreshed: %s rows (was %s), schedule_as_of %s (was %s), AS Hawaii rows %d, new months %s; previous file kept as %s.prev",
             f"{stats['rows']:,}", f"{old_rows:,}" if old_rows else "none", stats["as_of"], old_as_of,
             stats["AS_hawaii_rows"], added or "none", schedule.name)
    return f"refreshed {stats['as_of']}"


def _notify(title: str, body: str, priority: str = "default", tags: str = "") -> None:
    try:
        from swim import ntfy
        ntfy.send(title, body, priority, tags)
    except Exception as e:  # never let an alert problem mask the real result
        log.warning("ntfy failed: %s", type(e).__name__)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="check the gate and stop; no network, no changes")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    os.environ.pop(holdout.UNLOCK_VAR, None)    # this job must never carry the holdout unlock
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(os.environ.get("FLYRELY_ENV_FILE", ROOT / ".env")))
    except ImportError:
        pass
    os.environ.pop(holdout.UNLOCK_VAR, None)
    try:
        status = refresh(a.dry_run)
    except Exception as e:
        log.exception("schedule refresh failed")
        _notify("FlyRely schedule refresh failed", f"{type(e).__name__}: {str(e)[:200]}", "high", "warning")
        return 1
    if status.startswith("refreshed"):
        _notify("FlyRely schedule", f"Schedule refreshed to {status.split()[1]}", "default", "white_check_mark")
    return 0


if __name__ == "__main__":
    sys.exit(main())
