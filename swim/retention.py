"""Retention for raw SWIM archives: delete day folders older than N days, never the newest.

Layout handled:  <root>/<source>/<YYYY-MM-DD>/*.jsonl.gz|*.jsonl.zst   (only those files are removed,
and only inside real (non-symlink) day folders whose NAME is a date).

Usage:  python -m swim.retention [--root data/swim] [--days 14] [--dry-run]
                                 [--min-free-gb 10] [--alert]
Exit code: 0 ok, 2 free disk space below --min-free-gb (also logged as a WARNING).
"""
import argparse
import logging
import re
import shutil
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from swim.archive import EXTS

log = logging.getLogger("swim.retention")
DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# STDDS is archived unfiltered (about 4-5 GB/day compressed), so it keeps far fewer days than the
# filtered TFMS/SFDPS feeds. Override with --source-days stdds=N.
DEFAULT_SOURCE_DAYS = {"stdds": 1}


def day_dirs(source_dir: Path) -> list:
    out = []
    for d in sorted(source_dir.iterdir()) if source_dir.is_dir() else []:
        if d.is_symlink() or not d.is_dir() or not DAY_RE.match(d.name):
            continue
        try:
            out.append((date.fromisoformat(d.name), d))
        except ValueError:
            continue
    return out


def plan(root: Path, days: int, today: date = None, source_days: dict = None) -> list:
    """[(day_dir, [files])] to delete. Per source: day older than its days (`source_days` overrides
    `days`; default DEFAULT_SOURCE_DAYS) and not the newest day."""
    today = today or datetime.now(timezone.utc).date()
    overrides = DEFAULT_SOURCE_DAYS if source_days is None else source_days
    out = []
    for src in sorted(p for p in Path(root).iterdir() if p.is_dir() and not p.is_symlink()) if Path(root).is_dir() else []:
        dirs = day_dirs(src)
        if not dirs:
            continue
        newest = max(d for d, _ in dirs)
        cutoff = today - timedelta(days=min(days, overrides.get(src.name, days)))
        for d, path in dirs:
            if d < cutoff and d != newest:
                files = [f for f in path.iterdir() if f.is_file() and not f.is_symlink() and f.name.endswith(EXTS)]
                out.append((path, files))
    return out


def free_gb(path: Path) -> float:
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def run(root: Path, days: int, dry_run: bool, min_free_gb: float, alert: bool = False, today=None,
        source_days: dict = None) -> int:
    if days < 1:
        raise ValueError("--days must be at least 1")
    freed = 0
    for path, files in plan(root, days, today, source_days):
        size = sum(f.stat().st_size for f in files)
        log.info("%s %s (%d files, %.2f GB)", "would delete" if dry_run else "deleting", path, len(files), size / 1e9)
        freed += size
        if dry_run:
            continue
        for f in files:
            f.unlink()
        try:
            path.rmdir()  # only succeeds when empty; unknown files are left alone
        except OSError:
            log.info("kept %s (contains other files)", path)
    log.info("%s %.2f GB", "would free" if dry_run else "freed", freed / 1e9)
    free = free_gb(root)
    log.info("free disk space: %.1f GB (threshold %.1f GB)", free, min_free_gb)
    if free < min_free_gb:
        msg = f"SWIM archive disk low: {free:.1f} GB free (threshold {min_free_gb:g} GB) after retention of {days} days."
        log.warning(msg)
        if alert:
            try:
                from alerts import send_alert
                send_alert("FlyRely SWIM disk low", msg)
            except Exception as e:  # alerts must never crash the job
                log.warning("alert failed: %s", type(e).__name__)
        return 2
    return 0


def main(argv=None) -> int:
    from swim.consumer import DATA_ROOT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DATA_ROOT)
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=10.0)
    ap.add_argument("--source-days", action="append", default=[], metavar="SRC=N",
                    help="keep N days for one source instead of --days (default stdds=1); repeatable")
    ap.add_argument("--alert", action="store_true", help="send an alert if disk space is low")
    args = ap.parse_args(argv)
    try:
        source_days = dict(DEFAULT_SOURCE_DAYS)
        for item in args.source_days:
            k, v = item.split("=", 1)
            source_days[k.strip()] = int(v)
    except ValueError:
        ap.error("--source-days must look like stdds=2")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.alert:  # alert channel settings live in <env file>.alerts or the env file
        try:
            from dotenv import load_dotenv
            from swim.consumer import ENV_PATH
            load_dotenv(ENV_PATH.with_name(ENV_PATH.name + ".alerts"))
            load_dotenv(ENV_PATH)
        except ImportError:
            pass
    return run(args.root, args.days, args.dry_run, args.min_free_gb, args.alert, source_days=source_days)


if __name__ == "__main__":
    sys.exit(main())
