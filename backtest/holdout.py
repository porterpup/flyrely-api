"""The one holdout policy (PROTOCOL_B5_1_HOLDOUT.md, section 8).

Months from LOCKED_FROM on are the protected holdout. Nothing reads, downloads or loads them unless the command
carries BOTH `--holdout YYYY-MM` and a prefix `FLYRELY_HOLDOUT_UNLOCK=YYYY-MM` naming the same month. The unlock opens
that one month only. The old value `1` is refused. Holdout files live only under data/holdout/YYYY-MM/, which no main
loader reads.

`fetch_bts.sh` mirrors this policy in POSIX sh (HOLDOUT_START there must equal LOCKED_FROM; a test checks it).
"""

import calendar
import os
import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOCKED_FROM = "2026-09"                                   # first protected month
LOCKED_FROM_TS = pd.Timestamp(f"{LOCKED_FROM}-01")
LAST_ALLOWED_DATE = (LOCKED_FROM_TS - pd.Timedelta(days=1)).date()   # 2026-08-31
UNLOCK_VAR = "FLYRELY_HOLDOUT_UNLOCK"
HOLDOUT_ROOT = ROOT / "data" / "holdout"
MODELS_DIR = HOLDOUT_ROOT / "models"
_MONTH = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


def is_month(s) -> bool:
    return isinstance(s, str) and bool(_MONTH.match(s))


def is_locked_month(ym: str) -> bool:
    return ym >= LOCKED_FROM          # YYYY-MM strings sort like dates


def is_locked_date(d) -> bool:
    return pd.Timestamp(d) >= LOCKED_FROM_TS


def holdout_dir(month: str) -> Path:
    return HOLDOUT_ROOT / month


def month_bounds(month: str) -> tuple[date, date]:
    """First and last day of the month."""
    y, m = int(month[:4]), int(month[5:])
    return date(y, m, 1), date(y, m, calendar.monthrange(y, m)[1])


def input_window(month: str) -> tuple[date, date]:
    """Dates fetched for feature inputs: the month plus one day (late evening cutoffs fall on the next UTC day)."""
    first, last = month_bounds(month)
    return first, last + timedelta(days=1)


def unlocked_month(holdout_arg: str | None = None, env=None) -> str | None:
    """The one unlocked month, or None (everything stays locked).

    A month is returned only when `holdout_arg` is that month AND FLYRELY_HOLDOUT_UNLOCK equals it. The old value `1`
    is refused with an explanation."""
    env = os.environ if env is None else env
    val = env.get(UNLOCK_VAR)
    if val == "1":
        raise SystemExit(f"{UNLOCK_VAR}=1 is no longer accepted: the unlock must name the month, for example "
                         f"{UNLOCK_VAR}={LOCKED_FROM} (and the command needs --holdout {LOCKED_FROM})")
    if holdout_arg is None or not val:
        return None
    if not (is_month(holdout_arg) and is_month(val)):
        raise SystemExit(f"invalid holdout month: --holdout '{holdout_arg}', {UNLOCK_VAR}='{val}' (expected YYYY-MM)")
    if val != holdout_arg:
        return None
    return holdout_arg if is_locked_month(holdout_arg) else None


def require_unlock(holdout_arg: str, env=None) -> str:
    """For commands that touch holdout data: validate `--holdout` and return the month, or exit."""
    if not is_month(holdout_arg):
        raise SystemExit(f"invalid --holdout '{holdout_arg}': expected YYYY-MM")
    if not is_locked_month(holdout_arg):
        raise SystemExit(f"--holdout {holdout_arg} is not a protected month (the holdout starts at {LOCKED_FROM})")
    if unlocked_month(holdout_arg, env) != holdout_arg:
        raise SystemExit(f"refusing --holdout {holdout_arg}: the holdout is locked. Prefix this one command with "
                         f"{UNLOCK_VAR}={holdout_arg} (never export it)")
    return holdout_arg


def check_end_date(end: str, holdout_arg: str | None = None, what: str = "--end") -> None:
    """Refuse a fetch end date inside the holdout, or outside the unlocked month's input window."""
    try:
        end_d = date.fromisoformat(end)
    except ValueError:
        raise SystemExit(f"invalid {what} '{end}': expected YYYY-MM-DD")
    if holdout_arg is None:
        if is_locked_date(end_d):
            raise SystemExit(f"refusing {what} {end}: dates after {LAST_ALLOWED_DATE} are the untouched holdout "
                             f"(use --holdout YYYY-MM with {UNLOCK_VAR}=YYYY-MM for the one-time download)")
        return
    require_unlock(holdout_arg)
    first, last = input_window(holdout_arg)
    if not first <= end_d <= last:
        raise SystemExit(f"refusing {what} {end}: --holdout {holdout_arg} allows {first} .. {last} only")


def locked_files(paths, stem_of=lambda p: p.stem) -> list:
    """Drop the holdout months from a list of month-named files (`YYYY-MM...`)."""
    return [p for p in paths if stem_of(p)[:7] < LOCKED_FROM]


def parquet_files_before_lock(parquet_dir: Path) -> list[Path]:
    files = sorted(parquet_dir.glob("2025-*.parquet")) + sorted(parquet_dir.glob("2026-*.parquet"))
    return locked_files(files)


def refuse_rows_outside(df: pd.DataFrame, month: str, col: str = "FlightDate") -> None:
    """A holdout BTS frame must hold flight dates of `month` only (no October rows, ever)."""
    first, last = month_bounds(month)
    d = pd.to_datetime(df[col])
    if len(d) and (d.min() < pd.Timestamp(first) or d.max() > pd.Timestamp(last)):
        raise RuntimeError(f"BTS rows outside {month}: {d.min().date()} .. {d.max().date()} (refused)")


def holdout_present(month: str) -> list[str]:
    """Names of holdout files already on disk for `month` (the freeze stage refuses if any)."""
    d = holdout_dir(month)
    return sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()) if d.exists() else []


def zip_month(name: str) -> str | None:
    """YYYY-MM of a BTS zip name (`..._2026_9.zip`), or None."""
    m = re.search(r"_(\d{4})_(\d{1,2})\.zip$", name)
    return f"{m.group(1)}-{int(m.group(2)):02d}" if m else None


def locked_files_in_main(root: Path = ROOT / "data") -> list[str]:
    """Month-named files of a locked month sitting in the main data folders (they must not be there)."""
    found = []
    for p in sorted((root / "bts" / "parquet_v2").glob("*.parquet")):
        if p.stem[:7] >= LOCKED_FROM:
            found.append(str(p))
    for p in sorted((root / "bts" / "marketing").glob("*.zip")):
        if (zip_month(p.name) or "0000-00") >= LOCKED_FROM:
            found.append(str(p))
    return found


AIRPORTS_FILE = ROOT / "backtest" / "results" / "h1_airports.json"


def frozen_airports(key: str, path: Path = None) -> list[str]:
    """The airport list frozen at stage 1 (`metar`, `weather` or `faa_covered`). Never recomputed for the holdout."""
    import json
    path = path or AIRPORTS_FILE
    if not path.exists():
        raise SystemExit(f"{path} missing: run `experiments_b5.py --holdout YYYY-MM --stage freeze` first")
    return list(json.loads(path.read_text())[key])
