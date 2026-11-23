"""Merge a spent holdout month into the main data folders (PROTOCOL_B5_1_HOLDOUT.md sections 6 and 8.6).

    venv/bin/python backtest/merge_holdout_month.py 2026-09 [--dry-run]

Only for a month that is NOT protected any more (holdout.is_locked_month is False): run it after the lock moved past
the month. It reads data/holdout/<month>/ and COPIES (never deletes) into the main folders the loaders read:

- bts:      the BTS zip -> data/bts/marketing/, <month>.parquet -> data/bts/parquet_v2/ (a new file; an existing one
            must be byte-identical, otherwise the script stops)
- metar:    <IATA>_<month>.csv rows -> data/metar/<IATA>_<year>.csv (appended)
- faa_adv:  list pages -> data/faa_adv/list/, programs.parquet rows -> data/faa_adv/programs_v2.parquet (appended)
- weather:  <IATA>.parquet rows -> data/weather/<IATA>.parquet (appended)

Only rows dated inside the month are taken. The holdout files also hold feature inputs for the next month's first
day (month plus one day); those belong to the next month, which is still locked, so they stay out of the main folders.
Appending is idempotent (a second run changes nothing) and every main file that is modified is first copied to
data/backup/holdout_merge_<month>/ (an existing backup is never overwritten). data/holdout/<month>/ is left in place.

It prints file names, row counts and date ranges only. It never computes a delay, arrival or cancellation rate.
"""

import argparse
import csv
import hashlib
import io
import shutil
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import holdout  # noqa: E402

ZIP_NAME = "On_Time_Marketing_Carrier_On_Time_Performance_Beginning_January_2018_{y}_{m}.zip"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Merger:
    def __init__(self, month: str, data: Path, dry_run: bool = False):
        if not holdout.is_month(month):
            raise SystemExit(f"invalid month '{month}': expected YYYY-MM")
        if holdout.is_locked_month(month):
            raise SystemExit(f"refusing {month}: it is a protected holdout month (the lock starts at "
                             f"{holdout.LOCKED_FROM}). Only spent months can be merged into the main folders")
        self.month, self.data, self.dry = month, data, dry_run
        self.src = data / "holdout" / month
        if not self.src.is_dir():
            raise SystemExit(f"{self.src} not found")
        self.first, self.last = holdout.month_bounds(month)
        self.next_first = self.last + timedelta(days=1)
        self.backup = data / "backup" / f"holdout_merge_{month}"
        self.changed: list[str] = []

    def say(self, msg: str):
        print(("[dry-run] " if self.dry else "") + msg, flush=True)

    def back_up(self, path: Path):
        dest = self.backup / path.relative_to(self.data)
        if dest.exists() or self.dry:
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)

    def copy_new(self, src: Path, dest: Path) -> bool:
        """Copy a file that must not exist yet (or must be identical)."""
        if dest.exists():
            if sha256(dest) != sha256(src):
                raise SystemExit(f"{dest} exists and differs from {src}: stopping, nothing is overwritten")
            return False
        if not self.dry:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        self.changed.append(str(dest.relative_to(self.data)))
        return True

    # ------------------------------------------------------------------ BTS
    def bts(self):
        y, m = int(self.month[:4]), int(self.month[5:])
        zips = sorted((self.src / "bts").glob("*.zip"))
        if [z.name for z in zips] != [ZIP_NAME.format(y=y, m=m)]:
            raise SystemExit(f"expected exactly {ZIP_NAME.format(y=y, m=m)} in {self.src / 'bts'}, found "
                             f"{[z.name for z in zips]}")
        pq = self.src / "bts" / f"{self.month}.parquet"
        d = pd.read_parquet(pq, columns=["FlightDate"])["FlightDate"]
        holdout.refuse_rows_outside(pd.DataFrame({"FlightDate": d}), self.month)
        n_pq = len(d)
        n1 = self.copy_new(zips[0], self.data / "bts" / "marketing" / zips[0].name)
        n2 = self.copy_new(pq, self.data / "bts" / "parquet_v2" / pq.name)
        self.say(f"bts: {zips[0].name} {'copied' if n1 else 'already there'}; parquet_v2/{pq.name} "
                 f"{'copied' if n2 else 'already there'}: {n_pq} rows, {d.min().date()} .. {d.max().date()}")

    # ------------------------------------------------------------------ METAR
    def metar(self):
        total_new = total_files = 0
        for hf in sorted((self.src / "metar").glob(f"*_{self.month}.csv")):
            iata = hf.name[:-len(f"_{self.month}.csv")]
            main = self.data / "metar" / f"{iata}_{self.month[:4]}.csv"
            h_lines = hf.read_text().splitlines(keepends=True)
            m_text = main.read_text() if main.exists() else None
            if m_text is None:
                m_header, m_keys = h_lines[0], set()
            else:
                m_lines = m_text.splitlines(keepends=True)
                m_header = m_lines[0]
                m_keys = {(r[0], r[1]) for r in csv.reader(io.StringIO("".join(m_lines[1:])))}
            if h_lines[0] != m_header:
                raise SystemExit(f"{hf.name}: header differs from {main.name}")
            new = []
            for line in h_lines[1:]:
                row = next(csv.reader([line]))
                if not self.first <= date.fromisoformat(row[1][:10]) <= self.last:
                    continue                                    # next month's first day: stays out
                key = (row[0], row[1])
                if key in m_keys:
                    continue
                m_keys.add(key)
                new.append(line if line.endswith("\n") else line + "\n")
            total_files += 1
            total_new += len(new)
            if not new:
                continue
            if m_text is not None:
                self.back_up(main)
            if not self.dry:
                body = m_text if m_text is not None else m_header
                if not body.endswith("\n"):
                    body += "\n"
                main.write_text(body + "".join(new))
            self.changed.append(str(main.relative_to(self.data)))
        self.say(f"metar: {total_files} airport files read, {total_new} rows appended")

    # ------------------------------------------------------------------ FAA advisories
    def faa(self):
        lists = sorted((self.src / "faa_adv" / "list").glob("*.html"))
        month_pages = [p for p in lists if str(self.first) <= p.stem <= str(self.last)]
        new_pages = sum(self.copy_new(p, self.data / "faa_adv" / "list" / p.name) for p in month_pages)
        first, last = (month_pages[0].stem, month_pages[-1].stem) if month_pages else ("-", "-")
        self.say(f"faa_adv: {len(month_pages)} list pages in the month ({first} .. {last}), {new_pages} newly copied")
        hp = pd.read_parquet(self.src / "faa_adv" / "programs.parquet")
        ld = hp["list_date"].astype(str)
        add = hp[(ld >= str(self.first)) & (ld <= str(self.last))]
        target = self.data / "faa_adv" / "programs_v2.parquet"
        main = pd.read_parquet(target)
        if list(main.columns) != list(add.columns):
            raise SystemExit("programs_v2.parquet columns differ from the holdout programs file")
        present = main["list_date"].astype(str).between(str(self.first), str(self.last)).any()
        if present:
            self.say(f"faa_adv: programs_v2.parquet already has {self.month} rows; nothing appended")
            return
        self.back_up(target)
        out = pd.concat([main, add.astype(main.dtypes.to_dict())], ignore_index=True)
        if not self.dry:
            out.to_parquet(target)
        self.changed.append(str(target.relative_to(self.data)))
        self.say(f"faa_adv: programs_v2.parquet {len(main)} -> {len(out)} rows (+{len(add)}), list dates "
                 f"{add['list_date'].min()} .. {add['list_date'].max()}")

    # ------------------------------------------------------------------ weather
    def weather(self):
        files = sorted((self.src / "weather").glob("*.parquet"))
        total_new = 0
        for hf in files:
            main = self.data / "weather" / hf.name
            h = pd.read_parquet(hf)
            h = h[(h["utc_hour"] >= pd.Timestamp(self.first)) & (h["utc_hour"] < pd.Timestamp(self.next_first))]
            m = pd.read_parquet(main) if main.exists() else h.iloc[0:0]
            if list(m.columns) != list(h.columns):
                raise SystemExit(f"{hf.name}: columns differ from the main file")
            seen = set(zip(m["airport"], m["utc_hour"]))
            add = h[[k not in seen for k in zip(h["airport"], h["utc_hour"])]].drop_duplicates(["airport", "utc_hour"])
            total_new += len(add)
            if add.empty:
                continue
            if main.exists():
                self.back_up(main)
            out = pd.concat([m, add.astype(m.dtypes.to_dict())], ignore_index=True)
            if not self.dry:
                out.to_parquet(main)
            self.changed.append(str(main.relative_to(self.data)))
        self.say(f"weather: {len(files)} airport files read, {total_new} hourly rows appended")

    def run(self):
        self.bts()
        self.metar()
        self.faa()
        self.weather()
        self.say(f"{len(self.changed)} main files created or changed; backups in {self.backup}; "
                 f"{self.src} left untouched")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("month", help="spent month, YYYY-MM")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    Merger(a.month, holdout.ROOT / "data", a.dry_run).run()


if __name__ == "__main__":
    main()
