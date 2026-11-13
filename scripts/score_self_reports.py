"""Score model predictions against what users told us ("Were you delayed?" answers, /flight-reports).

    python scripts/score_self_reports.py
    python scripts/score_self_reports.py --include-testers --out-dir reports/accuracy

Reads the prediction log (same store as score_live.py), the flight_reports SQLite file (opened read-only), and,
for the BTS agreement check, whatever is already in the `outcomes` table. It writes only the two report files:
never to `outcomes` or `predictions`, and it needs no network.

Join: report -> prediction on flight_id + origin + sched_dep_local (exact). Predictions are de-duplicated with
outcomes.dedupe_predictions (one per flight / model / lead-time bucket; rows logged after departure are dropped),
so a report whose only predictions came after departure is counted as unmatched.
Label: on_time = 0, late_* = 1 (left 15+ min late, matching label_d15). cancelled and did_not_fly are counted
but not scored. Rows from testers (tester=1) are left out unless --include-testers is given.
"""

import argparse
import json
import logging
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import outcomes  # noqa: E402
import score_live  # noqa: E402
from flight_reports import db_path as reports_db_path  # noqa: E402
from prediction_log import PredictionStore  # noqa: E402

ANSWERS = ["on_time", "late_15_44", "late_45_119", "late_120_plus", "cancelled", "did_not_fly"]
LABELS = {"on_time": 0, "late_15_44": 1, "late_45_119": 1, "late_120_plus": 1}
LEAD_ORDER = ["booking", "t-1d", "t-3h", "t-1h", "unknown-lead"]
REPORT_COLUMNS = ["user_id", "flight_id", "origin", "sched_dep_local", "answer", "tester"]


def _naive(series: pd.Series) -> pd.Series:
    local = pd.to_datetime(series, format="ISO8601")
    return local.dt.tz_localize(None) if local.dt.tz is not None else local


def read_reports(path: Path) -> pd.DataFrame:
    """Self-reports from the SQLite file, opened read-only. Empty if the file or table doesn't exist."""
    path = Path(path)
    rows = []
    if path.exists():
        try:
            with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=5)) as conn:
                rows = conn.execute(f"SELECT {', '.join(REPORT_COLUMNS)} FROM flight_reports").fetchall()
        except sqlite3.OperationalError as e:
            if "no such table" not in str(e):
                raise
    df = pd.DataFrame(rows, columns=REPORT_COLUMNS)
    df["sched_dep_local"] = _naive(df["sched_dep_local"])
    df["tester"] = df["tester"].fillna(0).astype(int)
    return df


def _empty_preds() -> pd.DataFrame:
    df = pd.DataFrame(columns=outcomes.PRED_COLUMNS)
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    df["sched_dep_local"] = pd.to_datetime(df["sched_dep_local"])
    df["delay_probability"] = df["delay_probability"].astype(float)
    return df


def read_predictions(store: PredictionStore) -> pd.DataFrame:
    # Do not let sqlite create an empty prediction DB just because we looked for it.
    if not store.database_url and not store.sqlite_path.exists():
        return _empty_preds()
    return outcomes.read_predictions(store)


def read_outcomes(store: PredictionStore) -> pd.DataFrame:
    """The `outcomes` table as it is; no DDL (OutcomeStore.read would create the table). Empty if absent."""
    if not store.database_url and not store.sqlite_path.exists():
        return pd.DataFrame(columns=outcomes.COLUMNS)
    try:
        return outcomes._query_df(store, f"SELECT {', '.join(outcomes.COLUMNS)} FROM outcomes", outcomes.COLUMNS)
    except Exception as e:
        if outcomes._is_missing_table(e):
            return pd.DataFrame(columns=outcomes.COLUMNS)
        raise


def agreement_with_bts(joined: pd.DataFrame, outcome_rows: pd.DataFrame) -> dict:
    """2x2 of self-report label vs BTS label_d15, one latest prediction per report."""
    out = {"n": 0, "agreement_rate": None,
           "matrix": {"self_on_time": {"bts_on_time": 0, "bts_late": 0},
                      "self_late": {"bts_on_time": 0, "bts_late": 0}}}
    if joined.empty or outcome_rows.empty:
        return out
    one = joined.sort_values("ts_utc").drop_duplicates("rid", keep="last")
    m = outcomes.match_predictions(one.drop(columns=["label"]), outcome_rows)
    m = m.merge(one[["rid", "label"]], on="rid")
    m = m[m["matched"] & (m["cancelled"] == False) & m["label_d15"].notna() & m["label"].notna()]  # noqa: E712
    for r in m.itertuples():
        out["matrix"]["self_late" if r.label == 1 else "self_on_time"]["bts_late" if int(r.label_d15) == 1 else "bts_on_time"] += 1
    mx = out["matrix"]
    out["n"] = sum(v for row in mx.values() for v in row.values())
    if out["n"]:
        out["agreement_rate"] = (mx["self_on_time"]["bts_on_time"] + mx["self_late"]["bts_late"]) / out["n"]
    return out


def build_report(reports: pd.DataFrame, preds: pd.DataFrame, outcome_rows: pd.DataFrame, now: datetime,
                 include_testers: bool = False, dedupe_stats: dict = None) -> dict:
    """reports: read_reports(); preds: outcomes.dedupe_predictions output; outcome_rows: read_outcomes()."""
    total_in_db = int(len(reports))
    testers_left_out = 0
    if not include_testers:
        testers_left_out = int((reports["tester"] == 1).sum())
        reports = reports[reports["tester"] != 1]
    reports = reports.reset_index(drop=True).assign(rid=lambda d: range(len(d)))
    reports["label"] = reports["answer"].map(LABELS)  # NaN for cancelled / did_not_fly

    joined = reports.merge(preds, on=["flight_id", "origin", "sched_dep_local"], how="inner")
    matched_ids = set(joined["rid"])
    scored = joined[joined["label"].notna()]
    n_scored_reports = int(scored["rid"].nunique())

    report = {
        "generated_utc": now.isoformat(),
        "include_testers": include_testers,
        "counts": {
            "reports_in_db": total_in_db,
            "testers_left_out": testers_left_out,
            "reports": int(len(reports)),
            "by_answer": {a: int((reports["answer"] == a).sum()) for a in ANSWERS},
            "not_scored": int(reports["answer"].isin(["cancelled", "did_not_fly"]).sum()),
            "matched_to_prediction": int(len(matched_ids)),
            "unmatched": int(len(reports) - len(matched_ids)),
            "scored_reports": n_scored_reports,
            "predictions_raw": (dedupe_stats or {}).get("raw", int(len(preds))),
            "dropped_after_departure": (dedupe_stats or {}).get("after_departure", 0),
            "predictions_deduped": int(len(preds)),
        },
        "by_model_lead_time": [],
        "agreement_with_bts": agreement_with_bts(joined[joined["label"].notna()], outcome_rows),
    }
    groups = [(mv, hz, g) for (mv, hz), g in scored.groupby(["model_version", "horizon"])]
    groups.sort(key=lambda t: (t[0], LEAD_ORDER.index(t[1]) if t[1] in LEAD_ORDER else 99))
    for mv, hz, g in groups:
        m = score_live.metrics(g.assign(label_d15=g["label"]))
        report["by_model_lead_time"].append({"model_version": mv, "lead_time": hz, **m})
    return report


def to_markdown(report: dict) -> str:
    c = report["counts"]
    f = score_live._f
    L = ["# FlyRely self-report accuracy", "", f"Generated {report['generated_utc']}", "",
         "Scored against what users said happened (on time = 0, left 15+ min late = 1). Cancelled and did-not-fly "
         "answers are counted but not scored. Self-reports are a rough signal, not ground truth; see the BTS "
         "agreement table below.", "",
         "## Counts", "",
         f"- Reports in database: {c['reports_in_db']}; testers left out: {c['testers_left_out']}"
         + (" (testers included)" if report["include_testers"] else ""),
         f"- Reports used: {c['reports']}",
         "- By answer: " + ", ".join(f"{a} {n}" for a, n in c["by_answer"].items()),
         f"- Not scored (cancelled / did not fly): {c['not_scored']}",
         f"- Matched to a logged prediction: {c['matched_to_prediction']}; unmatched: {c['unmatched']}",
         f"- Scored reports: {c['scored_reports']}",
         f"- Predictions: {c['predictions_raw']} raw, {c['dropped_after_departure']} dropped (logged after departure), "
         f"{c['predictions_deduped']} after keeping one per flight/model/lead-time bucket", "",
         "## By model and lead time", ""]
    if report["by_model_lead_time"]:
        L += ["| model | lead time | n | Brier | log loss | AUC | ECE | late rate | mean predicted |",
              "|---|---|---|---|---|---|---|---|---|"]
        for g in report["by_model_lead_time"]:
            L.append(f"| {g['model_version']} | {g['lead_time']} | {g['n']} | {f(g['brier'])} | {f(g['log_loss'])} | "
                     f"{f(g['auc'])} | {f(g['ece'])} | {f(g['base_rate'], 3)} | {f(g['mean_predicted'], 3)} |")
    else:
        L.append("No scored reports yet.")
    a = report["agreement_with_bts"]
    mx = a["matrix"]
    L += ["", "## Agreement with BTS", "",
          f"Reports whose flight also has a BTS outcome (not cancelled): {a['n']}. "
          f"Agreement rate: {'n/a' if a['agreement_rate'] is None else format(a['agreement_rate'], '.1%')}.", "",
          "| | BTS on time | BTS late (15+ min) |", "|---|---|---|",
          f"| Self-report on time | {mx['self_on_time']['bts_on_time']} | {mx['self_on_time']['bts_late']} |",
          f"| Self-report late | {mx['self_late']['bts_on_time']} | {mx['self_late']['bts_late']} |"]
    return "\n".join(L) + "\n"


def run(args, now=None) -> dict:
    now = now or datetime.now(timezone.utc)
    store = PredictionStore(sqlite_path=args.sqlite)
    reports = read_reports(Path(args.reports_db) if args.reports_db else reports_db_path())
    preds, dstats = outcomes.dedupe_predictions(read_predictions(store))
    report = build_report(reports, preds, read_outcomes(store), now, include_testers=args.include_testers,
                          dedupe_stats=dstats)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y-%m-%d")
    (out_dir / f"self_reports_{stamp}.json").write_text(json.dumps(report, indent=2, default=str))
    (out_dir / f"self_reports_{stamp}.md").write_text(to_markdown(report))
    logging.info(f"Wrote {out_dir}/self_reports_{stamp}.md and .json")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(ROOT / "reports" / "accuracy"))
    ap.add_argument("--sqlite", default=None, help="prediction SQLite path (ignored when DATABASE_URL is set)")
    ap.add_argument("--reports-db", default=None,
                    help="flight_reports SQLite path (default: FLYRELY_FLIGHT_REPORTS_DB or next to the prediction DB)")
    ap.add_argument("--include-testers", action="store_true", help="also score reports with tester=1")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = run(args)
    print(json.dumps(report["counts"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
