"""Live accuracy scorer: logged predictions vs final BTS outcomes.

    python scripts/score_live.py                          # load outcomes from data/bts/parquet_v2, write reports
    python scripts/score_live.py --since 2026-08-01 --out-dir reports/accuracy
    python scripts/score_live.py --no-load --alert        # score what is already in `outcomes`, run alert checks

Uses DATABASE_URL (Postgres) when set, else SQLite (FLYRELY_PREDICTION_DB). Needs no network.
Scoring set: matched flights that were not cancelled (cancelled/diverted counts are reported separately),
label = departure delay >= 15 min (the training label). Each prediction row counts once.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import alerts  # noqa: E402
import outcomes  # noqa: E402
from prediction_log import PredictionStore  # noqa: E402

N_BINS = 10
RECENT_DAYS = 14


def auc(y: np.ndarray, p: np.ndarray):
    """Rank-based AUC with tie handling; None when only one class is present."""
    pos = y == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return None
    ranks = pd.Series(p).rank(method="average").to_numpy()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def reliability(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> list:
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            rows.append({"bin": f"{b / n_bins:.1f}-{(b + 1) / n_bins:.1f}", "n": int(m.sum()),
                         "mean_predicted": float(p[m].mean()), "observed_rate": float(y[m].mean())})
    return rows


def metrics(df: pd.DataFrame) -> dict:
    y = df["label_d15"].astype(float).to_numpy()
    p = df["delay_probability"].astype(float).to_numpy()
    n = len(df)
    if n == 0:
        return {"n": 0, "brier": None, "log_loss": None, "auc": None, "ece": None, "base_rate": None,
                "mean_predicted": None, "reliability": []}
    eps = 1e-6
    pc = np.clip(p, eps, 1 - eps)
    rel = reliability(y, p)
    ece = sum(r["n"] * abs(r["mean_predicted"] - r["observed_rate"]) for r in rel) / n
    return {"n": n, "brier": float(np.mean((p - y) ** 2)),
            "log_loss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
            "auc": auc(y, p), "ece": float(ece), "base_rate": float(y.mean()),
            "mean_predicted": float(p.mean()), "reliability": rel}


def build_report(matched: pd.DataFrame, now: datetime, recent_days: int = RECENT_DAYS) -> dict:
    """matched: output of outcomes.match_predictions."""
    scored = matched[matched["matched"] & (matched["cancelled"] == False) & matched["label_d15"].notna()].copy()  # noqa: E712
    scored["week"] = (scored["sched_dep_local"].dt.normalize()
                      - pd.to_timedelta(scored["sched_dep_local"].dt.day_of_week, unit="D")).dt.strftime("%Y-%m-%d")
    unmatched = matched[~matched["matched"]]
    report = {
        "generated_utc": now.isoformat(),
        "counts": {
            "predictions": int(len(matched)),
            "matched": int(matched["matched"].sum()),
            "unmatched": int(len(unmatched)),
            "scored": int(len(scored)),
            "cancelled": int((matched["matched"] & (matched["cancelled"] == True)).sum()),  # noqa: E712
            "diverted": int((matched["matched"] & (matched["diverted"] == True)).sum()),  # noqa: E712
        },
        "overall": metrics(scored),
        "by_model_horizon": [], "by_week": [], "recent": [], "recent_days": recent_days,
        "unmatched_sample": [],
    }
    if len(unmatched):
        # Not matched usually means BTS has not published that month yet
        newest = unmatched["sched_dep_local"].max()
        report["unmatched_by_month"] = {k: int(v) for k, v in
                                        unmatched["sched_dep_local"].dt.strftime("%Y-%m").value_counts().sort_index().items()}
        report["unmatched_sample"] = [
            {"prediction_id": str(r.prediction_id), "flight_id": r.flight_id, "origin": r.origin,
             "sched_dep_local": r.sched_dep_local.isoformat() if pd.notna(r.sched_dep_local) else None,
             "operating": f"{r.operating_airline or ''}{r.operating_number or ''}",
             "marketing": f"{r.marketing_airline or ''}{r.marketing_number or ''}"}
            for r in unmatched.sort_values("sched_dep_local").tail(20).itertuples()]
        report["unmatched_newest"] = newest.isoformat() if pd.notna(newest) else None
    for (mv, hz), g in scored.groupby(["model_version", "horizon"]):
        report["by_model_horizon"].append({"model_version": mv, "horizon": hz, **metrics(g)})
    for (wk, mv, hz), g in scored.groupby(["week", "model_version", "horizon"]):
        m = metrics(g)
        m.pop("reliability")
        report["by_week"].append({"week": wk, "model_version": mv, "horizon": hz, **m})
    cutoff = pd.Timestamp(now).tz_localize(None) - timedelta(days=recent_days)
    recent = scored[scored["sched_dep_local"] >= cutoff]
    for (mv, hz), g in recent.groupby(["model_version", "horizon"]):
        m = metrics(g)
        m.pop("reliability")
        report["recent"].append({"model_version": mv, "horizon": hz, **m})
    return report


def _f(v, nd=4):
    return "n/a" if v is None else f"{v:.{nd}f}"


def to_markdown(report: dict) -> str:
    c = report["counts"]
    L = [f"# FlyRely live accuracy report", "", f"Generated {report['generated_utc']}", "",
         "Scored against BTS final outcomes (label: departure delay >= 15 min; cancelled flights excluded from scoring).", "",
         "## Counts", "",
         f"- Predictions logged: {c['predictions']}", f"- Matched to a BTS record: {c['matched']}",
         f"- Scored (matched, not cancelled): {c['scored']}", f"- Cancelled: {c['cancelled']}, diverted: {c['diverted']}",
         f"- Unmatched: {c['unmatched']}", ""]
    o = report["overall"]
    L += ["## Overall", "", f"n={o['n']}, Brier {_f(o['brier'])}, log loss {_f(o['log_loss'])}, AUC {_f(o['auc'])}, "
          f"ECE {_f(o['ece'])}, observed delay rate {_f(o['base_rate'], 3)}, mean predicted {_f(o['mean_predicted'], 3)}", ""]
    L += ["## By model and horizon", "", "| model | horizon | n | Brier | log loss | AUC | ECE | base rate |", "|---|---|---|---|---|---|---|---|"]
    for g in report["by_model_horizon"]:
        L.append(f"| {g['model_version']} | {g['horizon']} | {g['n']} | {_f(g['brier'])} | {_f(g['log_loss'])} | "
                 f"{_f(g['auc'])} | {_f(g['ece'])} | {_f(g['base_rate'], 3)} |")
    L += ["", "## By week (week starting Monday, by scheduled departure)", "",
          "| week | model | horizon | n | Brier | AUC | ECE | base rate |", "|---|---|---|---|---|---|---|---|"]
    for g in report["by_week"]:
        L.append(f"| {g['week']} | {g['model_version']} | {g['horizon']} | {g['n']} | {_f(g['brier'])} | "
                 f"{_f(g['auc'])} | {_f(g['ece'])} | {_f(g['base_rate'], 3)} |")
    for g in report["by_model_horizon"]:
        L += ["", f"## Reliability: {g['model_version']} / {g['horizon']}", "",
              "When we said X, this share actually happened.", "",
              "| predicted bin | n | mean predicted | observed |", "|---|---|---|---|"]
        for r in g["reliability"]:
            L.append(f"| {r['bin']} | {r['n']} | {r['mean_predicted']:.1%} | {r['observed_rate']:.1%} |")
    L += ["", "## Unmatched predictions", ""]
    if c["unmatched"] == 0:
        L.append("None.")
    else:
        L.append(f"{c['unmatched']} predictions have no BTS record yet (BTS publishes about two months late; "
                 "also flight-number or codeshare mismatches). By month of scheduled departure:")
        L += ["", "| month | n |", "|---|---|"] + [f"| {k} | {v} |" for k, v in report.get("unmatched_by_month", {}).items()]
        L += ["", "Most recent unmatched (up to 20):", ""]
        L += [f"- {s['sched_dep_local']} {s['origin']} {s['operating'] or s['marketing']} ({s['flight_id']})"
              for s in report["unmatched_sample"]]
    return "\n".join(L) + "\n"


def run(args, store=None, now=None) -> dict:
    now = now or datetime.now(timezone.utc)
    store = store or PredictionStore(sqlite_path=args.sqlite)
    ostore = outcomes.OutcomeStore(store)
    preds = outcomes.read_predictions(store, since=args.since)
    if not args.no_load and len(preds):
        rows = outcomes.load_bts_outcomes(preds, Path(args.bts_dir), now=now)
        n = ostore.upsert(rows)
        logging.info(f"Loaded {n} BTS outcome rows from {args.bts_dir} ({ostore.backend})")
    matched = outcomes.match_predictions(preds, ostore.read()) if len(preds) else preds.assign(
        matched=pd.Series(dtype=bool), label_d15=pd.Series(dtype=float), cancelled=pd.Series(dtype=object),
        diverted=pd.Series(dtype=object), dep_delay_min=pd.Series(dtype=float), outcome_source=pd.Series(dtype=object))
    report = build_report(matched, now)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y-%m-%d")
    (out_dir / f"accuracy_{stamp}.json").write_text(json.dumps(report, indent=2, default=str))
    (out_dir / f"accuracy_{stamp}.md").write_text(to_markdown(report))
    logging.info(f"Wrote {out_dir}/accuracy_{stamp}.md and .json")
    if args.alert:
        hours = float(os.environ.get("FLYRELY_ALERT_VOLUME_HOURS", "24"))
        recent_n = int((preds["ts_utc"] >= pd.Timestamp(now) - timedelta(hours=hours)).sum()) if len(preds) else 0
        report["alerts"] = alerts.run_checks(report, recent_n)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bts-dir", default=str(ROOT / "data" / "bts" / "parquet_v2"))
    ap.add_argument("--out-dir", default=str(ROOT / "reports" / "accuracy"))
    ap.add_argument("--sqlite", default=None, help="SQLite path (ignored when DATABASE_URL is set)")
    ap.add_argument("--since", default=None, help="only predictions scheduled on/after this date (YYYY-MM-DD)")
    ap.add_argument("--no-load", action="store_true", help="skip loading BTS outcomes; score what is in the table")
    ap.add_argument("--alert", action="store_true", help="run alert checks (no-op log if no channel configured)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = run(args)
    print(json.dumps(report["counts"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
