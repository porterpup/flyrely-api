"""scripts/score_self_reports.py: join, dedupe, tester filter, unscored answers, BTS agreement, empty DB.
Temporary SQLite files and fixed dates only (no dependence on today's date, no network)."""
import json
import sqlite3
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import flight_reports  # noqa: E402
import outcomes  # noqa: E402
import score_self_reports as ssr  # noqa: E402
from prediction_log import PredictionStore, build_record  # noqa: E402

NOW = datetime(2026, 9, 20, 12, tzinfo=timezone.utc)
DEP = datetime(2026, 8, 3, 9, 30)


@pytest.fixture
def paths(tmp_path):
    return {"pred": tmp_path / "p.sqlite", "reports": tmp_path / "fr.sqlite", "out": tmp_path / "out"}


@pytest.fixture
def store(paths):
    return PredictionStore(sqlite_path=str(paths["pred"]), database_url="")


def pred(store, number, origin="ATL", dep=DEP, p=0.3, minutes=5000, model="m1", ts=NOW):
    rec = build_record(
        flight_id=f"DL{number}", origin=origin, dest="LAX", sched_dep_local=dep, marketing_airline="DL",
        marketing_number=str(number), operating_airline="DL", operating_number=str(number), delay_probability=p,
        risk_level="low", model_version=model, features={}, weather_available=True, horizon="booking", now=ts)
    rec["sched_dep_utc"] = None
    rec["minutes_to_departure"] = minutes
    store.write(rec)


def report(paths, number, answer, origin="ATL", dep=DEP, tester=0, user="u1"):
    with sqlite3.connect(paths["reports"]) as c:
        c.executescript(flight_reports._DDL)
        c.execute("INSERT INTO flight_reports VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                  (user, f"DL{number}", "DL", str(number), origin, "LAX", dep.isoformat(), answer, tester,
                   NOW.isoformat(), NOW.isoformat()))


def bts_row(number, delay, cancelled=False, origin="ATL", dep=DEP):
    return {"carrier": "DL", "flight_number": str(number), "origin": origin, "sched_dep_date": dep.date().isoformat(),
            "source": "bts", "marketing_airline": "DL", "marketing_number": str(number), "dest": "LAX",
            "sched_dep_local": dep.isoformat(), "dep_delay_min": None if cancelled else float(delay),
            "cancelled": cancelled, "diverted": False,
            "label_d15": None if cancelled else int(delay >= 15), "label_quality": "final",
            "fetched_utc": NOW.isoformat()}


def args(paths, **kw):
    return Namespace(sqlite=str(paths["pred"]), reports_db=str(paths["reports"]), out_dir=str(paths["out"]),
                     include_testers=kw.get("include_testers", False))


def run(paths, **kw):
    return ssr.run(args(paths, **kw), now=NOW)


def tables(path):
    with sqlite3.connect(path) as c:
        return {r[0]: c.execute(f"SELECT COUNT(*) FROM {r[0]}").fetchone()[0]
                for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def test_join_needs_flight_origin_and_date(store, paths):
    pred(store, 100)
    report(paths, 100, "on_time", user="a")                                       # exact
    report(paths, 100, "late_15_44", origin="JFK", user="b")                      # origin differs
    report(paths, 100, "late_45_119", dep=datetime(2026, 8, 4, 9, 30), user="c")  # date differs
    report(paths, 999, "on_time", user="d")                                       # no such flight
    r = run(paths)
    assert r["counts"]["reports"] == 4
    assert r["counts"]["matched_to_prediction"] == 1
    assert r["counts"]["unmatched"] == 3
    assert r["counts"]["scored_reports"] == 1


def test_post_departure_predictions_are_dropped(store, paths):
    pred(store, 100, minutes=-10)
    pred(store, 101, minutes=-10)
    pred(store, 101, minutes=120, ts=datetime(2026, 8, 3, 7, 30, tzinfo=timezone.utc))
    report(paths, 100, "on_time", user="a")
    report(paths, 101, "late_15_44", user="b")
    r = run(paths)
    assert r["counts"]["dropped_after_departure"] == 2
    assert r["counts"]["unmatched"] == 1  # only 100 had nothing but post-departure rows
    assert r["counts"]["scored_reports"] == 1
    assert [g["lead_time"] for g in r["by_model_lead_time"]] == ["t-3h"]


def test_testers_excluded_unless_asked(store, paths):
    pred(store, 100)
    report(paths, 100, "on_time", user="real")
    report(paths, 100, "late_15_44", user="tester", tester=1)
    r = run(paths)
    assert r["counts"]["reports"] == 1 and r["counts"]["testers_left_out"] == 1
    assert r["by_model_lead_time"][0]["n"] == 1
    r = run(paths, include_testers=True)
    assert r["counts"]["reports"] == 2 and r["counts"]["testers_left_out"] == 0
    assert r["by_model_lead_time"][0]["n"] == 2


def test_cancelled_and_did_not_fly_counted_not_scored(store, paths):
    for n in (100, 101, 102, 103):
        pred(store, n)
    report(paths, 100, "cancelled", user="a")
    report(paths, 101, "did_not_fly", user="b")
    report(paths, 102, "on_time", user="c")
    report(paths, 103, "late_120_plus", user="d")
    r = run(paths)
    c = r["counts"]
    assert c["by_answer"]["cancelled"] == 1 and c["by_answer"]["did_not_fly"] == 1
    assert c["not_scored"] == 2 and c["scored_reports"] == 2 and c["matched_to_prediction"] == 4
    g = r["by_model_lead_time"][0]
    assert g["n"] == 2 and g["base_rate"] == 0.5


def test_metrics_use_label_zero_for_on_time_one_for_late(store, paths):
    pred(store, 100, p=0.2)
    pred(store, 101, p=0.8)
    report(paths, 100, "on_time", user="a")
    report(paths, 101, "late_15_44", user="b")
    g = run(paths)["by_model_lead_time"][0]
    assert g["auc"] == 1.0
    assert g["brier"] == pytest.approx(((0.2 - 0) ** 2 + (0.8 - 1) ** 2) / 2)


def test_agreement_matrix(store, paths):
    flights = {  # number: (self answer, BTS delay or None for cancelled, whether BTS has the flight)
        100: ("on_time", 5, True),         # agree on time
        101: ("on_time", 0, True),         # agree on time
        102: ("late_45_119", 50, True),    # agree late
        103: ("late_15_44", 3, True),      # self late, BTS on time
        104: ("on_time", 40, True),        # self on time, BTS late
        105: ("late_15_44", None, True),   # BTS cancelled: not in the table
        106: ("late_15_44", 0, False),     # no BTS record
    }
    outs = []
    for n, (ans, delay, has) in flights.items():
        pred(store, n)
        report(paths, n, ans, user=f"u{n}")
        if has:
            outs.append(bts_row(n, delay or 0, cancelled=delay is None))
    outcomes.OutcomeStore(store).upsert(pd.DataFrame(outs))
    a = run(paths)["agreement_with_bts"]
    assert a["matrix"] == {"self_on_time": {"bts_on_time": 2, "bts_late": 1},
                           "self_late": {"bts_on_time": 1, "bts_late": 1}}
    assert a["n"] == 5 and a["agreement_rate"] == pytest.approx(3 / 5)


def test_empty_databases_still_work(paths):
    r = run(paths)  # neither database exists
    assert r["counts"]["reports"] == 0 and r["counts"]["unmatched"] == 0
    assert r["by_model_lead_time"] == [] and r["agreement_with_bts"]["agreement_rate"] is None
    assert not paths["pred"].exists() and not paths["reports"].exists()
    stamp = NOW.strftime("%Y-%m-%d")
    assert json.loads((paths["out"] / f"self_reports_{stamp}.json").read_text())["counts"]["reports"] == 0
    assert "No scored reports yet." in (paths["out"] / f"self_reports_{stamp}.md").read_text()


def test_reports_without_predictions_are_all_unmatched(paths):
    report(paths, 100, "on_time")
    r = run(paths)
    assert r["counts"]["unmatched"] == 1 and r["by_model_lead_time"] == []


def test_never_writes_to_predictions_or_outcomes(store, paths):
    pred(store, 100)
    report(paths, 100, "on_time")
    before = tables(paths["pred"])
    reports_before = tables(paths["reports"])
    run(paths)
    assert tables(paths["pred"]) == before and "outcomes" not in tables(paths["pred"])
    assert tables(paths["reports"]) == reports_before
