"""Live accuracy tracking: outcomes store, BTS loader, matching, scorer, alerts. Synthetic data, no network."""

import json
import logging
import os
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import alerts  # noqa: E402
from alerts import _post as REAL_ALERT_POST  # noqa: E402  (captured before conftest blocks alerts._post)
import outcomes  # noqa: E402
import score_live  # noqa: E402
from prediction_log import PredictionStore, build_record  # noqa: E402

NOW = datetime(2026, 9, 20, 12, tzinfo=timezone.utc)


def _pred(store, flight_no, origin, dep, p, op="DL", mk="DL", op_no=None, horizon="booking", model="m1", minutes=5000, ts=None):
    rec = build_record(
        flight_id=f"{mk}{flight_no}", origin=origin, dest="LAX", sched_dep_local=dep,
        marketing_airline=mk, marketing_number=str(flight_no), operating_airline=op,
        operating_number=str(op_no if op_no is not None else flight_no), delay_probability=p,
        risk_level="low", model_version=model, features={}, weather_available=True, horizon=horizon,
        now=ts or NOW)
    rec["sched_dep_utc"] = None
    rec["minutes_to_departure"] = minutes
    store.write(rec)


def _bts(tmp_path, rows):
    d = tmp_path / "bts"
    d.mkdir(exist_ok=True)
    df = pd.DataFrame(rows, columns=["FlightDate", "Reporting_Airline", "Marketing_Airline_Network",
                                     "Flight_Number_Reporting_Airline", "Flight_Number_Marketing_Airline",
                                     "Origin", "Dest", "CRSDepTime", "DepDelay", "Cancelled", "Diverted"])
    df["FlightDate"] = pd.to_datetime(df["FlightDate"])
    for ym, g in df.groupby(df["FlightDate"].dt.strftime("%Y-%m")):
        g.to_parquet(d / f"{ym}.parquet", index=False)
    return d


@pytest.fixture
def store(tmp_path):
    return PredictionStore(sqlite_path=str(tmp_path / "p.sqlite"), database_url="")


def test_norm_number():
    assert outcomes.norm_number("DL1234") == "1234"
    assert outcomes.norm_number("0123") == "123"
    assert outcomes.norm_number(1234.0) == "1234"
    assert outcomes.norm_number(None) is None
    assert outcomes.norm_number("ABC") is None


def test_loader_matches_label_and_cancel(store, tmp_path):
    _pred(store, 100, "ATL", datetime(2026, 8, 3, 9, 30), 0.3)           # delayed 20 -> label 1
    _pred(store, 101, "ATL", datetime(2026, 8, 3, 10, 0), 0.3)           # 14 min -> label 0
    _pred(store, 102, "ATL", datetime(2026, 8, 3, 11, 0), 0.3)           # cancelled
    _pred(store, 103, "ATL", datetime(2026, 8, 3, 12, 0), 0.3)           # not in BTS
    _pred(store, 200, "JFK", datetime(2026, 8, 4, 8, 0), 0.3, op="9E", mk="DL", op_no=5555)  # codeshare
    _pred(store, 300, "ORD", datetime(2026, 9, 4, 8, 0), 0.3)           # month with no BTS file
    bts = _bts(tmp_path, [
        ["2026-08-03", "DL", "DL", 100, 100, "ATL", "LAX", 930, 20.0, 0.0, 0.0],
        ["2026-08-03", "DL", "DL", 101, 101, "ATL", "LAX", 1000, 14.0, 0.0, 0.0],
        ["2026-08-03", "DL", "DL", 102, 102, "ATL", "LAX", 1100, np.nan, 1.0, 0.0],
        ["2026-08-04", "9E", "DL", 5555, 200, "JFK", "LAX", 800, 60.0, 0.0, 1.0],
        ["2026-08-03", "AA", "AA", 999, 999, "ATL", "LAX", 800, 5.0, 0.0, 0.0],  # nobody predicted this
    ])
    preds = outcomes.read_predictions(store)
    rows = outcomes.load_bts_outcomes(preds, bts, now=NOW)
    assert len(rows) == 4                      # AA999 excluded
    ostore = outcomes.OutcomeStore(store)
    assert ostore.upsert(rows) == 4
    m = outcomes.match_predictions(preds, ostore.read()).set_index("flight_id")
    assert m.loc["DL100", "label_d15"] == 1 and m.loc["DL101", "label_d15"] == 0
    assert bool(m.loc["DL102", "cancelled"]) and pd.isna(m.loc["DL102", "label_d15"])
    assert m.loc["DL200", "matched"] and m.loc["DL200", "label_d15"] == 1 and bool(m.loc["DL200", "diverted"])
    assert not m.loc["DL103", "matched"] and not m.loc["DL300", "matched"]


def test_marketing_fallback_when_operator_unknown(store, tmp_path):
    rec = build_record(flight_id="UA77", origin="DEN", dest="LAX", sched_dep_local=datetime(2026, 8, 5, 7, 0),
                       marketing_airline="UA", marketing_number="77", operating_airline=None, operating_number=None,
                       delay_probability=0.4, risk_level="low", model_version="m1", features={},
                       weather_available=False, now=NOW)
    store.write(rec)
    bts = _bts(tmp_path, [["2026-08-05", "OO", "UA", 4321, 77, "DEN", "LAX", 700, 30.0, 0.0, 0.0]])
    preds = outcomes.read_predictions(store)
    m = outcomes.match_predictions(preds, outcomes.load_bts_outcomes(preds, bts, now=NOW))
    assert bool(m.iloc[0]["matched"]) and m.iloc[0]["label_d15"] == 1


def test_upsert_idempotent_and_replaces(store, tmp_path):
    _pred(store, 100, "ATL", datetime(2026, 8, 3, 9, 30), 0.3)
    preds = outcomes.read_predictions(store)
    bts = _bts(tmp_path, [["2026-08-03", "DL", "DL", 100, 100, "ATL", "LAX", 930, 20.0, 0.0, 0.0]])
    rows = outcomes.load_bts_outcomes(preds, bts, now=NOW)
    ostore = outcomes.OutcomeStore(store)
    ostore.upsert(rows)
    rows["dep_delay_min"] = 5.0
    rows["label_d15"] = 0
    ostore.upsert(rows)
    got = ostore.read()
    assert len(got) == 1 and got.iloc[0]["label_d15"] == 0


def test_metrics_known_values():
    y = np.array([1, 0, 1, 0])
    df = pd.DataFrame({"label_d15": y, "delay_probability": [0.9, 0.1, 0.6, 0.4]})
    m = score_live.metrics(df)
    assert m["brier"] == pytest.approx((0.01 + 0.01 + 0.16 + 0.16) / 4)
    assert m["auc"] == 1.0
    assert score_live.auc(np.array([1, 0]), np.array([0.5, 0.5])) == 0.5
    assert score_live.auc(np.array([1, 1]), np.array([0.5, 0.6])) is None
    d2 = pd.DataFrame({"label_d15": [1] * 7 + [0] * 3, "delay_probability": [0.7] * 10})
    assert score_live.metrics(d2)["ece"] == pytest.approx(0.0)
    assert score_live.metrics(d2)["reliability"][0]["observed_rate"] == pytest.approx(0.7)
    assert score_live.metrics(df.iloc[0:0])["n"] == 0


def test_end_to_end_report_and_alerts(store, tmp_path, monkeypatch, caplog):
    rng = np.random.default_rng(0)
    bts_rows = []
    for i in range(60):
        day = 3 + (i % 14)
        dep = datetime(2026, 8, day, 6 + i % 12, 0)
        _pred(store, 1000 + i, "ATL", dep, 0.9, horizon="booking" if i % 2 else "t-3h")  # overconfident
        delay = 30.0 if rng.random() < 0.3 else 0.0
        bts_rows.append([f"2026-08-{day:02d}", "DL", "DL", 1000 + i, 1000 + i, "ATL", "LAX",
                         dep.hour * 100, delay, 0.0, 0.0])
    _pred(store, 5, "ATL", datetime(2026, 8, 20, 6), 0.5)  # unmatched
    bts = _bts(tmp_path, bts_rows)
    for k in list(os.environ):
        if k.startswith("FLYRELY_ALERT_"):
            monkeypatch.delenv(k)
    monkeypatch.setenv("FLYRELY_ALERT_MIN_N", "10")
    out = tmp_path / "out"
    args = Namespace(bts_dir=str(bts), out_dir=str(out), sqlite=None, since=None, no_load=False, alert=True)
    with caplog.at_level(logging.INFO):
        rep = score_live.run(args, store=store, now=datetime(2026, 11, 1, tzinfo=timezone.utc))
    assert rep["counts"]["predictions"] == 61 and rep["counts"]["unmatched"] == 1
    assert rep["counts"]["scored"] == 60
    assert {g["horizon"] for g in rep["by_model_horizon"]} == {"booking", "t-3h"}
    assert len({w["week"] for w in rep["by_week"]}) >= 2
    assert rep["unmatched_sample"][0]["flight_id"] == "DL5"
    md = (out / "accuracy_2026-11-01.md").read_text()
    assert "Reliability" in md and "Unmatched predictions" in md
    assert json.loads((out / "accuracy_2026-11-01.json").read_text())["counts"]["scored"] == 60
    assert rep["recent"] and rep["recent_anchor"].startswith("2026-08-1")   # anchored at newest scored flight, not now
    assert any("Calibration drift" in a for a in rep["alerts"])
    assert any("no channel configured" in r.message for r in caplog.records)


def test_empty_log_is_ok(store, tmp_path, monkeypatch):
    for k in list(os.environ):
        if k.startswith("FLYRELY_ALERT_"):
            monkeypatch.delenv(k)
    args = Namespace(bts_dir=str(tmp_path), out_dir=str(tmp_path / "o"), sqlite=None, since=None,
                     no_load=False, alert=True)
    rep = score_live.run(args, store=store, now=NOW)
    assert rep["counts"]["predictions"] == 0 and rep["overall"]["n"] == 0
    assert any("No predictions logged" in a for a in rep["alerts"])


def test_alert_noop_without_config():
    assert alerts.configured_channels({}) == []
    assert alerts.send_alert("s", "b", env={}) == []


def test_alert_channels(monkeypatch):
    posted = []

    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b""

    monkeypatch.setattr(alerts, "_post", REAL_ALERT_POST)  # exercise the real post, with urlopen faked below
    monkeypatch.setattr(alerts.urllib.request, "urlopen", lambda req, timeout=0: posted.append(req) or Resp())
    sent = {}

    class FakeSMTP:
        def __init__(self, host, port, timeout=0): sent["host"] = (host, port)
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def starttls(self, context=None): sent["tls"] = context is not None and context.verify_mode == alerts.ssl.CERT_REQUIRED
        def login(self, u, p): sent["login"] = u
        def send_message(self, msg): sent["to"] = msg["To"]

    monkeypatch.setattr(alerts.smtplib, "SMTP", FakeSMTP)
    env = {"FLYRELY_ALERT_DISCORD_WEBHOOK": "https://discord.example/hook", "FLYRELY_ALERT_NTFY_TOPIC": "t",
           "FLYRELY_ALERT_SMTP_HOST": "smtp.example", "FLYRELY_ALERT_SMTP_TO": "a@b.c",
           "FLYRELY_ALERT_SMTP_USER": "u", "FLYRELY_ALERT_SMTP_PASSWORD": "pw"}
    assert alerts.send_alert("subj", "body", env=env) == ["discord", "ntfy", "smtp"]
    assert posted[0].full_url == "https://discord.example/hook"
    assert posted[1].full_url == "https://ntfy.sh/t"
    assert sent["tls"] and sent["to"] == "a@b.c" and sent["login"] == "u"


def test_dedupe_buckets_and_after_departure(store):
    dep = datetime(2026, 8, 3, 9, 30)
    _pred(store, 100, "ATL", dep, 0.2, minutes=5000, ts=datetime(2026, 7, 30, tzinfo=timezone.utc))
    _pred(store, 100, "ATL", dep, 0.3, minutes=4000, ts=datetime(2026, 7, 31, tzinfo=timezone.utc))  # same bucket, later
    _pred(store, 100, "ATL", dep, 0.4, minutes=120, ts=datetime(2026, 8, 3, tzinfo=timezone.utc))      # t-3h
    _pred(store, 100, "ATL", dep, 0.9, minutes=-10, ts=datetime(2026, 8, 3, 10, tzinfo=timezone.utc))  # after departure
    _pred(store, 100, "ATL", dep, 0.5, minutes=None)                                                  # unknown lead
    p, stats = outcomes.dedupe_predictions(outcomes.read_predictions(store))
    assert stats == {"raw": 5, "after_departure": 1, "deduped": 3}
    got = dict(zip(p["horizon"], p["delay_probability"]))
    assert got == {"booking": 0.3, "t-3h": 0.4, "unknown-lead": 0.5}


def test_duplicate_bts_key_picks_closest_and_flags_mismatch(store, tmp_path):
    _pred(store, 100, "ATL", datetime(2026, 8, 3, 18, 0), 0.3)
    _pred(store, 101, "ATL", datetime(2026, 8, 3, 18, 0), 0.3)   # BTS has 06:00 only -> UTC/local style mismatch
    bts = _bts(tmp_path, [
        ["2026-08-03", "DL", "DL", 100, 100, "ATL", "LAX", 600, 0.0, 0.0, 0.0],
        ["2026-08-03", "DL", "DL", 100, 100, "ATL", "LAX", 1810, 45.0, 0.0, 0.0],
        ["2026-08-03", "DL", "DL", 101, 101, "ATL", "LAX", 600, 45.0, 0.0, 0.0]])
    preds = outcomes.read_predictions(store)
    rows = outcomes.load_bts_outcomes(preds, bts, now=NOW)
    assert len(rows) == 3                      # both legs of DL100 kept
    m = outcomes.match_predictions(preds, rows).set_index("flight_id")
    assert m.loc["DL100", "label_d15"] == 1 and m.loc["DL100", "match_status"] == "matched"
    assert m.loc["DL101", "match_status"] == "time_mismatch" and not m.loc["DL101", "matched"]


def test_ambiguous_tie_not_scored(store, tmp_path):
    _pred(store, 100, "ATL", datetime(2026, 8, 3, 12, 0), 0.3)
    bts = _bts(tmp_path, [
        ["2026-08-03", "DL", "DL", 100, 100, "ATL", "LAX", 1100, 0.0, 0.0, 0.0],
        ["2026-08-03", "DL", "DL", 100, 100, "ATL", "LAX", 1300, 45.0, 0.0, 0.0]])
    preds = outcomes.read_predictions(store)
    m = outcomes.match_predictions(preds, outcomes.load_bts_outcomes(preds, bts, now=NOW))
    assert m.iloc[0]["match_status"] == "ambiguous"


def test_nan_delay_is_on_time_and_null_marketing(store, tmp_path):
    _pred(store, 100, "ATL", datetime(2026, 8, 3, 9, 30), 0.3)
    d = _bts(tmp_path, [["2026-08-03", "DL", None, 100, 100, "ATL", "LAX", 930, np.nan, 0.0, 0.0]])
    rows = outcomes.load_bts_outcomes(outcomes.read_predictions(store), d, now=NOW)
    assert rows.iloc[0]["label_d15"] == 0 and rows.iloc[0]["marketing_airline"] is None


def test_read_predictions_missing_table(tmp_path):
    s = PredictionStore(sqlite_path=str(tmp_path / "none.sqlite"), database_url="")
    s.sqlite_path.parent.mkdir(exist_ok=True)
    assert outcomes.read_predictions(s).empty


def test_malformed_thresholds_fall_back(caplog):
    rep = {"recent_days": 14, "recent": [{"model_version": "m", "horizon": "b", "n": 600, "ece": 0.2, "brier": 0.2}]}
    msgs = alerts.evaluate(rep, 5, env={"FLYRELY_ALERT_ECE_MAX": "abc", "FLYRELY_ALERT_MIN_N": "x",
                                         "FLYRELY_ALERT_BASELINE_BRIER": "nope"})
    assert any("Calibration drift" in m for m in msgs)   # default 0.05 applied


@pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="needs TEST_DATABASE_URL (a throwaway Postgres)")
def test_postgres_round_trip(tmp_path):
    pg = PredictionStore(sqlite_path=str(tmp_path / "x.sqlite"), database_url=os.environ["TEST_DATABASE_URL"])
    _pred(pg, 7777, "ATL", datetime(2026, 8, 3, 9, 30), 0.3, op="ZZ", mk="ZZ")
    preds = outcomes.read_predictions(pg)
    bts = _bts(tmp_path, [["2026-08-03", "ZZ", "ZZ", 7777, 7777, "ATL", "LAX", 930, 20.0, 0.0, 0.0]])
    rows = outcomes.load_bts_outcomes(preds[preds["operating_airline"] == "ZZ"], bts, now=NOW)
    ostore = outcomes.OutcomeStore(pg)
    ostore.upsert(rows)
    ostore.upsert(rows)  # idempotent
    got = ostore.read()
    got = got[got["carrier"] == "ZZ"]
    assert len(got) == 1 and got.iloc[0]["label_d15"] == 1
    m = outcomes.match_predictions(preds[preds["operating_airline"] == "ZZ"], got)
    assert m["matched"].all()


def test_alert_failure_never_raises(monkeypatch, caplog):
    def boom(*a, **k):
        raise OSError("https://secret-webhook")
    monkeypatch.setattr(alerts, "_post", REAL_ALERT_POST)
    monkeypatch.setattr(alerts.urllib.request, "urlopen", boom)
    assert alerts.send_alert("s", "b", env={"FLYRELY_ALERT_DISCORD_WEBHOOK": "https://secret-webhook"}) == []
    assert "secret-webhook" not in caplog.text


def test_evaluate_thresholds():
    rep = {"recent_days": 14, "recent": [
        {"model_version": "m", "horizon": "booking", "n": 600, "ece": 0.01, "brier": 0.20},
        {"model_version": "m", "horizon": "t-3h", "n": 50, "ece": 0.5, "brier": 0.9}]}
    assert alerts.evaluate(rep, 5, env={}) == []                       # small n ignored, volume ok
    assert alerts.evaluate(rep, 5, env={"FLYRELY_ALERT_BASELINE_BRIER": "0.15"})  # Brier drift
    assert alerts.evaluate(rep, 0, env={})                             # zero volume
