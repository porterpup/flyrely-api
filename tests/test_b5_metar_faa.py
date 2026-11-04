"""B5 features: observed METAR and FAA GS/GDP programs. Leakage, parsing and training/serving parity."""

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backtest"))

import fetch_faa_adv as ff  # noqa: E402
from flyrely_features import faa, metar  # noqa: E402
from flyrely_features._common import DAY0, SHIFT  # noqa: E402

AIRPORTS = pd.Index(["ATL", "JFK", "ORD"])


def utc_min(s: str) -> float:
    return float((pd.Timestamp(s) - DAY0) / pd.Timedelta(minutes=1))


def flights(dep_utc, arr_utc, o=0, d=1):
    """Flight arrays with utc_shift 0 so s_dep = UTC minutes + SHIFT."""
    n = len(dep_utc)
    return {"s_dep": np.array([utc_min(x) + SHIFT for x in dep_utc]),
            "s_arr": np.array([utc_min(x) + SHIFT for x in arr_utc]),
            "utc_shift": np.zeros(n), "o_id": np.full(n, o), "d_id": np.full(n, d)}


# ---------------------------------------------------------------- METAR

def raw_metar(rows):
    cols = ["station", "valid", "vsby", "sknt", "gust", "skyc1", "skyc2", "skyc3", "skyc4",
            "skyl1", "skyl2", "skyl3", "skyl4", "wxcodes", "p01i"]
    return pd.DataFrame([{**{c: "" for c in cols}, **r} for r in rows])


def test_derive_flags():
    raw = raw_metar([
        {"station": "ATL", "valid": "2024-01-01 10:00", "vsby": "10.00", "sknt": "5", "skyc1": "FEW", "skyl1": "2500"},
        {"station": "ATL", "valid": "2024-01-01 11:00", "vsby": "2.00", "sknt": "18", "gust": "30", "skyc1": "OVC",
         "skyl1": "700", "wxcodes": "+TSRA BR"},
        {"station": "ATL", "valid": "2024-01-01 12:00", "vsby": "0.25", "skyc1": "VV", "skyl1": "100", "wxcodes": "FZFG -SN"},
    ])
    d = metar.derive(raw)
    assert list(d["ceiling_ft"]) == [metar.NO_CEILING, 700, 100]       # FEW is not a ceiling
    assert list(d["ifr"]) == [0, 1, 1]
    assert list(d["thunder"]) == [0, 1, 0] and list(d["precip"]) == [0, 1, 1]
    assert list(d["snow"]) == [0, 0, 1] and list(d["freezing"]) == [0, 0, 1] and list(d["fog"]) == [0, 0, 1]
    assert d["gust_kt"].tolist() == [0, 30, 0]


def test_metar_uses_only_reports_at_or_before_cutoff():
    raw = raw_metar([
        {"station": "ATL", "valid": "2024-01-01 09:52", "vsby": "10", "skyc1": "CLR"},
        {"station": "ATL", "valid": "2024-01-01 10:30", "vsby": "1", "skyc1": "OVC", "skyl1": "300"},   # after the cutoff
    ])
    M = metar.Metar(AIRPORTS, raw=raw)
    A = flights(["2024-01-01 11:00"], ["2024-01-01 13:00"], o=0, d=1)
    f = metar.serve_row(M, A, 60)                                              # cutoff 10:00
    assert f["mt_o_vis_mi"] == 10 and f["mt_o_ceiling_ft"] == metar.NO_CEILING
    assert f["mt_o_age_min"] == 8
    f2 = metar.serve_row(M, flights(["2024-01-01 11:30"], ["2024-01-01 13:00"]), 60)   # cutoff 10:30 inclusive
    assert f2["mt_o_vis_mi"] == 1 and f2["mt_o_ceiling_ft"] == 300
    # later reports never change an earlier answer
    raw2 = pd.concat([raw, raw_metar([{"station": "ATL", "valid": "2024-01-01 10:45", "vsby": "0.5"}])])
    f3 = metar.serve_row(metar.Metar(AIRPORTS, raw=raw2), A, 60)
    assert {k: v for k, v in f3.items() if v == v} == {k: v for k, v in f.items() if v == v}


def test_metar_stale_or_uncovered_is_nan_not_calm():
    raw = raw_metar([{"station": "ATL", "valid": "2024-01-01 00:00", "vsby": "10", "skyc1": "CLR"}])
    M = metar.Metar(AIRPORTS, raw=raw)
    f = metar.serve_row(M, flights(["2024-01-01 12:00"], ["2024-01-01 14:00"]), 60)
    assert np.isnan(f["mt_o_vis_mi"]) and np.isnan(f["mt_d_vis_mi"])        # >3 h old / no station


def test_metar_serve_parity_with_vectorised():
    rng = np.random.default_rng(0)
    rows = [{"station": str(AIRPORTS[i % 3]), "valid": str(pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=int(m))),
             "vsby": str(rng.integers(1, 10)), "sknt": str(rng.integers(0, 30)), "skyc1": "BKN",
             "skyl1": str(rng.integers(2, 80) * 100), "wxcodes": ["", "RA", "TSRA"][rng.integers(0, 3)]}
            for i, m in enumerate(np.sort(rng.integers(0, 3000, 400)))]
    M = metar.Metar(AIRPORTS, raw=raw_metar(rows))
    n = 60
    dep = [str(pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=int(m))) for m in rng.integers(300, 3200, n)]
    A = flights(dep, dep)
    A["o_id"], A["d_id"] = rng.integers(0, 3, n), rng.integers(0, 3, n)
    full = M.features(A, np.arange(n), 180)
    for i in range(n):
        q = {k: v[i:i + 1] for k, v in A.items()}
        s = metar.serve_row(M, q, 180)
        for k, v in s.items():
            assert (np.isnan(v) and np.isnan(full[k][i])) or v == full[k][i], k


def test_from_aviationweather_maps_live_fields():
    rec = [{"icaoId": "KATL", "obsTime": 1704103200, "visib": "10+", "wspd": 8, "wgst": 20, "wxString": "-RA",
            "clouds": [{"cover": "BKN", "base": 900}]}]
    d = metar.derive(metar.from_aviationweather(rec))
    assert d["ceiling_ft"][0] == 900 and d["ifr"][0] == 1 and d["gust_kt"][0] == 20 and d["precip"][0] == 1
    assert d["vis_mi"][0] == 10


# ---------------------------------------------------------------- FAA

LIST_HTML = """
<table><tr><td><a href="/adv/adv_otherdis?adv_date=03152025&advn=149">149 </a></td><td>MSY/ZHU</td><td>03/15/25</td>
<td>CDM GS CNX</td><td>03/15/25 22:48</td></tr>
<tr><td><a href="/adv/adv_otherdis?adv_date=03152025&advn=148">148 </a></td><td>MSY/ZHU</td><td>03/15/25</td>
<td>CDM GROUND STOP</td><td>03/15/25 22:12</td></tr>
<tr><td><a href="/adv/adv_otherdis?adv_date=03152025&advn=110">110 </a></td><td>EWR/ZNY</td><td>03/15/25</td>
<td>CDM PROPOSED GROUND DELAY PROGRAM</td><td>03/15/25 17:22</td></tr>
<tr><td><a href="/adv/adv_otherdis?adv_date=03152025&advn=98">098 </a></td><td>DCA/ZDC</td><td>03/15/25</td>
<td>WASHINGTON NATIONAL AIRPORT ARRIVAL DELAYS</td><td>03/15/25 16:07</td></tr></table>"""


def test_parse_list_classifies_and_skips_non_programs():
    rows = ff.parse_list(LIST_HTML)
    assert [(r["advn"], r["element"], r["kind"], r["status"]) for r in rows] == [
        (149, "MSY", "GS", "cancel"), (148, "MSY", "GS", "active"), (110, "EWR", "GDP", "proposed")]
    assert rows[1]["sent_utc"] == datetime(2025, 3, 15, 22, 12, tzinfo=timezone.utc)


def test_parse_detail_periods_including_month_rollover():
    sent = datetime(2025, 3, 31, 22, 12, tzinfo=timezone.utc)
    d = ff.parse_detail("CTL ELEMENT: MSY\nGROUND STOP PERIOD: 31/2156Z - 01/0015Z\n", sent)
    assert d["period_start"] == pd.Timestamp("2025-03-31 21:56", tz="UTC")
    assert d["period_end"] == pd.Timestamp("2025-04-01 00:15", tz="UTC")
    g = ff.parse_detail("CUMULATIVE PROGRAM PERIOD: 15/1100Z - 16/0459Z", datetime(2025, 3, 15, 17, 38, tzinfo=timezone.utc))
    assert g["period_end"] == pd.Timestamp("2025-03-16 04:59", tz="UTC")
    assert pd.isna(ff.parse_detail("no period here", sent)["period_start"])


def programs():
    t = pd.Timestamp
    return pd.DataFrame([
        # JFK: proposed GDP at 10:00, activated at 11:00 (replaces the proposal), cancelled at 14:00
        {"element": "JFK", "kind": "GDP", "status": "proposed", "sent_utc": t("2024-01-01 10:00"),
         "period_start": t("2024-01-01 12:00"), "period_end": t("2024-01-02 00:00")},
        {"element": "JFK", "kind": "GDP", "status": "active", "sent_utc": t("2024-01-01 11:00"),
         "period_start": t("2024-01-01 12:00"), "period_end": t("2024-01-02 00:00")},
        {"element": "JFK", "kind": "GDP", "status": "cancel", "sent_utc": t("2024-01-01 14:00"),
         "period_start": pd.NaT, "period_end": pd.NaT},
        # ATL: a ground stop 09:00-10:00
        {"element": "ATL", "kind": "GS", "status": "active", "sent_utc": t("2024-01-01 09:00"),
         "period_start": t("2024-01-01 09:00"), "period_end": t("2024-01-01 10:00")},
    ])


def feats(P, dep, arr, h, o=0, d=1):
    return faa.serve_row(P, flights([dep], [arr], o=o, d=d), h)


def test_faa_proposed_then_active_then_cancelled():
    P = faa.Programs(programs(), AIRPORTS, list(AIRPORTS))
    prop = feats(P, "2024-01-01 11:30", "2024-01-01 13:00", 60)        # cutoff 10:30
    assert (prop["faa_d_gdp_prop"], prop["faa_d_gdp_cover"]) == (1, 0)
    act = feats(P, "2024-01-01 12:30", "2024-01-01 14:00", 60)         # cutoff 11:30
    assert (act["faa_d_gdp_prop"], act["faa_d_gdp_cover"]) == (0, 1)
    assert act["faa_d_gdp_now"] == 0                                     # period starts 12:00, not yet "now" at 11:30
    live = feats(P, "2024-01-01 13:30", "2024-01-01 15:00", 60)        # cutoff 12:30
    assert live["faa_d_gdp_now"] == 1
    gone = feats(P, "2024-01-01 15:30", "2024-01-01 17:00", 60)        # cutoff 14:30: cancelled at 14:00
    assert (gone["faa_d_gdp_now"], gone["faa_d_gdp_cover"], gone["faa_d_gdp_prop"]) == (0, 0, 0)


def test_faa_ground_stop_window_and_origin_side():
    P = faa.Programs(programs(), AIRPORTS, list(AIRPORTS))
    on = feats(P, "2024-01-01 10:00", "2024-01-01 12:00", 60, o=0, d=2)   # cutoff 09:00, GS at origin ATL
    assert on["faa_o_gs_now"] == 1 and on["faa_d_gs_now"] == 0
    off = feats(P, "2024-01-01 11:30", "2024-01-01 12:00", 60, o=0, d=2)  # cutoff 10:30, GS ended at 10:00
    assert off["faa_o_gs_now"] == 0


def test_faa_uncovered_airport_is_nan_not_zero():
    P = faa.Programs(programs(), AIRPORTS, ["ATL"])
    f = feats(P, "2024-01-01 12:00", "2024-01-01 14:00", 60, o=0, d=1)
    assert f["faa_o_gs_now"] == 0 and np.isnan(f["faa_d_gs_now"])


def test_faa_serve_parity_with_vectorised():
    P = faa.Programs(programs(), AIRPORTS, list(AIRPORTS))
    rng = np.random.default_rng(1)
    n = 80
    dep = [str(pd.Timestamp("2024-01-01 06:00") + pd.Timedelta(minutes=int(m))) for m in rng.integers(0, 900, n)]
    arr = [str(pd.Timestamp(x) + pd.Timedelta(minutes=120)) for x in dep]
    A = flights(dep, arr)
    A["o_id"], A["d_id"] = rng.integers(0, 3, n), rng.integers(0, 3, n)
    full = P.features(A, np.arange(n), 180)
    for i in range(n):
        s = faa.serve_row(P, {k: v[i:i + 1] for k, v in A.items()}, 180)
        for k, v in s.items():
            assert (np.isnan(v) and np.isnan(full[k][i])) or v == full[k][i], k
    assert any(full[k].max() > 0 for k in full)           # the fixture actually exercises the lookup


def test_faa_missing_period_uses_documented_default():
    p = pd.DataFrame([{"element": "ATL", "kind": "GS", "status": "active", "sent_utc": pd.Timestamp("2024-01-01 09:00"),
                       "period_start": pd.NaT, "period_end": pd.NaT}])
    P = faa.Programs(p, AIRPORTS, ["ATL"])
    assert P.chains["GS"].n_missing_period == 1
    assert feats(P, "2024-01-01 11:30", "2024-01-01 13:00", 60, o=0)["faa_o_gs_now"] == 1    # 10:30 < 09:00+180
    assert feats(P, "2024-01-01 13:30", "2024-01-01 15:00", 60, o=0)["faa_o_gs_now"] == 0    # 12:30 > 12:00


def test_holdout_lock_in_fetchers(monkeypatch):
    import fetch_metar
    monkeypatch.delenv("FLYRELY_HOLDOUT_UNLOCK", raising=False)
    for mod in (ff, fetch_metar):
        with pytest.raises(SystemExit):
            mod.check_holdout("2026-09-01")
        mod.check_holdout("2026-08-31")


def test_decision_rule_is_the_preregistered_one():
    import experiments_b5 as e
    ok = {"ci95": [-0.002, -0.0005]}
    assert e.decide(ok, 0.010, 0.008)["adopt"]                       # ECE +0.002 is within tolerance
    assert not e.decide(ok, 0.020, 0.008)["adopt"]                   # ECE +0.012 is a material worsening
    assert not e.decide({"ci95": [-0.002, 0.0001]}, 0.005, 0.008)["adopt"]   # CI touches 0
    assert e.ECE_TOLERANCE == 0.005
