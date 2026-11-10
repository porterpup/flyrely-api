"""L1 T1: lead computation of scripts/l1_measure_fill.py on a synthetic 3-message fixture. Offline."""
import gzip
import importlib.util
import json
from datetime import date, datetime, timezone
from pathlib import Path

spec = importlib.util.spec_from_file_location("l1_measure_fill", Path(__file__).resolve().parent.parent / "scripts" / "l1_measure_fill.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

SECRET_REG = "N12345"


def tfms(acid, dep, arr, igtd, mtype="FlightCreate"):
    return (f'<x><fdm:fltdMessage acid="{acid}" depArpt="{dep}" arrArpt="{arr}" msgType="{mtype}">'
            f'<nxce:igtd>{igtd}</nxce:igtd></fdm:fltdMessage></x>')


def sfdps(ident, dep, arr, reg=SECRET_REG):
    return (f'<message><flight><aircraftDescription registration="{reg}"/><arrival arrivalPoint="{arr}"/>'
            f'<departure departurePoint="{dep}"/><flightIdentification aircraftIdentification="{ident}"/></flight></message>')


def ts(h, mi=0, d=8):
    return datetime(2026, 11, d, h, mi, tzinfo=timezone.utc)


def fixture():
    t = [(ts(5), tfms("UAL100", "EWR", "KLAX", "2026-11-08T12:00:00Z")),          # reg at T-150 -> lead 150
         (ts(5), tfms("UAL100", "EWR", "KLAX", "2026-11-08T12:00:00Z")),          # duplicate plan, merged
         (ts(5), tfms("UAL200", "KORD", "KDEN", "2026-11-08T15:00:00Z", "FlightScheduleActivate")),  # never gets a reg
         (ts(5), tfms("DAL300", "KATL", "KJFK", "2026-11-08T18:00:00Z"))]          # reg at T-45
    s = [(ts(9, 30), sfdps("UAL100", "EWR", "KLAX")), (ts(11, 0), sfdps("UAL100", "KEWR", "KLAX")),
         (ts(17, 15), sfdps("DAL300", "KATL", "KJFK"))]
    return t, s


def test_lead_computation_and_fill():
    t, s = fixture()
    res = m.analyse(date(2026, 11, 8), t, s, [])
    ual, dal = res["carriers"]["UAL"], res["carriers"]["DAL"]
    assert ual["scheduled_legs"] == 2 and ual["legs_with_reg"] == 1 and ual["fill"] == 0.5
    assert ual["reg_lead_min_deciles"]["p50"] == 150.0  # first reg message wins (09:30 vs igtd 12:00)
    assert ual["known_at_60"] == 0.5 and ual["known_at_180"] == 0.0
    assert dal["reg_lead_min_deciles"]["p50"] == 45.0 and dal["known_at_60"] == 0.0 and dal["fill"] == 1.0
    assert res["all_carriers"]["scheduled_legs"] == 3 and res["legs_domestic"] == 3


def test_stdds_coverage_nks_and_no_registration_leaks(tmp_path):
    t, s = fixture()
    t.append((ts(5), tfms("NKS1", "KFLL", "KATL", "2026-11-08T14:00:00Z")))
    smes = ('<ns2:SurfaceMovementEventMessage><callsign>UAL100</callsign><airport>KEWR</airport>'
            '<event>runwayout</event></ns2:SurfaceMovementEventMessage>')
    pdc = ('<ns4:TDLSCSPMessage><airportID>KEWR</airportID><aircraftID>UAL100</aircraftID>'
           f'<dataBody>UAL100 KEWR /AN {SECRET_REG} CLEARED</dataBody></ns4:TDLSCSPMessage>')
    res = m.analyse(date(2026, 11, 8), t, s, [(ts(11, 50), smes), (ts(11, 55), pdc)])
    ual = res["carriers"]["UAL"]
    assert ual["smes_legs"] == 1 and ual["pdc_legs"] == 1 and ual["pdc_with_reg_legs"] == 1
    assert res["carriers"]["NKS"]["scheduled_legs"] == 1 and res["nks"]["tfms_legs"] == 1 and "defunct" in res["nks"]["status"]
    assert SECRET_REG not in json.dumps(res) and "UAL100" not in json.dumps(res)


def test_reads_archive_dirs_and_writes_json_without_reg(tmp_path, capsys):
    t, s = fixture()
    for src, msgs in (("tfms", t), ("sfdps", s)):
        d = tmp_path / src / "2026-11-08"
        d.mkdir(parents=True)
        with gzip.open(d / f"{src}_x.jsonl.gz", "wt") as f:
            for recv, body in msgs:
                f.write(json.dumps({"recv": recv.isoformat(), "props": {}, "body": body}) + "\n")
    out = tmp_path / "o.json"
    assert m.main(["--date", "2026-11-08", "--root", str(tmp_path), "--out", str(out)]) == 0
    assert SECRET_REG not in out.read_text() and SECRET_REG not in capsys.readouterr().out
    assert json.loads(out.read_text())["carriers"]["UAL"]["legs_with_reg"] == 1


def test_hal_counted_inside_asa():
    t = [(ts(5), tfms("HAL5", "HNL", "LAX", "2026-11-08T12:00:00Z")), (ts(5), tfms("ASA7", "SEA", "LAX", "2026-11-08T13:00:00Z"))]
    res = m.analyse(date(2026, 11, 8), t, [], [])
    assert res["carriers"]["ASA"]["scheduled_legs"] == 2 and "HAL" not in res["carriers"]


def test_plan_message_over_12h_early_is_not_this_leg():
    t = [(ts(5), tfms("UAL100", "EWR", "LAX", "2026-11-08T12:00:00Z"))]
    res = m.analyse(date(2026, 11, 8), t, [(ts(23, d=7), sfdps("UAL100", "EWR", "LAX"))], [])
    assert res["carriers"]["UAL"]["legs_with_reg"] == 0


def _load_legacy():
    sp = importlib.util.spec_from_file_location("l1_legacy", Path(__file__).resolve().parent / "l1_measure_fill_legacy.py")
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


def test_streaming_matches_old_list_based_code(tmp_path):
    """Old code (all messages in lists) and the streaming code give identical JSON on a synthetic multi-file archive set."""
    import random
    from datetime import timedelta
    old = _load_legacy()
    rnd = random.Random(7)
    calls = [("UAL", "EWR", "LAX"), ("DAL", "ATL", "JFK"), ("SWA", "DEN", "PHX"), ("HAL", "HNL", "LAX"), ("NKS", "FLL", "ATL"), ("SKW", "ORD", "MSN")]
    codes = {"EWR": "KEWR", "LAX": "KLAX", "ATL": "KATL", "JFK": "KJFK", "DEN": "KDEN", "PHX": "KPHX", "HNL": "PHNL", "FLL": "KFLL", "ORD": "KORD", "MSN": "KMSN"}
    files = {"tfms": {}, "sfdps": {}, "stdds": {}}

    def put(src, day, hour, recv, body):
        files[src].setdefault((day, hour), []).append({"recv": recv.isoformat(), "props": {}, "body": body})

    for n in range(60):
        pre, dep, arr = calls[n % len(calls)]
        call = f"{pre}{100 + n % 7}"
        igtd = datetime(2026, 11, 8, rnd.randrange(0, 24), rnd.choice([0, 15, 30, 45]), tzinfo=timezone.utc) + timedelta(days=rnd.choice([0, 0, 0, 1, -1]))
        for _ in range(rnd.choice([1, 2])):  # duplicate plans
            r = igtd - timedelta(hours=rnd.randrange(14, 30))
            put("tfms", r.date().isoformat(), r.hour, r, tfms(call, dep, codes[arr], igtd.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                                              rnd.choice(["FlightCreate", "FlightScheduleActivate"])))
        for _ in range(rnd.choice([0, 1, 3])):
            r = igtd - timedelta(minutes=rnd.randrange(-30, 800))
            put("sfdps", r.date().isoformat(), r.hour, r, sfdps(call, codes[dep], codes[arr], rnd.choice(["N1", "N2"])))
        for _ in range(rnd.choice([0, 1, 2])):
            r = igtd - timedelta(minutes=rnd.randrange(-60, 200))
            if rnd.random() < 0.5:
                body = f"<ns2:SurfaceMovementEventMessage><callsign>{call}</callsign><airport>{codes[rnd.choice([dep, arr])]}</airport></ns2:SurfaceMovementEventMessage>"
            else:
                body = (f"<ns4:TDLSCSPMessage><airportID>{codes[dep]}</airportID><aircraftID>{call}</aircraftID>"
                        f"<dataBody>{'/AN N9 ' if rnd.random() < .5 else ''}X</dataBody></ns4:TDLSCSPMessage>")
            put("stdds", r.date().isoformat(), r.hour, r, body)
    for src, d in files.items():
        for (day, hour), recs in d.items():
            dd = tmp_path / src / day
            dd.mkdir(parents=True, exist_ok=True)
            with gzip.open(dd / f"{src}_{hour:02d}.jsonl.gz", "wt") as f:
                for rec in recs:
                    f.write(json.dumps(rec) + "\n")
    day = date(2026, 11, 8)
    dk = lambda k: day + timedelta(days=k)
    t = old.read_source(tmp_path, "tfms", [dk(-2), dk(-1), dk(0), dk(1)])
    s = old.read_source(tmp_path, "sfdps", [dk(-1), dk(0), dk(1)])
    st = old.read_source(tmp_path, "stdds", [dk(-1), dk(0), dk(1)])
    start = datetime(2026, 11, 8, tzinfo=timezone.utc)
    cov = {"tfms": old.coverage_of(t), "sfdps": old.coverage_of(s), "stdds": old.coverage_of(st)}
    cov["tfms"]["hours_with_data_prev_day"] = old.hours_covered(t, start - timedelta(days=1))
    cov["sfdps"]["hours_with_data_day"] = old.hours_covered(s, start)
    cov["stdds"]["hours_with_data_day"] = old.hours_covered(st, start)
    cov["schedule_universe_complete"] = cov["tfms"]["hours_with_data_prev_day"] == 24
    cov["complete_day"] = cov["schedule_universe_complete"] and cov["sfdps"]["hours_with_data_day"] == 24 and cov["stdds"]["hours_with_data_day"] == 24
    expected = old.analyse(day, t, s, st, cov)
    out = tmp_path / "new.json"
    assert m.main(["--date", "2026-11-08", "--root", str(tmp_path), "--out", str(out)]) == 0
    got = json.loads(out.read_text())
    assert got == json.loads(json.dumps(expected))
    assert got["legs_domestic"] > 10 and got["all_carriers"]["legs_with_reg"] > 3 and got["all_carriers"]["smes_legs"] > 0
