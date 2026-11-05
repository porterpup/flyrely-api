"""Receive-time filter tests. Synthetic messages shaped like the archived TFMS/SFDPS samples; offline."""
import json
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import pytest

from swim import archive, consumer, filters, replay_filter

T0 = datetime(2026, 11, 5, 19, 0, tzinfo=timezone.utc)


def tfms(acid, mtype, ref="1"):
    return (f'<?xml version="1.0"?><ds:tfmDataService xmlns:ds="x"><fltdOutput>'
            f'<fdm:fltdMessage xmlns:fdm="y" acid="{acid}" airline="{acid[:3]}" flightRef="{ref}" msgType="{mtype}">'
            f'<fdm:x/></fdm:fltdMessage></fltdOutput></ds:tfmDataService>')


FLIGHT = {"TFMDataClass": "FlightData"}


def sf_block(ident, gufi, desc=False, dep=None, arr=None, pos=False, reg="N1", addr="ABC123"):
    s = '<message xmlns:xsi="i" xsi:type="FlightMessageType"><flight source="HZ">'
    if arr:
        s += f'<arrival arrivalPoint="KJFK"><runwayPositionAndTime><runwayTime><actual time="{arr}"/></runwayTime></runwayPositionAndTime></arrival>'
    s += '<arrival arrivalPoint="KJFK"/>' if not arr else ""
    if dep:
        s += f'<departure departurePoint="KLAX"><runwayPositionAndTime><runwayTime><actual time="{dep}"/></runwayTime></runwayPositionAndTime></departure>'
    if pos:
        s += '<enRoute><position><location><pos>1 2</pos></location></position></enRoute>'
    s += f'<flightIdentification computerId="1" aircraftIdentification="{ident}"/>'
    if desc:
        s += f'<aircraftDescription registration="{reg}" aircraftAddress="{addr}"/>'
    s += f'<gufi codeSpace="urn:uuid">{gufi}</gufi></flight></message>'
    return s


def sf_body(*blocks):
    return '<?xml version="1.0"?><ns5:MessageCollection xmlns:ns5="n">\n    ' + "\n    ".join(blocks) + "\n</ns5:MessageCollection>"


SF = {"FDPS_MessageType": "HZ_FIXM", "FDPS_DataType": "FlightFIXM"}


def flt(source="tfms", cfg=None, ids=None, tmp_path=None):
    bl = filters.Blocklist(None)
    if ids is not None:
        p = tmp_path / "bl.txt"
        p.write_text("\n".join(ids))
        bl = filters.Blocklist(p)
    return filters.MessageFilter(source, cfg, bl)


# ------------------------------------------------------------------- TFMS
def test_tfms_keeps_airline_types_drops_ga_and_unwanted_types():
    f = flt()
    now = T0.timestamp()
    for t in ("FlightTimes", "flightPlanInformation", "flightPlanAmendmentInformation", "FlightCreate",
              "departureInformation", "arrivalInformation", "flightPlanCancellation"):
        assert f.apply(FLIGHT, tfms("UAL123", t), now)[0] is not None, t
    assert f.apply(FLIGHT, tfms("N21DP", "FlightTimes"), now)[1] == "drop:not_airline"
    assert f.apply(FLIGHT, tfms("UAL123", "FlightSectors"), now) == (None, "drop:type:FlightSectors")
    assert f.apply(FLIGHT, tfms("SKW5657", "boundaryCrossingUpdate"), now)[0] is None


def test_tfms_flow_and_status_always_kept():
    f = flt()
    body = "<tfmDataService><flowStuff/></tfmDataService>"
    assert f.apply({"TFMDataClass": "FlowInformation", "msgType": "RSTR"}, body, 0) == (body, "keep:flow")
    assert f.apply({"TFMDataClass": "Status"}, body, 0) == (body, "keep:status")
    off = flt(cfg={"keep_flow": False})
    assert off.apply({"TFMDataClass": "FlowInformation"}, body, 0)[0] is None


def test_tfms_track_thinning_interval_per_flight():
    f = flt(cfg={"track_interval_min": 5})
    t = T0.timestamp()
    track = tfms("UAL1", "trackInformation", "7")
    assert f.apply(FLIGHT, track, t)[1] == "keep:track_sample"
    assert f.apply(FLIGHT, track, t + 60)[1] == "drop:track_thinned"
    assert f.apply(FLIGHT, tfms("UAL2", "trackInformation", "8"), t + 60)[1] == "keep:track_sample"  # other flight
    assert f.apply(FLIGHT, track, t + 301)[1] == "keep:track_sample"


def test_tfms_track_interval_zero_drops_all_tracks():
    f = flt(cfg={"track_interval_min": 0})
    assert f.apply(FLIGHT, tfms("UAL1", "trackInformation"), 0)[0] is None


def test_airline_modes():
    assert filters.is_airline("UAL123", "carriers", {"UAL"})
    assert not filters.is_airline("XYZ123", "carriers", {"UAL"})
    assert filters.is_airline("XYZ123", "pattern", set())
    assert not filters.is_airline("N123AB", "pattern", set())
    assert not filters.is_airline("", "pattern", set())
    f = flt(cfg={"extra_carriers": ["xyz"]})
    assert f.apply(FLIGHT, tfms("XYZ12", "FlightTimes"), 0)[0] is not None


def test_unparseable_and_errors_fail_open(monkeypatch):
    f = flt()
    assert f.apply(FLIGHT, "<garbage", 0) == ("<garbage", "keep:unparsed")
    assert f.apply({"TFMDataClass": "NewClass"}, "<a/>", 0)[1] == "keep:other_class"
    monkeypatch.setattr(f, "_tfms", lambda *a: 1 / 0)
    out, reason = f.apply(FLIGHT, "<a/>", 0)
    assert out == "<a/>" and reason.startswith("keep:filter_error")


def test_filter_disabled_and_stdds_passthrough():
    f = flt(cfg={"enabled": False})
    assert f.apply(FLIGHT, tfms("N1", "trackInformation"), 0)[0] is not None
    assert flt("stdds").apply({}, "<a/>", 0) == ("<a/>", "keep:passthrough")


# ------------------------------------------------------------------ SFDPS
def test_sfdps_flight_plan_runway_time_track_and_other():
    f = flt("sfdps")
    t = T0.timestamp()
    plan = sf_body(sf_block("UAL1", "g1", desc=True))
    assert f.apply(SF, plan, t) == (plan, "keep:flight_plan")
    dep = sf_body(sf_block("UAL2", "g2", dep="2026-11-05T18:58:00Z", pos=True))
    assert f.apply(SF, dep, t)[1] == "keep:runway_time"
    assert f.apply(SF, dep, t + 10)[1] == "keep:track_sample"  # runway time already seen; first track sample
    assert f.apply(SF, dep, t + 20)[1] == "drop:track_thinned"
    newarr = sf_body(sf_block("UAL2", "g2", dep="2026-11-05T18:58:00Z", arr="2026-11-05T21:00:00Z"))
    assert f.apply(SF, newarr, t + 30)[1] == "keep:runway_time"  # a new actual (arrival) time
    other = sf_body(sf_block("UAL3", "g3"))
    assert f.apply(SF, other, t) == (None, "drop:other")
    assert f.apply(SF, sf_body(sf_block("N12AB", "g4", desc=True)), t)[1] == "drop:not_airline"


def test_sfdps_status_and_nonflight_kept():
    f = flt("sfdps")
    st = "<FDPSMsg><properties/></FDPSMsg>"
    assert f.apply({"FDPS_DataType": "STATUS", "FDPS_MessageType": "status"}, st, 0)[1] == "keep:status"
    assert f.apply(SF, st, 0)[1] == "keep:nonflight"


def test_sfdps_batch_is_split_and_reassembled_as_valid_xml():
    f = flt("sfdps", cfg={"track_interval_min": 5})
    t = T0.timestamp()
    batch = sf_body(sf_block("UAL1", "g1", pos=True), sf_block("N9", "g2", pos=True),
                    sf_block("DAL2", "g3", pos=True), sf_block("AAL3", "g4"))
    out, reason = f.apply({"FDPS_MessageType": "BATCH_TH_FIXM"}, batch, t)
    root = ET.fromstring(out)  # still well-formed
    kept = [e.attrib["aircraftIdentification"] for e in root.iter() if "aircraftIdentification" in e.attrib]
    assert kept == ["UAL1", "DAL2"] and reason == "keep:track_sample"
    out2, reason2 = f.apply({"FDPS_MessageType": "BATCH_TH_FIXM"}, batch, t + 30)
    assert out2 is None and reason2.startswith("drop:")  # all thinned/dropped
    allkept = sf_body(sf_block("UAL1", "g9", desc=True), sf_block("DAL2", "g8", desc=True))
    assert f.apply(SF, allkept, t)[0] is allkept  # unchanged: same object, no rewrite


def test_sfdps_message_type_override_keeps_whole_message():
    f = flt("sfdps", cfg={"keep_message_types": ["HZ_FIXM"]})
    body = sf_body(sf_block("N1", "g1"))
    assert f.apply(SF, body, 0) == (body, "keep:configured_type")


# -------------------------------------------------------------- LADD hook
def test_blocklist_drops_by_callsign_registration_and_address(tmp_path):
    f = flt(ids=["# comment", "UAL123, n750ec", "aa1a70  # trailing comment"], tmp_path=tmp_path)
    assert f.apply(FLIGHT, tfms("UAL123", "FlightTimes"), 0) == (None, "drop:ladd")
    assert f.apply(FLIGHT, tfms("UAL124", "FlightTimes"), 0)[0] is not None
    s = flt("sfdps", ids=["N750EC", "AA1A70"], tmp_path=tmp_path)
    assert s.apply(SF, sf_body(sf_block("UAL9", "g", desc=True, reg="N750EC")), 0)[1] == "drop:ladd"
    assert s.apply(SF, sf_body(sf_block("UAL9", "g", desc=True, addr="AA1A70")), 0)[1] == "drop:ladd"
    assert s.apply(SF, sf_body(sf_block("UAL9", "g", desc=True)), 0)[1] == "keep:flight_plan"


def test_blocklist_applies_even_when_filtering_is_off_for_airlines_only(tmp_path):
    f = flt(ids=["N21DP"], cfg={"airlines_only": False}, tmp_path=tmp_path)
    assert f.apply(FLIGHT, tfms("N21DP", "FlightTimes"), 0)[1] == "drop:ladd"


def test_blocklist_absent_is_noop_and_reloads_on_change(tmp_path):
    p = tmp_path / "bl.txt"
    bl = filters.Blocklist(p, check_s=0)
    assert not bl.hit("UAL1")
    p.write_text("UAL1\n")
    assert bl.hit("ual1")
    p.write_text("OTHER\n")
    os.utime(p, (1, 1))  # force a different mtime
    assert not bl.hit("UAL1") and bl.hit("OTHER")
    p.unlink()
    assert not bl.hit("OTHER")


# ----------------------------------------------------------------- config
def test_load_config_overlay_and_env(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"tfms": {"track_interval_min": 9}}))
    cfg = filters.load_config(str(p), env={})
    assert cfg["tfms"]["track_interval_min"] == 9 and cfg["sfdps"]["track_interval_min"] == 5
    cfg = filters.load_config(None, env={"SWIM_FILTER": "off", "SWIM_TRACK_INTERVAL_MIN": "2"})
    assert not cfg["tfms"]["enabled"] and cfg["sfdps"]["track_interval_min"] == 2.0
    p.write_text(json.dumps({"nope": {}}))
    with pytest.raises(ValueError):
        filters.load_config(str(p), env={})
    assert not filters.load_config(None, env={})["stdds"]["enabled"]


def test_example_config_is_valid_and_matches_defaults():
    ex = json.loads((__import__("pathlib").Path(__file__).resolve().parent.parent / "deploy" / "filter.example.json").read_text())
    cfg = filters.load_config(None, env={})
    for src, vals in ex.items():
        for k, v in vals.items():
            assert cfg[src][k] == v, (src, k)


# ----------------------------------------------------------------- replay
def test_replay_measures_reduction_on_a_synthetic_archive(tmp_path):
    w = archive.ArchiveWriter(tmp_path / "tfms", "tfms", "r", "gzip")
    recs = []
    for i in range(60):
        ts = T0 + timedelta(seconds=10 * i)
        body = tfms("UAL1" if i % 2 else "N21DP", "trackInformation", "5") + " " * 500
        recs.append((ts, consumer.encode_record(ts, FLIGHT, body.encode())))
    ts = T0 + timedelta(seconds=605)
    recs.append((ts, consumer.encode_record(ts, {"TFMDataClass": "FlowInformation"}, b"<flow/>")))
    w.write_batch(recs)
    w.close()
    r = replay_filter.replay("tfms", tmp_path / "tfms", filters.DEFAULTS["tfms"], "gzip")
    assert r["messages_in"] == 61 and r["messages_kept"] == 3  # 2 airline track samples (10 min / 5 min) + flow
    assert r["counts"]["drop:not_airline"] == 30
    assert r["raw_mb_kept"] < r["raw_mb_in"] and r["gb_day_filtered"] < r["gb_day_unfiltered_new_codec"]


# ------------------------------------------- TFMS batches (several flights per message)
def tfms_batch(*elements):
    """Real shape (from the archived samples, synthetic values): one tfmDataService / fltdOutput holding several
    namespaced fltdMessage elements, no whitespace between them."""
    return ('<?xml version="1.0" encoding="UTF-8" standalone="no"?><ds:tfmDataService xmlns:ds="urn:x" '
            'xmlns:fdm="urn:y" xmlns:nxce="urn:z" xsi:schemaLocation="a b" xmlns:xsi="urn:i"><fltdOutput>'
            + "".join(elements) + "</fltdOutput></ds:tfmDataService>")


def tfms_el(acid, mtype, ref):
    return (f'<fdm:fltdMessage acid="{acid}" airline="{acid[:3]}" cdmPart="false" flightRef="{ref}" msgType="{mtype}" '
            f'sensitivity="A"><fdm:{mtype}><nxce:aircraftId>{acid}</nxce:aircraftId><nxce:note a="&lt;x&gt;"/></fdm:{mtype}>'
            f'</fdm:fltdMessage>')


def test_tfms_mixed_batch_is_filtered_per_flight_and_stays_faithful():
    f = flt(cfg={"track_interval_min": 5})
    t = T0.timestamp()
    ga, ual, dal, ual_t = (tfms_el("N254DL", "trackInformation", "1"), tfms_el("UAL1439", "FlightTimes", "2"),
                           tfms_el("DAL7", "FlightSectors", "3"), tfms_el("UAL1439", "trackInformation", "2"))
    batch = tfms_batch(ga, ual, dal, ual_t)  # GA track, airline FlightTimes, airline unwanted type, airline track
    out, reason = f.apply(FLIGHT, batch, t)
    assert reason == "keep:FlightTimes"
    assert out == tfms_batch(ual, ual_t)  # kept elements byte-identical, envelope untouched
    root = ET.fromstring(out)
    assert [e.attrib["acid"] for e in root.iter("{urn:y}fltdMessage")] == ["UAL1439", "UAL1439"]
    # second time: the track for UAL1439 is thinned, FlightTimes stays
    out2, _ = f.apply(FLIGHT, batch, t + 30)
    assert out2 == tfms_batch(ual)
    # a batch with nothing wanted is dropped entirely, with the most common reason
    assert f.apply(FLIGHT, tfms_batch(ga, tfms_el("N1", "FlightTimes", "9")), t) == (None, "drop:not_airline")
    # nothing to remove: the very same object comes back (no re-serialisation)
    only = tfms_batch(ual, tfms_el("AAL2", "flightPlanInformation", "4"))
    assert f.apply(FLIGHT, only, t)[0] is only


def test_tfms_batch_ladd_drops_only_the_listed_flight(tmp_path):
    f = flt(ids=["UAL1439"], tmp_path=tmp_path)
    batch = tfms_batch(tfms_el("UAL1439", "FlightTimes", "2"), tfms_el("DAL7", "FlightTimes", "3"))
    assert f.apply(FLIGHT, batch, 0)[0] == tfms_batch(tfms_el("DAL7", "FlightTimes", "3"))
    assert f.apply(FLIGHT, tfms_batch(tfms_el("UAL1439", "FlightTimes", "2")), 0) == (None, "drop:ladd")


# ------------------------------------------------ LADD on every path
def test_ladd_with_filter_off_applies_per_flight(tmp_path):
    f = flt(cfg={"enabled": False}, ids=["UAL1439", "N750EC"], tmp_path=tmp_path)
    batch = tfms_batch(tfms_el("UAL1439", "trackInformation", "2"), tfms_el("N254DL", "trackInformation", "1"))
    out, reason = f.apply(FLIGHT, batch, 0)
    assert out == tfms_batch(tfms_el("N254DL", "trackInformation", "1")) and reason == "keep:filter_off"
    s = flt("sfdps", cfg={"enabled": False}, ids=["N750EC"], tmp_path=tmp_path)
    body = sf_body(sf_block("UAL9", "g1", desc=True, reg="N750EC"), sf_block("DAL2", "g2", desc=True))
    out, _ = s.apply(SF, body, 0)
    assert "N750EC" not in out and "DAL2" in out
    flow = "<flow>UAL1439</flow>"
    assert f.apply({"TFMDataClass": "FlowInformation"}, flow, 0) == (None, "drop:ladd_text")  # filter off: text scan
    on = flt(ids=["UAL1439"], tmp_path=tmp_path)  # filter on: flow data is kept whole (documented limitation)
    assert on.apply({"TFMDataClass": "FlowInformation"}, flow, 0)[0] == flow


def test_ladd_on_stdds_passthrough_and_unparseable_and_binary(tmp_path):
    st = flt("stdds", ids=["UAL1439", "AA1A70"], tmp_path=tmp_path)
    assert st.apply({}, "<m><acid>ual1439</acid></m>", 0) == (None, "drop:ladd_text")
    assert st.apply({}, "<m><acid>UAL14390</acid></m>", 0)[0] is not None  # whole-token match only
    f = flt(ids=["UAL1439"], tmp_path=tmp_path)
    assert f.apply(FLIGHT, '<fltdMessage acid="UAL1439" <broken', 0) == (None, "drop:ladd_text")  # unparseable
    assert f.apply(FLIGHT, "<garbage but fine", 0)[1] == "keep:unparsed"
    assert f.apply({"TFMDataClass": "Brand"}, "<x>UAL1439</x>", 0)[1] == "drop:ladd_text"  # unknown class
    s = flt("sfdps", ids=["AA1A70"], tmp_path=tmp_path)
    assert s.apply(SF, "<weird>aa1a70</weird>", 0)[1] == "drop:ladd_text"  # non-flight SFDPS text
    assert f.check_binary(b"\xff\xfe UAL1439 \x00") is False and f.check_binary(b"\xff\xfe nothing") is True
    assert flt().check_binary(b"\xff UAL1439") is True  # no list: no-op


def test_ladd_scan_survives_filter_errors(monkeypatch, tmp_path):
    f = flt(ids=["UAL1439"], tmp_path=tmp_path)
    monkeypatch.setattr(f, "_tfms", lambda *a: 1 / 0)
    assert f.apply(FLIGHT, "<a>UAL1439</a>", 0) == (None, "drop:ladd_text")
    assert f.apply(FLIGHT, "<a>fine</a>", 0)[1].startswith("keep:filter_error")


def test_sfdps_cancellation_kept():
    f = flt("sfdps")
    blk = sf_block("UAL5", "gc").replace('<gufi', '<flightStatus fdpsFlightStatus="CANCELLED"/><gufi')
    assert f.apply(SF, sf_body(blk), 0) == (sf_body(blk), "keep:cancellation")
    off = flt("sfdps", cfg={"keep_cancellations": False})
    assert off.apply(SF, sf_body(blk), 0)[0] is None
