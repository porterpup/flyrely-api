"""Linkage extractor: synthetic SWIM XML only. Registrations and icao24 values here are fake."""
import json
import sqlite3
import time
from datetime import datetime, timezone

import pytest

from linkage import carriers
from linkage.extract import Extractor, FileTail, Runner, backfill, parse_sfdps_block, parse_tfms_block, CARRIERS
from linkage.store import Store
from swim import archive, filters

REG, ICAO = "NTEST1", "A00001"        # fake
REG2, ICAO2 = "NTEST2", "A00002"      # fake


# ------------------------------------------------------------------ fixtures
def tfms(msg_type="FlightScheduleActivate", acid="AAL100", dep="DFW", arr="ORD", igtd="2026-11-08T14:00:00Z",
         fac="TFMS", extra_times="", status="PLANNED", model="B738", etd="", eta="", ref="1001"):
    ftd = f'<nxcm:flightTimeData airlineInTime="2026-11-08T18:00:00Z" {extra_times}/>' if msg_type != "flightPlanCancellation" else ""
    tag = {"FlightScheduleActivate": "ncsmFlightScheduleActivate"}.get(msg_type, "x")
    return (f'<fdm:fltdMessage acid="{acid}" airline="{acid[:3]}" arrArpt="{arr}" cdmPart="false" depArpt="{dep}" '
            f'flightRef="{ref}" msgType="{msg_type}" sourceFacility="{fac}" sourceTimeStamp="2026-11-07T14:00:00Z">'
            f'<fdm:{tag}><nxcm:qualifiedAircraftId><nxce:aircraftId>{acid}</nxce:aircraftId>'
            f'<nxce:computerId><nxce:facilityIdentifier>{fac}</nxce:facilityIdentifier></nxce:computerId>'
            f'<nxce:igtd>{igtd}</nxce:igtd><nxce:departurePoint><nxce:airport>{dep}</nxce:airport></nxce:departurePoint>'
            f'<nxce:arrivalPoint><nxce:airport>{arr}</nxce:airport></nxce:arrivalPoint></nxcm:qualifiedAircraftId>'
            f'<nxcm:flightStatusAndSpec><nxcm:flightStatus>{status}</nxcm:flightStatus>'
            f'<nxcm:aircraftModel>{model}</nxcm:aircraftModel></nxcm:flightStatusAndSpec>{etd}{eta}{ftd}'
            f'</fdm:{tag}></fdm:fltdMessage>')


def tfms_msg(*blocks):
    return ('<?xml version="1.0"?><ds:tfmDataService xmlns:ds="x"><fltdOutput>' + "".join(blocks)
            + "</fltdOutput></ds:tfmDataService>")


def sfdps(acid="AAL100", dep="KDFW", arr="KORD", reg=REG, addr=ICAO, source="AH", gufi="us.fdps.2026-11-08T12:30:00Z.000/01/AAA",
          ftype="SCHEDULED", status="ACTIVE", dep_actual="", arr_est="2026-11-08T17:10:00Z", plan=True):
    desc = (f'<aircraftDescription aircraftAddress="{addr}" registration="{reg}"><aircraftType>'
            f'<icaoModelIdentifier>B738</icaoModelIdentifier></aircraftType></aircraftDescription>') if plan else ""
    dep_t = f'<actual time="{dep_actual}"/>' if dep_actual else ""
    return (f'<message><flight flightType="{ftype}" source="{source}">{desc}'
            f'<arrival arrivalPoint="{arr}"><runwayPositionAndTime><runwayTime><estimated time="{arr_est}"/>'
            f'</runwayTime></runwayPositionAndTime></arrival>'
            f'<departure departurePoint="{dep}"><runwayPositionAndTime><runwayTime>{dep_t}</runwayTime>'
            f'</runwayPositionAndTime></departure>'
            f'<flightIdentification aircraftIdentification="{acid}"/><flightStatus fdpsFlightStatus="{status}"/>'
            f'<gufi codeSpace="urn:uuid">uuid-{gufi[-3:]}</gufi><supplementalData><additionalFlightInformation>'
            f'<nameValue name="FDPS_GUFI" value="{gufi}"/></additionalFlightInformation></supplementalData>'
            f'</flight></message>')


def sf_msg(*blocks):
    return "<ns5:MessageCollection>" + "".join(blocks) + "</ns5:MessageCollection>"


def smes(cs="AAL100", icao=ICAO, ap="KDFW", event="runwayout", t="2026-11-08T14:20:00.000Z",
         gufi="us.fdps.2026-11-08T12:30:00Z.000/01/AAA", dep="KDFW", dest="KORD"):
    return (f'<ns2:SurfaceMovementEventMessage xmlns:ns2="u"><callsign>{cs}</callsign><airport>{ap}</airport>'
            f'<acAddress>{icao}</acAddress><time>{t}</time><event>{event}</event><enhancedData><eramGufi>KX1</eramGufi>'
            f'<sfdpsGufi>{gufi}</sfdpsGufi><departureAirport>{dep}</departureAirport>'
            f'<destinationAirport>{dest}</destinationAirport></enhancedData></ns2:SurfaceMovementEventMessage>')


def pdc(cs="AAL100", reg=REG, ap="KDFW", dest="KORD", gufi="us.fdps.2026-11-08T12:30:00Z.000/01/AAA"):
    return (f'<ns2:TDLSCSPMessage xmlns:ns2="u"><time>11082026141000</time><airportID>{ap}</airportID>'
            f'<aircraftID>{cs}</aircraftID><dataBody>CPDLC DCL {cs} {ap} B738/L P1430 /AN {reg} 76N FL320</dataBody>'
            f'<enhancedData><sfdpsGufi>{gufi}</sfdpsGufi><destinationAirport>{dest}</destinationAirport>'
            f'</enhancedData></ns2:TDLSCSPMessage>')


def tde(cs="AAL100", ap="KDFW", dest="KORD", gufi="us.fdps.2026-11-08T12:30:00Z.000/01/AAA"):
    return (f'<ns6:TowerDepartureEventMessage xmlns:ns6="u"><eventTime>2026-11-08T14:05:00.000Z</eventTime>'
            f'<aircraftID>{cs}</aircraftID><departureAirport>{ap}</departureAirport>'
            f'<clearanceDeliveryTime>2026-11-08T14:05:00.000Z</clearanceDeliveryTime><enhancedData>'
            f'<sfdpsGufi>{gufi}</sfdpsGufi><destinationAirport>{dest}</destinationAirport></enhancedData>'
            f'</ns6:TowerDepartureEventMessage>')


def rec(recv, body, **props):
    return {"recv": recv, "props": props, "body": body}


@pytest.fixture
def ex(tmp_path):
    st = Store(tmp_path / "live.sqlite")
    e = Extractor(st)
    yield e
    st.close()


def feed(ex, source, *records):
    ex.store.begin()
    for r in records:
        ex.process_record(source, r)
    ex.store.commit()


T0 = "2026-11-07T14:00:00.100000+00:00"
T1 = "2026-11-07T14:30:00.200000+00:00"
T2 = "2026-11-08T12:31:00.300000+00:00"
LEG = "AAL100|DFW|ORD|2026-11-08"


# ------------------------------------------------------------------ parsing
def test_tfms_fields_parsed(ex):
    blk = tfms(extra_times='gateDeparture="2026-11-08T14:02:00Z" runwayDeparture="2026-11-08T14:15:00Z"',
               dep="KDFW", arr="KORD", fac="OAG", msg_type="FlightModify")
    f = parse_tfms_block(blk)
    assert (f["callsign"], f["carrier"], f["flight_no"]) == ("AAL100", "AAL", "100")
    assert (f["origin"], f["dest"]) == ("DFW", "ORD")        # K prefix normalised
    assert f["igtd"] == "2026-11-08T14:00:00Z" and f["aircraft_type"] == "B738" and f["in_schedule"]
    assert f["actuals"] == {"out": "2026-11-08T14:02:00Z", "off": "2026-11-08T14:15:00Z"}


def test_sfdps_fields_parsed():
    f = parse_sfdps_block(sfdps(dep_actual="2026-11-08T14:15:00Z"), {})
    assert (f["callsign"], f["origin"], f["dest"]) == ("AAL100", "DFW", "ORD")
    assert (f["reg"], f["icao24"]) == (REG, ICAO)
    assert f["actuals"] == {"off": "2026-11-08T14:15:00Z"} and f["est"] == {"on": "2026-11-08T17:10:00Z"}
    assert f["plan_created"] == "2026-11-08T12:30:00Z" and f["flight_type"] == "SCHEDULED"
    assert f["aircraft_type"] == "B738" and f["source"] == "AH" and not f["cancelled"]


def test_all_sources_land_in_store(ex):
    feed(ex, "tfms", rec(T0, tfms_msg(tfms())))
    feed(ex, "sfdps", rec(T2, sf_msg(sfdps())))
    feed(ex, "stdds", rec(T2, smes()), rec(T2, pdc()), rec(T2, tde()))
    leg = ex.store.get_leg(LEG)
    assert leg["in_schedule"] == 1 and leg["aircraft_type"] == "B738" and leg["sfdps_gufi"].startswith("us.fdps")
    rows = {(r["source"], r["reg"], r["icao24"]) for r in ex.store.db.execute("SELECT * FROM leg_aircraft")}
    assert rows == {("sfdps", REG, ICAO), ("smes", "", ICAO), ("pdc", REG, "")}
    ev = {r["event"] for r in ex.store.db.execute("SELECT * FROM surface_events")}
    assert ev == {"runwayout", "pdc_clearance"}
    c = ex.store.counts()
    assert c["legs"] == 1 and c["legs_with_reg"] == 1 and c["surface_events"] == 2


def test_actuals_and_cancel(ex):
    feed(ex, "tfms", rec(T0, tfms_msg(tfms())),
         rec(T1, tfms_msg(tfms("FlightModify", fac="OAG", extra_times='runwayDeparture="2026-11-08T14:15:00Z"'))))
    leg = ex.store.get_leg(LEG)
    assert leg["off_utc"] == "2026-11-08T14:15:00Z" and leg["off_recv"].startswith("2026-11-07T14:30:00.200")
    feed(ex, "tfms", rec(T2, tfms_msg(tfms("flightPlanCancellation", fac="KZFW"))))
    assert ex.store.get_leg(LEG)["cancelled"] == 1


# ------------------------------------------------------------------ first_recv
def test_first_recv_is_monotonic_under_replay(ex):
    r_late = rec("2026-11-08T12:40:00.000000+00:00", sf_msg(sfdps()))
    r_early = rec(T2, sf_msg(sfdps()))
    feed(ex, "tfms", rec(T0, tfms_msg(tfms())))
    feed(ex, "sfdps", r_late)
    row = ex.store.db.execute("SELECT * FROM leg_aircraft").fetchone()
    assert row["first_recv"] == row["last_recv"] == "2026-11-08T12:40:00.000Z"
    feed(ex, "sfdps", r_early)          # an earlier-received copy (re-read order) can only move first_recv back
    row = ex.store.db.execute("SELECT * FROM leg_aircraft").fetchone()
    assert row["first_recv"] == "2026-11-08T12:31:00.300Z" and row["last_recv"] == "2026-11-08T12:40:00.000Z"
    feed(ex, "sfdps", rec("2026-11-08T13:00:00.000000+00:00", sf_msg(sfdps())))   # later copy: first_recv fixed
    row = ex.store.db.execute("SELECT * FROM leg_aircraft").fetchone()
    assert row["first_recv"] == "2026-11-08T12:31:00.300Z" and row["last_recv"] == "2026-11-08T13:00:00.000Z"
    leg = ex.store.get_leg(LEG)
    assert leg["first_seen_recv"] == "2026-11-07T14:00:00.100Z"
    assert all(r["first_recv"] <= r["last_recv"] for r in ex.store.db.execute("SELECT * FROM leg_aircraft"))


def test_replay_is_idempotent(ex):
    recs = [rec(T0, tfms_msg(tfms())), ]
    sf = [rec(T2, sf_msg(sfdps()))]
    feed(ex, "tfms", *recs)
    feed(ex, "sfdps", *sf)
    a = ex.store.counts()
    feed(ex, "tfms", *recs)
    feed(ex, "sfdps", *sf)
    assert ex.store.counts() == a


# ------------------------------------------------------------------ privacy drops
@pytest.mark.parametrize("ident", ["AAL100", REG, ICAO])
def test_ladd_drops_leg_by_callsign_reg_or_icao24(tmp_path, ident):
    bl = tmp_path / "ladd.txt"
    bl.write_text(f"# test list\n{ident}\n")
    st = Store(tmp_path / "s.sqlite")
    e = Extractor(st, filters.Blocklist(bl))
    feed(e, "tfms", rec(T0, tfms_msg(tfms(), tfms(acid="AAL200", igtd="2026-11-08T15:00:00Z"))))
    feed(e, "sfdps", rec(T2, sf_msg(sfdps(), sfdps(acid="AAL200", reg=REG2, addr=ICAO2,
                                                  gufi="us.fdps.2026-11-08T13:30:00Z.000/01/BBB",
                                                  arr_est="2026-11-08T18:10:00Z"))))
    feed(e, "stdds", rec(T2, smes()), rec(T2, pdc()), rec(T2, tde()))
    legs = {r["leg_id"] for r in st.db.execute("SELECT leg_id FROM legs")}
    assert "AAL100|DFW|ORD|2026-11-08" not in legs and "AAL200|DFW|ORD|2026-11-08" in legs
    assert st.db.execute("SELECT COUNT(*) FROM leg_aircraft WHERE reg=? OR icao24=?", (REG, ICAO)).fetchone()[0] == 0
    assert st.db.execute("SELECT COUNT(*) FROM surface_events WHERE callsign='AAL100'").fetchone()[0] == 0
    # later messages for the blocked leg are dropped too
    feed(e, "tfms", rec(T1, tfms_msg(tfms("FlightModify", fac="OAG"))))
    assert st.get_leg(LEG) is None
    st.close()


def test_ladd_list_change_scrubs_stored_data(tmp_path):
    bl = tmp_path / "ladd.txt"
    st = Store(tmp_path / "s.sqlite")
    e = Extractor(st, filters.Blocklist(bl, check_s=0))
    feed(e, "tfms", rec(T0, tfms_msg(tfms())))
    feed(e, "sfdps", rec(T2, sf_msg(sfdps())))
    assert st.get_leg(LEG) is not None
    time.sleep(0.01)
    bl.write_text(REG + "\n")
    e.check_blocklist_change()
    assert st.get_leg(LEG) is None and st.counts()["legs_with_reg"] == 0
    st.close()


def test_blockatfaa_drop(ex):
    feed(ex, "tfms", rec(T0, tfms_msg(tfms())))
    feed(ex, "sfdps", rec(T2, sf_msg(sfdps()), FDPS_BlockAtFAA="true"))
    assert ex.store.get_leg(LEG) is None
    assert ex.store.db.execute("SELECT COUNT(*) FROM leg_aircraft").fetchone()[0] == 0
    assert ex.counts["drop_blockatfaa"] == 1
    # the Sensitive flag and the STDDS blockAtFAA marker drop the same way
    feed(ex, "tfms", rec(T0, tfms_msg(tfms(acid="UAL5", igtd="2026-11-08T16:00:00Z"))))
    feed(ex, "sfdps", rec(T2, sf_msg(sfdps(acid="UAL5", gufi="us.fdps.2026-11-08T14:30:00Z.000/01/CCC")),
                          FDPS_Sensitive="true"))
    assert ex.store.get_leg("UAL5|DFW|ORD|2026-11-08") is None
    assert ex.store.counts()["blocked_legs"] >= 2
    feed(ex, "stdds", rec(T2, smes(cs="DAL9", gufi="g").replace("</callsign>", "</callsign><blockAtFAA>true</blockAtFAA>")))
    assert ex.store.db.execute("SELECT COUNT(*) FROM surface_events WHERE callsign='DAL9'").fetchone()[0] == 0


# ------------------------------------------------------------------ merging / universe
def test_refiled_plan_is_merged(ex):
    feed(ex, "tfms", rec(T0, tfms_msg(tfms(ref="1001"))),
         rec(T1, tfms_msg(tfms("FlightCreate", fac="AAL", ref="1002"))))    # TFMS re-file: new flightRef, same igtd
    assert ex.store.counts()["legs"] == 1
    g1, g2 = "us.fdps.2026-11-08T12:30:00Z.000/01/AAA", "us.fdps.2026-11-08T13:00:00Z.000/01/ZZZ"
    feed(ex, "sfdps", rec(T2, sf_msg(sfdps(gufi=g1))),
         rec("2026-11-08T13:05:00.000000+00:00", sf_msg(sfdps(gufi=g2, reg=REG2, addr=ICAO2))))   # SFDPS re-file, tail swap
    assert ex.store.counts()["legs"] == 1
    assert {r[0] for r in ex.store.db.execute("SELECT gufi FROM sfdps_gufis")} == {g1, g2}
    regs = [r["reg"] for r in ex.store.db.execute("SELECT reg FROM leg_aircraft ORDER BY first_recv")]
    assert regs == [REG, REG2]           # the swap is kept for audit


def test_cancel_then_refile_is_not_a_cancellation(ex):
    g1, g2 = "us.fdps.2026-11-08T12:30:00Z.000/01/AAA", "us.fdps.2026-11-08T13:00:00Z.000/01/ZZZ"
    feed(ex, "tfms", rec(T0, tfms_msg(tfms())))
    feed(ex, "sfdps", rec(T2, sf_msg(sfdps(gufi=g1))),
         rec("2026-11-08T12:40:00.000000+00:00", sf_msg(sfdps(gufi=g1, source="CL", status="CANCELLED", plan=False))))
    assert ex.store.get_leg(LEG)["cancelled"] == 1
    feed(ex, "sfdps", rec("2026-11-08T12:50:00.000000+00:00", sf_msg(sfdps(gufi=g2))))
    assert ex.store.get_leg(LEG)["cancelled"] == 0


def test_orphan_sfdps_leg_is_adopted_by_later_schedule(ex):
    feed(ex, "sfdps", rec(T2, sf_msg(sfdps())))
    assert ex.store.counts()["legs"] == 1 and ex.store.counts()["legs_in_schedule"] == 0
    feed(ex, "tfms", rec("2026-11-08T12:35:00.000000+00:00", tfms_msg(tfms())))
    c = ex.store.counts()
    assert c["legs"] == 1 and c["legs_in_schedule"] == 1 and c["legs_with_reg"] == 1


def test_ferry_leg_excluded_from_chain_universe(ex):
    feed(ex, "tfms", rec(T0, tfms_msg(tfms(), tfms(acid="AAL300", dep="DFW", arr="EGLL", igtd="2026-11-08T20:00:00Z"),
                                       tfms(acid="AAL400", arr="MIA", igtd="2026-11-08T21:00:00Z"))))
    # AAL400 is a non-scheduled ferry plan; AAL500 is a plan with no schedule at all (SFDPS only)
    feed(ex, "sfdps", rec(T2, sf_msg(sfdps(acid="AAL400", arr="KMIA", ftype="NON_SCHEDULED",
                                           gufi="us.fdps.2026-11-08T19:00:00Z.000/01/FFF"),
                                     sfdps(acid="AAL500", dep="KDFW", arr="KORD", gufi="us.fdps.2026-11-08T12:00:00Z.000/01/EEE"))))
    ids = {r["leg_id"] for r in ex.store.chain_legs(CARRIERS)}
    assert ids == {LEG}                      # international, ferry and unscheduled legs are all out
    assert ex.store.counts()["legs"] == 4
    # a leg whose TFMS schedule message predates the archive is in once the FAA plan says SCHEDULED ...
    feed(ex, "tfms", rec(T0, tfms_msg(tfms("FlightTimes", acid="AAL600", fac="DAL", igtd="2026-11-08T22:00:00Z"))))
    assert "AAL600|DFW|ORD|2026-11-08" not in {r["leg_id"] for r in ex.store.chain_legs(CARRIERS)}
    feed(ex, "sfdps", rec(T2, sf_msg(sfdps(acid="AAL600", gufi="us.fdps.2026-11-08T21:00:00Z.000/01/GGG"))))
    assert "AAL600|DFW|ORD|2026-11-08" in {r["leg_id"] for r in ex.store.chain_legs(CARRIERS)}
    # ... and a non-scheduled flag wins over the schedule evidence
    feed(ex, "sfdps", rec(T2, sf_msg(sfdps(acid="AAL100", ftype="CHARTER_OR_OTHER", gufi="us.fdps.2026-11-08T12:31:00Z.000/01/HHH"))))
    assert LEG not in {r["leg_id"] for r in ex.store.chain_legs(CARRIERS)}


def test_non_airline_and_unknown_carriers_ignored(ex):
    feed(ex, "tfms", rec(T0, tfms_msg(tfms(acid="N123AB"), tfms(acid="ZZZ100"))))
    feed(ex, "sfdps", rec(T2, sf_msg(sfdps(acid="N123AB"))))
    assert ex.store.counts()["legs"] == 0


def test_carriers_helpers():
    assert carriers.callsign_for("OO", 5331) == "SKW5331" and carriers.callsign_for("AA", "0100") == "AAL100"
    assert carriers.callsign_for("QQ", 1) == ""
    assert carriers.norm_airport("KORD") == "ORD" and carriers.norm_airport("PHNL") == "HNL" and carriers.norm_airport("ORD") == "ORD"
    assert carriers.is_us_normalised("ORD") and not carriers.is_us_normalised("EGLL")


# ------------------------------------------------------------------ tailing
def _write_hour(root, src, day, name, lines, codec="zstd", close=True):
    w = archive.ArchiveWriter(root / src, src, "r1", codec)
    w._seq = 0
    ts = datetime(2026, 11, 7, 14, 5, tzinfo=timezone.utc)
    w.write_batch([(ts, (json.dumps(l) + "\n").encode()) for l in lines])
    return w


def test_tail_follows_growing_file_and_resumes(tmp_path):
    root = tmp_path / "swim"
    w = archive.ArchiveWriter(root / "tfms", "tfms", "r1", "zstd")
    ts = datetime(2026, 11, 7, 14, 5, tzinfo=timezone.utc)
    mk = lambda i: (ts, (json.dumps({"recv": T0, "props": {}, "body": f"m{i}"}) + "\n").encode())
    w.write_batch([mk(0), mk(1)])
    path = next((root / "tfms").glob("*/*.zst"))
    t = FileTail(path)
    assert [json.loads(x)["body"] for x in list(t.poll())] == ["m0", "m1"]
    assert list(t.poll()) == []
    w.write_batch([mk(2)])
    assert [json.loads(x)["body"] for x in list(t.poll())] == ["m2"]
    # a restart re-reads from the start and skips what was already processed
    t2 = FileTail(path, skip_lines=3)
    assert list(t2.poll()) == []
    w.write_batch([mk(3), mk(4)])
    assert [json.loads(x)["body"] for x in list(t2.poll())] == ["m3", "m4"]
    # decoder trouble falls back to re-read + skip
    t3 = FileTail(path)
    list(t3.poll())
    t3.offset = 7                       # decoder is now misaligned: next feed raises ZstdError
    w.write_batch([mk(5)])
    assert [json.loads(x)["body"] for x in list(t3.poll())] == ["m5"]     # fallback re-read, skipping the 5 done
    assert t3.reread and list(t3.poll()) == []
    w.close()


def test_runner_end_to_end_and_backfill(tmp_path):
    root = tmp_path / "swim"
    day = "2026-11-07"
    ts = datetime(2026, 11, 7, 14, 5, tzinfo=timezone.utc)
    wt = archive.ArchiveWriter(root / "tfms", "tfms", "r1", "zstd")
    ws = archive.ArchiveWriter(root / "sfdps", "sfdps", "r1", "zstd")
    line = lambda r: (json.dumps(r) + "\n").encode()
    wt.write_batch([(ts, line(rec(T0, tfms_msg(tfms()))))])
    ws.write_batch([(ts, line(rec(T2, sf_msg(sfdps()))))])
    st = Store(tmp_path / "live.sqlite")
    runner = Runner(st, Extractor(st), root, tmp_path / "status.json")
    assert runner.poll([day]) == 2
    c = st.counts()
    assert c["legs"] == 1 and c["legs_with_reg"] == 1
    status = json.loads((tmp_path / "status.json").read_text())
    assert "store_lag_s" in status and REG not in json.dumps(status) and ICAO not in json.dumps(status)
    ws.write_batch([(ts, line(rec("2026-11-08T12:50:00.000000+00:00", sf_msg(sfdps(reg=REG2, addr=ICAO2)))))])
    assert runner.poll([day]) == 1
    assert st.db.execute("SELECT COUNT(*) FROM leg_aircraft").fetchone()[0] == 2
    # restart: a new runner skips the persisted line counts
    runner2 = Runner(st, Extractor(st), root)
    assert runner2.poll([day]) == 0
    wt.close(); ws.close()
    # back-fill into a fresh store gives the same counts (idempotent re-extraction)
    st2 = Store(tmp_path / "bf.sqlite")
    assert backfill(st2, Extractor(st2), root, day) == st.counts()
    assert backfill(st2, Extractor(st2), root, day) == st.counts()
    st.close(); st2.close()


def test_launchd_template_renders_and_keeps_alive():
    import plistlib
    from pathlib import Path
    text = (Path(__file__).resolve().parent.parent / "deploy" / "templates" / "launchd-linkage.plist.tmpl").read_text()
    for k, v in {"LABEL": "com.flyrely.linkage.extract", "PYTHON": "/x/py", "REPO": "/x", "ENV_FILE": "/x/.env",
                 "LOG_DIR": "/x/logs"}.items():
        text = text.replace(f"__{k}__", v)
    assert "__" not in text
    d = plistlib.loads(text.encode())
    assert d["RunAtLoad"] is True and d["KeepAlive"] is True and d["Label"] == "com.flyrely.linkage.extract"
    assert d["ProgramArguments"][1:3] == ["-m", "linkage.extract"]
