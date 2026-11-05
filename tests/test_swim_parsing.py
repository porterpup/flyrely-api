"""Parsing helpers for the SWIM archive. Synthetic data only; no network."""
import gzip
import json
from datetime import datetime, timezone

from swim import consumer
from swim import inspect_tfms as it

XML = """<?xml version="1.0"?>
<ns5:tfmDataService xmlns:ns5="urn:x" xmlns:ns3="urn:y">
  <ns5:fltdOutput>
    <ns3:fltdMessage acid="TST123" depArpt="KAAA" arrArpt="KBBB" flightRef="42" msgType="FlightTimes">
      <ns3:gufi>US.TEST.1</ns3:gufi>
      <ns3:flightTimeData airlineOffTime="2026-01-01T10:00:00Z" airlineOnTime="" runwayArrival="2026-01-01T12:00:00Z"/>
      <ns3:aircraftRegistration>N123AB</ns3:aircraftRegistration>
      <ns3:remarks>RMK/ CODE/A1B2C3 OPR/X</ns3:remarks>
    </ns3:fltdMessage>
  </ns5:fltdOutput>
</ns5:tfmDataService>"""


def test_root_and_local_names():
    root = it.parse(XML)
    assert it.root_tag(root) == "tfmDataService"
    assert it.local("{urn:x}foo") == "foo"
    assert it.parse("<broken") is None


def test_registration_icao_times_and_key():
    root = it.parse(XML)
    assert it.find_registration(root) == "N123AB"
    assert it.find_icao(root) == "A1B2C3"
    assert it.find_times(root) == {"off", "on"}  # empty airlineOnTime ignored, runwayArrival counts
    assert it.flight_key(root) == "r:42"


def test_reg_in_other_info_and_absent():
    root = it.parse('<a><fltdMessage acid="X"><other>REG/N9XYZ RMK/Q</other></fltdMessage></a>')
    assert it.find_registration(root) == "N9XYZ"
    assert it.find_icao(root) is None
    assert it.find_times(root) == set()


def test_archive_roundtrip_and_redact(tmp_path, monkeypatch):
    ts = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    recs = [(ts, consumer.encode_record(ts, {"k": "v"}, b"<a/>"))]
    consumer.write_batch(tmp_path, recs)
    consumer.write_batch(tmp_path, recs)  # second batch appends a new gzip member
    files = list(tmp_path.glob("2026-01-02/*.jsonl.gz"))
    assert len(files) == 1 and "20260102T03" in files[0].name
    lines = gzip.open(files[0], "rt").read().splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["body"] == "<a/>"
    monkeypatch.setenv("SCDS_TFMS_PASSWORD", "s3cret-pw")
    assert "s3cret-pw" not in consumer.redact("fail with s3cret-pw")


def test_fixm_actual_times():
    root = it.parse(
        '<m><flight><departure><runwayPositionAndTime><runwayTime><actual time="2026-01-01T10:00:00Z"/>'
        '</runwayTime></runwayPositionAndTime></departure>'
        '<arrival><runwayPositionAndTime><runwayTime><estimated time="2026-01-01T12:00:00Z"/>'
        '</runwayTime></runwayPositionAndTime></arrival>'
        '<aircraftDescription registration="N1AB" aircraftAddress="ABCDEF"/></flight></m>')
    assert it.find_times(root) == {"off"}
    assert it.find_registration(root) == "N1AB"
    assert it.find_icao(root) == "ABCDEF"
