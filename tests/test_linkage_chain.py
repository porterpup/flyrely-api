"""Tests for the CHAIN rules and study (linkage/chain_rules.py, priors_offline.py, studies/chain_measure.py).
Synthetic fixtures only; the one test that needs the live store skips when it is absent (CI has none)."""
import json
import os
from datetime import date, datetime, timezone

import pytest

from linkage import chain_rules as cr
from linkage import priors_offline as po
from linkage.chain_rules import AsOfState, Leg
from linkage.studies import chain_measure as cm

H = 3600.0
X, Y = "AC0001", "AC0002"


def leg(i, carrier="DAL", fn="100", o="AAA", d="BBB", so=0.0, dur=2 * H, fam=""):
    return Leg(i, carrier, fn, o, d, so, so + dur, fam)


def deps_of(legs):
    idx = {}
    for lg in sorted(legs, key=lambda g: g.so):
        idx.setdefault((lg.carrier, lg.origin), []).append(lg)
    return lambda c, a: idx.get((c, a), ())


def no_prior(_):
    return None


def test_walk_across_two_links():
    a = leg("a", fn="100", o="AAA", d="BBB", so=0)
    b = leg("b", fn="100", o="BBB", d="CCC", so=3 * H)       # same flight number continuing
    c = leg("c", fn="200", o="CCC", d="DDD", so=6 * H)       # already filed with the same aircraft
    ident = {"a": X, "c": X}.get
    steps = cr.walk(a, X, deps_of([a, b, c]), ident, no_prior)
    assert [(s.leg.id, s.method) for s in steps] == [("b", "m2"), ("c", "direct")]


def test_exclusion_of_departure_filed_with_another_aircraft():
    a = leg("a", so=0)
    b1 = leg("b1", fn="100", o="BBB", d="CCC", so=3 * H)
    b2 = leg("b2", fn="100", o="BBB", d="DDD", so=4 * H)
    dep = deps_of([a, b1, b2])
    ident = {"a": X, "b1": Y}.get                                 # b1 belongs to another aircraft
    assert [s.leg.id for s in cr.walk(a, X, dep, ident, no_prior, max_steps=1)] == ["b2"]
    ident = {"a": X, "b1": Y, "b2": Y}.get
    assert cr.walk(a, X, dep, ident, no_prior) == []


def test_as_of_ignores_fact_one_second_after_cutoff():
    st = AsOfState([(100.0, "l1", X, "sfdps"), (101.0, "l1", Y, "sfdps"), (101.0, "l2", Y, "smes")])
    st.advance(100.0)
    assert st.ident("l1") == X and st.ident_sfdps("l1") == X and st.ident("l2") is None
    st.advance(101.0)
    assert st.ident("l1") == Y and st.ident("l2") == Y and st.ident_sfdps("l2") is None


def test_ua_excluded_from_m2_but_allowed_carrier_works():
    for carrier, expect in (("UAL", []), ("DAL", ["b"])):
        a = leg("a", carrier=carrier, so=0)
        b = leg("b", carrier=carrier, fn="100", o="BBB", d="CCC", so=3 * H)
        got = cr.walk(a, X, deps_of([a, b]), {"a": X}.get, no_prior)
        assert [s.leg.id for s in got] == expect
    assert not {"UAL", "GJS", "UCA", "ASH", "AAY"} & cr.M2_ALLOW


def test_type_family_filter():
    assert cr.type_family("B738") == cr.type_family("B39M") == "B737"
    assert cr.type_family("A21N") != cr.type_family("B38M")
    assert cr.type_family("") == "" and cr.family_ok("", "A320")
    a = leg("a", so=0, fam="A320")
    b = leg("b", fn="100", o="BBB", d="CCC", so=3 * H, fam="B737")
    assert cr.walk(a, X, deps_of([a, b]), {"a": X}.get, no_prior) == []
    b2 = b._replace(fam="A320")
    assert [s.leg.id for s in cr.walk(a, X, deps_of([a, b2]), {"a": X}.get, no_prior)] == ["b"]


def test_walk_caps_steps_and_span():
    ports = ["P%d" % i for i in range(12)]
    legs = [leg("l%d" % i, fn="100", o=ports[i], d=ports[i + 1], so=i * 2.5 * H, dur=2 * H) for i in range(10)]
    ident = {"l0": X}.get
    steps = cr.walk(legs[0], X, deps_of(legs), ident, no_prior)
    assert len(steps) == 6                                         # 6 steps (15 h) fit in 18 h
    legs = [leg("l%d" % i, fn="100", o=ports[i], d=ports[i + 1], so=i * 5 * H, dur=4 * H) for i in range(10)]
    steps = cr.walk(legs[0], X, deps_of(legs), ident, no_prior)
    assert len(steps) == 3                                         # legs at 5, 10, 15 h; the 20 h one is past 18 h


def test_prior_step_confidence_needs_pattern_match():
    a = leg("a", carrier="UAL", fn="100", so=0)
    b = leg("b", carrier="UAL", fn="777", o="BBB", d="CCC", so=3 * H)
    dep = deps_of([a, b])
    prior = lambda cur: "777" if cur.id == "a" else None
    lo = cr.walk(a, X, dep, {"a": X}.get, prior, prev_matched=False)
    hi = cr.walk(a, X, dep, {"a": X}.get, prior, prev_matched=True)
    assert [(s.method, s.high) for s in lo] == [("prior", False)]
    assert [(s.method, s.high) for s in hi] == [("prior", True)]


def test_priors_use_only_days_before_the_measured_day():
    d0 = datetime(2026, 11, 9, 15, tzinfo=timezone.utc).timestamp()
    legs = [leg("a1", fn="1", o="AAA", d="BBB", so=d0, dur=H),
            leg("b1", fn="2", o="BBB", d="CCC", so=d0 + 2 * H),
            leg("a2", fn="1", o="AAA", d="BBB", so=d0 + 86400, dur=H),
            leg("b2", fn="9", o="BBB", d="CCC", so=d0 + 86400 + 2 * H)]
    truth = {i: X for i in ("a1", "b1", "a2", "b2")}
    day = po.op_day(d0)
    pri = po.build(legs, truth, day + 1)                           # day 0 only
    assert pri.next_fn("DAL", "1", "BBB", day + 1) == "2"          # day before
    assert pri.prev_fn("DAL", "2", "BBB", day + 1) == "1"
    assert pri.next_fn("DAL", "1", "BBB", day + 7) == "2"          # same weekday a week earlier
    pri = po.build(legs, truth, day)                               # nothing earlier
    assert pri.next_fn("DAL", "1", "BBB", day + 1) is None


# ------------------------------------------------------------------ synthetic day through the whole study
def synthetic():
    base = datetime(2026, 11, 10, tzinfo=timezone.utc).timestamp()
    t = lambda h: base + h * H
    legs = [leg("L1", fn="100", o="ATL", d="JFK", so=t(15), dur=2 * H, fam="B737"),
            leg("L2", fn="100", o="JFK", d="ATL", so=t(18), dur=2 * H, fam="B737"),
            leg("L3", fn="300", o="ATL", d="BOS", so=t(21), dur=2 * H, fam="B737"),
            leg("M1", fn="500", o="ORD", d="LAX", so=t(16), dur=4 * H, fam="A320")]
    truth = {"L1": "HEXAAA", "L2": "HEXAAA", "L3": "HEXAAA", "M1": "HEXBBB"}
    obs = [(t(12), "L1", "HEXAAA", "sfdps"), (t(13), "M1", "HEXBBB", "sfdps"),
           (t(10), "L2", "HEXAAA", "smes")]                          # smes only: not an M1 fact
    return legs, set(), obs, truth


def test_study_end_to_end_and_no_identifiers_in_output():
    legs, cancelled, obs, truth = synthetic()
    res = cm.measure(legs, cancelled, obs, truth, [date(2026, 11, 10)], [180, 60], tick=300)
    r60 = res["results"]["60"]["ALL"]
    assert r60["n"] == 4
    assert r60["by_method"]["M1"]["linked_share"] == 0.5          # L1 and M1 carry their own SFDPS fact
    assert r60["by_method"]["M4"]["precision"] == 1.0             # L2 reached by the walk from L1
    assert r60["effective"] >= 0.75
    text = json.dumps(res)
    assert "HEX" not in text and "icao24" not in text and "reg" not in text.replace("regional", "")


def test_decision_rule():
    row = lambda eff, prec: {"ALL": {"effective": eff, "precision": prec}}
    for r3, r1, verdict in (((0.5, 0.8), (0.7, 0.9), "GO"), ((0.4, 0.8), (0.7, 0.9), "T-1h-only GO"),
                            ((0.5, 0.8), (0.69, 0.95), "STOP"), ((0.4, 0.5), (0.5, 0.5), "STOP")):
        res = {"180": row(*r3), "60": row(*r1)}
        assert cm.decide(res, [180, 60])["verdict"] == verdict


def test_brier_interpolation_points():
    assert cm.interp(cm.A7[60], 0.5) == pytest.approx(0.022)
    assert cm.interp(cm.A7[180], 0.29) == pytest.approx(0.009)
    assert cm.interp(cm.A7[180], 1.0) == pytest.approx(0.031)
    assert 0.015 < cm.interp(cm.A7[180], 0.6) < 0.023


STORE = os.environ.get("FLYRELY_LINKAGE_STORE", "data/linkage/live.sqlite")


@pytest.mark.skipif(not os.path.exists(STORE), reason="live linkage store not present (CI)")
def test_store_loads_read_only():
    db = cm.open_ro(STORE)
    legs, cancelled, obs, truth = cm.load(db)
    assert legs and obs and truth
    with pytest.raises(Exception):
        db.execute("CREATE TABLE should_fail(x)")
