"""copilot/links.py and copilot/partners.py: plain and affiliate hand-off URLs.

No network: partner URLs are asserted as strings, never fetched. Environments are passed
explicitly (never os.environ) so tests do not depend on the developer's shell.
"""

import logging
from datetime import date
from urllib.parse import parse_qs, quote, unquote, urlsplit

import pytest

from copilot import links as L
from copilot import partners as P

CIN = date(2026, 11, 15)
COUT = date(2026, 11, 17)
LAT, LON = 40.6413, -73.7781

PLAIN_FLIGHTS = {
    "google_flights": "https://www.google.com/travel/flights?q=Flights+from+JFK+to+LAX+on+2026-11-15",
    "kayak": "https://www.kayak.com/flights/JFK-LAX/2026-11-15",
    "skyscanner": "https://www.skyscanner.com/transport/flights/jfk/lax/261115/",
    "expedia": "https://www.expedia.com/Flights-Search?trip=oneway&leg1=from:JFK,to:LAX,"
               "departure:11/15/2026TANYT&passengers=adults:1&mode=search",
    "travelpayouts": "https://www.aviasales.com/search/JFK1511LAX1",
}
PLAIN_HOTELS = {
    "booking": "https://www.booking.com/searchresults.html?latitude=40.6413&longitude=-73.7781"
               "&checkin=2026-11-15&checkout=2026-11-17&group_adults=1",
    "expedia": "https://www.expedia.com/Hotel-Search?latLong=40.6413,-73.7781&startDate=11/15/2026"
               "&endDate=11/17/2026&adults=1&rooms=1",
    "stay22": "https://www.stay22.com/allez/map?lat=40.6413&lng=-73.7781&checkin=2026-11-15"
              "&checkout=2026-11-17&adults=1",
    "google_maps": "https://www.google.com/maps/search/hotels+near+JFK+airport",
}
PLAIN_RIDES = {
    "uber": "https://m.uber.com/ul/?action=setPickup&pickup=my_location",
    "lyft": "https://lyft.com/ride?id=lyft",
}

# Partners that have an AFFILIATE_<P> mechanism, with the key used in the env var name.
AFFILIATE_KEYS = {
    "kayak": "KAYAK",
    "skyscanner": "SKYSCANNER",
    "expedia": "EXPEDIA",
    "travelpayouts": "TRAVELPAYOUTS",
    "stay22": "STAY22",
    "booking": "BOOKING",
    "uber": "UBER",
    "lyft": "LYFT",
}
NO_AFFILIATE = ("google_flights", "google_maps")


@pytest.fixture(autouse=True)
def _reset_warnings():
    L._warned.clear()
    yield
    L._warned.clear()


# The helpers list every partner so each one can be exercised; unconfigured affiliate-only
# partners are still dropped by the builder, which the tests assert on directly.
ALL_FLIGHTS = ",".join(P.FLIGHT_PARTNERS)
ALL_HOTELS = ",".join(P.HOTEL_PARTNERS)


def _flight(env):
    env = {"FLYRELY_COPILOT_FLIGHT_PARTNERS": ALL_FLIGHTS, **env}
    return {link.partner: link for link in L.flight_links("jfk", "LAX", "2026-11-15", env=env)}


def _hotel(env):
    env = {"FLYRELY_COPILOT_HOTEL_PARTNERS": ALL_HOTELS, **env}
    return {link.partner: link for link in L.hotel_links("JFK", LAT, LON, CIN, COUT, env=env)}


def _ride(env, dropoff=None):
    return {link.partner: link for link in L.ride_links(dropoff, env=env)}


# --- plain links (no affiliate env) ------------------------------------------------

@pytest.mark.parametrize("partner,url", PLAIN_FLIGHTS.items())
def test_flight_plain_url_when_ids_absent(partner, url):
    if partner == "travelpayouts":
        # Affiliate-only: skipped when unconfigured, so it only appears when listed AND set.
        assert partner not in _flight({})
        return
    link = _flight({})[partner]
    assert link.url == url
    assert link.is_affiliate is False


@pytest.mark.parametrize("partner,url", PLAIN_HOTELS.items())
def test_hotel_plain_url_when_ids_absent(partner, url):
    if partner == "stay22":
        assert partner not in _hotel({})  # affiliate-only
        return
    link = _hotel({})[partner]
    assert link.url == url
    assert link.is_affiliate is False


@pytest.mark.parametrize("partner,url", PLAIN_RIDES.items())
def test_ride_plain_url_when_ids_absent(partner, url):
    link = _ride({})[partner]
    assert link.url == url
    assert link.is_affiliate is False


def test_default_order_when_env_unset():
    assert [l.partner for l in L.flight_links("JFK", "LAX", CIN, env={})] == [
        "google_flights", "kayak", "skyscanner", "expedia",
    ]
    assert [l.partner for l in L.hotel_links("JFK", LAT, LON, CIN, COUT, env={})] == [
        "booking", "expedia", "google_maps",  # stay22 skipped: affiliate-only, unconfigured
    ]
    assert [l.partner for l in L.ride_links(env={})] == ["uber", "lyft"]


def test_ride_dropoff_coordinates_are_encoded():
    uber = _ride({}, dropoff=(LAT, LON))["uber"]
    assert uber.url == (
        "https://m.uber.com/ul/?action=setPickup&pickup=my_location"
        "&dropoff[latitude]=40.6413&dropoff[longitude]=-73.7781"
    )
    lyft = _ride({}, dropoff=(LAT, LON))["lyft"]
    assert lyft.url == (
        "https://lyft.com/ride?id=lyft&destination[latitude]=40.6413&destination[longitude]=-73.7781"
    )


def test_accepts_dropoff_address_flag_only_on_rides():
    rides = _ride({})
    assert rides["uber"].to_dict()["accepts_dropoff_address"] is True
    assert rides["lyft"].to_dict()["accepts_dropoff_address"] is False
    assert "accepts_dropoff_address" not in _flight({})["kayak"].to_dict()


def test_google_partners_have_no_affiliate_mechanism_even_if_env_is_set():
    env = {
        "AFFILIATE_GOOGLE_FLIGHTS_PARAMS": "aid=1",
        "AFFILIATE_GOOGLE_MAPS_WRAP": "https://x.example/?u={url}",
    }
    flights, hotels = _flight(env), _hotel(env)
    assert flights["google_flights"].url == PLAIN_FLIGHTS["google_flights"]
    assert hotels["google_maps"].url == PLAIN_HOTELS["google_maps"]
    assert not flights["google_flights"].is_affiliate
    assert not hotels["google_maps"].is_affiliate


# --- IDs present: PARAMS -----------------------------------------------------------

@pytest.mark.parametrize("partner", [p for p in AFFILIATE_KEYS if p in PLAIN_FLIGHTS])
def test_flight_params_appended_when_set(partner):
    key = AFFILIATE_KEYS[partner]
    link = _flight({f"AFFILIATE_{key}_PARAMS": "aid=123456"})[partner]
    base = PLAIN_FLIGHTS[partner]
    sep = "&" if "?" in base else "?"
    assert link.url == f"{base}{sep}aid=123456"
    assert link.is_affiliate is True


@pytest.mark.parametrize("partner", [p for p in AFFILIATE_KEYS if p in PLAIN_HOTELS])
def test_hotel_params_appended_when_set(partner):
    key = AFFILIATE_KEYS[partner]
    link = _hotel({f"AFFILIATE_{key}_PARAMS": "aid=123456"})[partner]
    base = PLAIN_HOTELS[partner]
    sep = "&" if "?" in base else "?"
    assert link.url == f"{base}{sep}aid=123456"
    assert link.is_affiliate is True


@pytest.mark.parametrize("partner", ["uber", "lyft"])
def test_ride_params_appended_when_set(partner):
    key = AFFILIATE_KEYS[partner]
    link = _ride({f"AFFILIATE_{key}_PARAMS": "client_id=abc"})[partner]
    base = PLAIN_RIDES[partner]
    sep = "&" if "?" in base else "?"
    assert link.url == f"{base}{sep}client_id=abc"
    assert link.is_affiliate is True


def test_params_leading_separator_is_stripped():
    link = _flight({"AFFILIATE_KAYAK_PARAMS": "?aid=9"})["kayak"]
    assert link.url == PLAIN_FLIGHTS["kayak"] + "?aid=9"
    link = _hotel({"AFFILIATE_BOOKING_PARAMS": "&aid=9"})["booking"]
    assert link.url == PLAIN_HOTELS["booking"] + "&aid=9"


def test_blank_params_count_as_unset():
    link = _flight({"AFFILIATE_KAYAK_PARAMS": "   ?  "})["kayak"]
    assert link.url == PLAIN_FLIGHTS["kayak"]
    assert link.is_affiliate is False


# --- IDs present: WRAP -------------------------------------------------------------

@pytest.mark.parametrize("partner", [p for p in AFFILIATE_KEYS if p in PLAIN_FLIGHTS])
def test_flight_wrap_encodes_destination(partner):
    key = AFFILIATE_KEYS[partner]
    wrap = "https://tp.media/r?marker=123&u={url}"
    link = _flight({f"AFFILIATE_{key}_WRAP": wrap})[partner]
    base = PLAIN_FLIGHTS[partner]
    assert link.url == "https://tp.media/r?marker=123&u=" + _quote(base)
    assert link.is_affiliate is True


@pytest.mark.parametrize("partner", [p for p in AFFILIATE_KEYS if p in PLAIN_HOTELS])
def test_hotel_wrap_encodes_destination(partner):
    key = AFFILIATE_KEYS[partner]
    wrap = "https://tp.media/r?marker=123&u={url}"
    link = _hotel({f"AFFILIATE_{key}_WRAP": wrap})[partner]
    assert link.url == "https://tp.media/r?marker=123&u=" + _quote(PLAIN_HOTELS[partner])
    assert link.is_affiliate is True


@pytest.mark.parametrize("partner", ["uber", "lyft"])
def test_ride_wrap_round_trips_destination(partner):
    key = AFFILIATE_KEYS[partner]
    link = _ride({f"AFFILIATE_{key}_WRAP": "https://track.example/go?to={url}"}, dropoff=(LAT, LON))[partner]
    parts = urlsplit(link.url)
    assert parts.netloc == "track.example"
    # The destination survives a decode unchanged, including its own & and = characters.
    assert unquote(parts.query.split("to=", 1)[1]) == _ride({}, dropoff=(LAT, LON))[partner].url


def test_wrap_encodes_reserved_characters_in_destination():
    link = _hotel({"AFFILIATE_BOOKING_WRAP": "https://w.example/?u={url}"})["booking"]
    encoded = link.url.split("?u=", 1)[1]
    assert "&" not in encoded and "?" not in encoded and " " not in encoded
    assert parse_qs(urlsplit(link.url).query) == {"u": [PLAIN_HOTELS["booking"]]}


def test_wrap_wins_over_params_and_warns(caplog):
    env = {
        "AFFILIATE_KAYAK_PARAMS": "aid=1",
        "AFFILIATE_KAYAK_WRAP": "https://w.example/?u={url}",
    }
    with caplog.at_level(logging.WARNING, logger="copilot.links"):
        link = _flight(env)["kayak"]
    assert link.url.startswith("https://w.example/?u=")
    assert "aid=1" not in link.url
    assert any("using WRAP" in r.getMessage() for r in caplog.records)


def test_wrap_without_placeholder_is_ignored_with_warning(caplog):
    env = {"AFFILIATE_KAYAK_WRAP": "https://w.example/no-placeholder"}
    with caplog.at_level(logging.WARNING, logger="copilot.links"):
        link = _flight(env)["kayak"]
    assert link.url == PLAIN_FLIGHTS["kayak"]
    assert link.is_affiliate is False
    assert any("no {url} placeholder" in r.getMessage() for r in caplog.records)


def test_warning_is_logged_once_per_partner(caplog):
    env = {"AFFILIATE_KAYAK_PARAMS": "a=1", "AFFILIATE_KAYAK_WRAP": "https://w/{url}"}
    with caplog.at_level(logging.WARNING, logger="copilot.links"):
        _flight(env)
        _flight(env)
    assert len([r for r in caplog.records if "both set" in r.getMessage() or "using WRAP" in r.getMessage()]) == 1


# --- affiliate-only partners -------------------------------------------------------

def test_stay22_included_when_configured_and_placed_by_order():
    env = {"AFFILIATE_STAY22_PARAMS": "aid=stay1"}
    links = L.hotel_links("JFK", LAT, LON, CIN, COUT, env=env)
    assert [l.partner for l in links] == ["stay22", "booking", "expedia", "google_maps"]
    assert links[0].url == PLAIN_HOTELS["stay22"] + "&aid=stay1"
    assert links[0].is_affiliate is True


def test_travelpayouts_only_when_listed_and_configured():
    env = {"FLYRELY_COPILOT_FLIGHT_PARTNERS": "travelpayouts,kayak"}
    assert [l.partner for l in L.flight_links("JFK", "LAX", CIN, env=env)] == ["kayak"]
    env["AFFILIATE_TRAVELPAYOUTS_WRAP"] = "https://tp.media/r?marker=7&u={url}"
    links = L.flight_links("JFK", "LAX", CIN, env=env)
    assert [l.partner for l in links] == ["travelpayouts", "kayak"]
    assert links[0].url == "https://tp.media/r?marker=7&u=" + _quote(PLAIN_FLIGHTS["travelpayouts"])


# --- partner order -----------------------------------------------------------------

def test_custom_flight_order_and_google_alias():
    env = {"FLYRELY_COPILOT_FLIGHT_PARTNERS": " Expedia , google ,kayak"}
    assert [l.partner for l in L.flight_links("JFK", "LAX", CIN, env=env)] == [
        "expedia", "google_flights", "kayak",
    ]


def test_unknown_partner_names_are_ignored_with_warning(caplog):
    env = {"FLYRELY_COPILOT_FLIGHT_PARTNERS": "kayak,nosuchthing,uber"}
    with caplog.at_level(logging.WARNING, logger="copilot.links"):
        links = L.flight_links("JFK", "LAX", CIN, env=env)
    assert [l.partner for l in links] == ["kayak"]  # uber is a ride partner, not a flight one
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "nosuchthing" in messages and "uber" in messages


def test_order_with_only_unknown_names_falls_back_to_defaults(caplog):
    env = {"FLYRELY_COPILOT_HOTEL_PARTNERS": "bogus"}
    with caplog.at_level(logging.WARNING, logger="copilot.links"):
        links = L.hotel_links("JFK", LAT, LON, CIN, COUT, env=env)
    assert [l.partner for l in links] == ["booking", "expedia", "google_maps"]
    assert any("using defaults" in r.getMessage() for r in caplog.records)


def test_duplicate_partner_names_are_collapsed():
    env = {"FLYRELY_COPILOT_FLIGHT_PARTNERS": "kayak,kayak,google,google_flights"}
    assert [l.partner for l in L.flight_links("JFK", "LAX", CIN, env=env)] == ["kayak", "google_flights"]


# --- input validation and dates ----------------------------------------------------

@pytest.mark.parametrize("bad", ["JF", "JFKK", "J1K", "", "  "])
def test_bad_airport_codes_rejected(bad):
    with pytest.raises(ValueError):
        L.flight_links(bad, "LAX", CIN, env={})


def test_same_origin_and_destination_rejected():
    with pytest.raises(ValueError):
        L.flight_links("JFK", "jfk", CIN, env={})


def test_checkout_must_follow_checkin():
    with pytest.raises(ValueError):
        L.hotel_links("JFK", LAT, LON, COUT, CIN, env={})


@pytest.mark.parametrize("lat,lon", [(91, 0), (0, 181), (float("nan"), 0), (0, float("inf"))])
def test_bad_coordinates_rejected(lat, lon):
    with pytest.raises(ValueError):
        L.hotel_links("JFK", lat, lon, CIN, COUT, env={})
    with pytest.raises(ValueError):
        L.ride_links((lat, lon), env={})


def test_iso_string_dates_accepted():
    a = _flight({})["skyscanner"].url
    assert a == L.flight_links("JFK", "LAX", "2026-11-15", env={})[2].url
    assert a == PLAIN_FLIGHTS["skyscanner"]


def test_link_dict_shape():
    d = _flight({})["kayak"].to_dict()
    assert set(d) == {"partner", "label", "url", "is_affiliate"}
    assert d["label"] == "KAYAK"


# --- helpers -----------------------------------------------------------------------

def _quote(s):
    return quote(s, safe="")


def test_every_partner_in_registry_is_reachable_from_some_category():
    listed = set(P.FLIGHT_PARTNERS) | set(P.HOTEL_PARTNERS) | set(P.RIDE_PARTNERS)
    assert listed == set(P.PARTNERS)
