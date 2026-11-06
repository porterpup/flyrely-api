"""Hand-off link builder: plain or affiliate URL per partner (plan section 3.6).

Affiliate config is read from the environment on every call (so links change
without a code deploy):

- ``AFFILIATE_<P>_PARAMS``: query string appended to the partner URL.
- ``AFFILIATE_<P>_WRAP``: tracking redirect containing ``{url}``, which is replaced
  with the URL-encoded destination.
- If both are set, WRAP wins and a warning is logged once per process.
- If neither is set, the plain link is returned with ``is_affiliate=False``.
  Affiliate-only partners (Stay22, Travelpayouts) are skipped instead.

Partner order comes from ``FLYRELY_COPILOT_FLIGHT_PARTNERS`` and
``FLYRELY_COPILOT_HOTEL_PARTNERS`` (comma lists). Unknown names are ignored
with a warning.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from urllib.parse import quote

from . import partners as P

log = logging.getLogger(__name__)

FLIGHT_ORDER_ENV = "FLYRELY_COPILOT_FLIGHT_PARTNERS"
HOTEL_ORDER_ENV = "FLYRELY_COPILOT_HOTEL_PARTNERS"
DEFAULT_FLIGHT_ORDER = ("google_flights", "kayak", "skyscanner", "expedia")
DEFAULT_HOTEL_ORDER = ("stay22", "booking", "expedia", "google_maps")
FLIGHT_ALIASES = {"google": "google_flights"}

_warned: set[str] = set()


@dataclass(frozen=True)
class Link:
    partner: str
    label: str
    url: str
    is_affiliate: bool
    accepts_dropoff_address: bool | None = None

    def to_dict(self) -> dict:
        d: dict = {
            "partner": self.partner,
            "label": self.label,
            "url": self.url,
            "is_affiliate": self.is_affiliate,
        }
        if self.accepts_dropoff_address is not None:
            d["accepts_dropoff_address"] = self.accepts_dropoff_address
        return d


def _warn_once(key: str, msg: str, *args: object) -> None:
    if key in _warned:
        return
    _warned.add(key)
    log.warning(msg, *args)


def _clean(value: str | None) -> str:
    return (value or "").strip()


def _apply_affiliate(
    partner_id: str, plain_url: str, env: Mapping[str, str]
) -> tuple[str, bool] | None:
    """Return (url, is_affiliate), or None when the partner must be skipped."""
    key = P.PARTNERS[partner_id].affiliate_key
    if key is None:
        return plain_url, False

    params = _clean(env.get(f"AFFILIATE_{key}_PARAMS")).lstrip("?&").strip()
    wrap = _clean(env.get(f"AFFILIATE_{key}_WRAP"))
    if wrap and "{url}" not in wrap:
        _warn_once(f"nourl:{key}", "AFFILIATE_%s_WRAP has no {url} placeholder; ignored", key)
        wrap = ""
    if wrap and params:
        _warn_once(
            f"both:{key}",
            "AFFILIATE_%s_PARAMS and AFFILIATE_%s_WRAP are both set; using WRAP",
            key,
            key,
        )

    if wrap:
        return wrap.replace("{url}", quote(plain_url, safe="")), True
    if params:
        sep = "&" if "?" in plain_url else "?"
        return f"{plain_url}{sep}{params}", True
    if P.PARTNERS[partner_id].affiliate_only:
        return None
    return plain_url, False


def _build(partner_id: str, plain_url: str, env: Mapping[str, str]) -> Link | None:
    applied = _apply_affiliate(partner_id, plain_url, env)
    if applied is None:
        return None
    url, is_affiliate = applied
    spec = P.PARTNERS[partner_id]
    accepts = spec.accepts_dropoff_address if partner_id in P.RIDE_PARTNERS else None
    return Link(partner_id, spec.label, url, is_affiliate, accepts)


def _order(
    env_name: str,
    default: tuple[str, ...],
    allowed: tuple[str, ...],
    env: Mapping[str, str],
    aliases: Mapping[str, str] | None = None,
) -> list[str]:
    raw = _clean(env.get(env_name))
    if not raw:
        return list(default)
    out: list[str] = []
    for item in raw.split(","):
        name = item.strip().lower()
        if not name:
            continue
        name = (aliases or {}).get(name, name)
        if name not in allowed:
            _warn_once(f"unknown:{env_name}:{name}", "%s: unknown partner %r ignored", env_name, name)
            continue
        if name not in out:
            out.append(name)
    if not out:
        _warn_once(f"empty:{env_name}", "%s has no valid partners; using defaults", env_name)
        return list(default)
    return out


def _collect(
    order: list[str], url_for: Callable[[str], str], env: Mapping[str, str]
) -> list[Link]:
    links = []
    for partner_id in order:
        link = _build(partner_id, url_for(partner_id), env)
        if link is not None:
            links.append(link)
    return links


def _as_date(value: date | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        return date.fromisoformat(value)
    return value


def flight_links(
    origin: str, destination: str, on: date | str, env: Mapping[str, str] = os.environ
) -> list[Link]:
    o = P.airport_code(origin)
    d = P.airport_code(destination)
    if o == d:
        raise ValueError("origin and destination must differ")
    day = _as_date(on)
    builders: dict[str, Callable[[], str]] = {
        "google_flights": lambda: P.google_flights_url(o, d, day),
        "kayak": lambda: P.kayak_flights_url(o, d, day),
        "skyscanner": lambda: P.skyscanner_flights_url(o, d, day),
        "expedia": lambda: P.expedia_flights_url(o, d, day),
        "travelpayouts": lambda: P.travelpayouts_flights_url(o, d, day),
    }
    order = _order(FLIGHT_ORDER_ENV, DEFAULT_FLIGHT_ORDER, P.FLIGHT_PARTNERS, env, FLIGHT_ALIASES)
    return _collect(order, lambda pid: builders[pid](), env)


def hotel_links(
    airport: str,
    lat: float,
    lon: float,
    checkin: date | str,
    checkout: date | str,
    env: Mapping[str, str] = os.environ,
) -> list[Link]:
    code = P.airport_code(airport)
    P.check_latlon(lat, lon)
    cin, cout = _as_date(checkin), _as_date(checkout)
    P.check_stay(cin, cout)
    builders: dict[str, Callable[[], str]] = {
        "stay22": lambda: P.stay22_hotels_url(lat, lon, cin, cout),
        "booking": lambda: P.booking_hotels_url(lat, lon, cin, cout),
        "expedia": lambda: P.expedia_hotels_url(lat, lon, cin, cout),
        "google_maps": lambda: P.google_maps_hotels_url(code),
    }
    order = _order(HOTEL_ORDER_ENV, DEFAULT_HOTEL_ORDER, P.HOTEL_PARTNERS, env)
    return _collect(order, lambda pid: builders[pid](), env)


def ride_links(
    dropoff: tuple[float, float] | None = None, env: Mapping[str, str] = os.environ
) -> list[Link]:
    if dropoff is not None:
        P.check_latlon(*dropoff)
    builders: dict[str, Callable[[], str]] = {
        "uber": lambda: P.uber_url(dropoff),
        "lyft": lambda: P.lyft_url(dropoff),
    }
    return _collect(list(P.RIDE_PARTNERS), lambda pid: builders[pid](), env)
