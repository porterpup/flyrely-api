"""Free, keyless aviation weather from the NOAA aviationweather.gov Data API (METAR + TAF).

Weather is NOT a model input. It only feeds origin_weather / destination_weather and the risk-factor text.
Pure parsing/selection functions live here so they can be tested with recorded fixtures (no network).

Units in the API: wind in knots, visibility in statute miles (may be "10+" / "6+" / "1 1/2"), temperature in C,
times as epoch seconds (UTC). TAFs carry no temperature, so a forecast block reuses the latest observed
temperature.
"""

import asyncio
import logging
import re
import time
from datetime import datetime, timezone
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://aviationweather.gov/api/data"
USER_AGENT = "FlyRely/1.0"
TIMEOUT_SECONDS = 3.0
FAILURE_COOLDOWN_SECONDS = 60  # after a failed fetch, skip that product this long so a stall costs one request
METAR_TTL_SECONDS = 10 * 60
TAF_TTL_SECONDS = 30 * 60
KT_TO_MPH = 1.15078
LOW_CEILING_FT = 1000

_WX_WORDS = {
    "TS": "thunderstorm", "SH": "showers", "FZ": "freezing", "DR": "drifting", "BL": "blowing",
    "MI": "shallow", "BC": "patches of", "PR": "partial",
    "RA": "rain", "SN": "snow", "DZ": "drizzle", "GR": "hail", "GS": "small hail", "PL": "ice pellets",
    "SG": "snow grains", "IC": "ice crystals", "UP": "precipitation", "BR": "mist", "FG": "fog",
    "FU": "smoke", "HZ": "haze", "DU": "dust", "SA": "sand", "VA": "volcanic ash", "SQ": "squalls",
    "FC": "funnel cloud", "SS": "sandstorm", "DS": "duststorm",
}
_COVER_WORDS = {"CLR": "clear", "SKC": "clear", "FEW": "few clouds", "SCT": "scattered clouds",
                "BKN": "broken clouds", "OVC": "overcast", "VV": "sky obscured"}
_COVER_ORDER = ["CLR", "SKC", "FEW", "SCT", "BKN", "OVC", "VV"]
_WINTRY = re.compile(r"SN|SG|PL|IC|GR|GS|FZ")


def parse_visibility(value) -> Optional[float]:
    """Statute miles from the API's visib field: 10, "10+", "6+", "1/2", "1 1/2", "" -> float or None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().upper().replace("P", "").replace("SM", "").rstrip("+").strip()
    if not s:
        return None
    try:
        total = 0.0
        for part in s.split():
            if "/" in part:
                num, den = part.split("/")
                total += float(num) / float(den)
            else:
                total += float(part)
        return total
    except (ValueError, ZeroDivisionError):
        return None


def _num(value) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def ceiling_ft(clouds, vert_vis=None) -> Optional[int]:
    """Lowest BKN/OVC/VV layer base in feet (or vertical visibility), None when there is no ceiling."""
    bases = []
    for layer in clouds or []:
        layer = layer or {}
        base = _num(layer.get("base"))
        if layer.get("cover") in ("BKN", "OVC", "VV") and base is not None:
            bases.append(int(base))
    vv = _num(vert_vis)
    if vv is not None:
        bases.append(int(vv))
    return min(bases) if bases else None


def sky_phrase(clouds) -> Optional[str]:
    covers = [(layer or {}).get("cover") for layer in clouds or []]
    covers = [c for c in covers if c in _COVER_ORDER]
    if not covers:
        return None
    return _COVER_WORDS[max(covers, key=_COVER_ORDER.index)]


def decode_wx(wx: Optional[str]) -> list[str]:
    """'-TSRA BR' -> ['light thunderstorm rain', 'mist'] (readable enough for a conditions string)."""
    out = []
    for token in (wx or "").split():
        intensity = {"-": "light ", "+": "heavy "}.get(token[0], "")
        body = token.lstrip("+-")
        if body.startswith("VC"):
            intensity, body = "nearby ", body[2:]
        words = [_WX_WORDS[body[i:i + 2]] for i in range(0, len(body) - 1, 2) if body[i:i + 2] in _WX_WORDS]
        if words:
            out.append(intensity + " ".join(words))
    return out


def select_taf_group(taf: dict, when_epoch: float) -> Optional[dict]:
    """The TAF block covering `when_epoch`, or None when the time is outside the TAF's valid period.

    Base/FM/BECMG groups are sequential: the one that started most recently wins (BECMG counts from the end
    of its transition). TEMPO/PROB groups overlay it: where they cover the time they are merged in worst-case
    (lower visibility and ceiling, extra weather phenomena, stronger wind).
    """
    start, end = _num(taf.get("validTimeFrom")), _num(taf.get("validTimeTo"))
    if start is None or end is None or not (start <= when_epoch < end):
        return None
    groups = [g for g in taf.get("fcsts") or [] if isinstance(g, dict)]

    def covers(g):
        a, b = _num(g.get("timeFrom")), _num(g.get("timeTo"))
        return a is not None and b is not None and a <= when_epoch < b

    def effective_start(g):
        if g.get("fcstChange") == "BECMG" and _num(g.get("timeBec")) is not None:
            return _num(g.get("timeBec"))
        return _num(g.get("timeFrom"))

    sequential = [g for g in groups if g.get("fcstChange") not in ("TEMPO", "PROB") and covers(g)]
    started = [g for g in sequential if effective_start(g) <= when_epoch]
    pool = started or sequential
    if not pool:
        return None
    base = dict(max(pool, key=effective_start))
    for g in (g for g in groups if g.get("fcstChange") in ("TEMPO", "PROB") and covers(g)):
        _merge_overlay(base, g)
    return base


def _merge_overlay(base: dict, ov: dict) -> None:
    vb, vo = parse_visibility(base.get("visib")), parse_visibility(ov.get("visib"))
    if vo is not None and (vb is None or vo < vb):
        base["visib"] = ov.get("visib")
    wb, wo = _num(base.get("wspd")), _num(ov.get("wspd"))
    if wo is not None and (wb is None or wo > wb):
        base["wspd"] = wo
    if ov.get("wxString"):
        base["wxString"] = f"{base.get('wxString') or ''} {ov['wxString']}".strip()
    cb, co = ceiling_ft(base.get("clouds"), base.get("vertVis")), ceiling_ft(ov.get("clouds"), ov.get("vertVis"))
    if co is not None and (cb is None or co < cb):
        base["clouds"], base["vertVis"] = ov.get("clouds"), ov.get("vertVis")


def _iso(epoch) -> Optional[str]:
    e = _num(epoch)
    return datetime.fromtimestamp(e, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if e is not None else None


def compose_weather(metar: Optional[dict], taf: Optional[dict], when: Optional[datetime]) -> Optional[dict]:
    """WeatherData-shaped dict (plus optional extras), or None when there is not enough data.

    `when` is a timezone-aware time. Forecast block when it falls in the TAF's valid period, else current.
    Temperature is always the latest observed value (TAFs have none), so no METAR means no result.
    """
    metar = metar or {}
    temp_c = _num(metar.get("temp"))
    if temp_c is None:
        return None
    when_epoch = when.timestamp() if when is not None else None
    group = select_taf_group(taf, when_epoch) if (taf and when_epoch is not None) else None

    m_wind, m_vis = _num(metar.get("wspd")), parse_visibility(metar.get("visib"))
    if group is not None:
        source, as_of = "forecast", _iso(when_epoch)
        wind, vis = _num(group.get("wspd")), parse_visibility(group.get("visib"))
        wx, clouds, vv = group.get("wxString"), group.get("clouds"), group.get("vertVis")
        if wind is None:
            wind = m_wind
        if vis is None:
            vis = m_vis
    else:
        source, as_of = "current", _iso(metar.get("obsTime")) or metar.get("reportTime")
        wind, vis, wx, clouds, vv = m_wind, m_vis, metar.get("wxString"), metar.get("clouds"), None
    if wind is None or vis is None:
        return None

    words = decode_wx(wx)
    sky = sky_phrase(clouds)
    if sky is None and group is None:  # METAR "clouds" is empty for CLR/SKC; its top-level "cover" says so
        sky = _COVER_WORDS.get(metar.get("cover"))
    conditions = ", ".join(words) if words else (sky or "unknown")
    if words and sky and sky != "clear":
        conditions += f"; {sky}"
    return {
        "temperature_f": round(temp_c * 9 / 5 + 32, 1),
        "wind_speed_mph": round(wind * KT_TO_MPH, 1),
        "visibility_miles": round(vis, 2),
        "conditions": conditions,
        "source": source,
        "as_of": as_of,
        "ceiling_ft": ceiling_ft(clouds, vv),
        "wx": (wx or "").strip() or None,
    }


# ---- risk-factor text -------------------------------------------------------------------------------------

def weather_risk_factors(w: dict, where: str) -> list[str]:
    """Conservative risk text from an airport's weather dict. `where` is "origin" or "destination"."""
    out = []
    tense = "in the forecast for that time" if w.get("source") == "forecast" else "reported now"
    tokens = [t.lstrip("+-") for t in (w.get("wx") or "").upper().split()]
    if any("TS" in t for t in tokens):
        out.append(f"Thunderstorms {tense} at {where}; delays may be more likely")
    if any(_WINTRY.search(t) for t in tokens):
        out.append(f"Snow or freezing precipitation {tense} at {where}; delays may be more likely")
    ceil = w.get("ceiling_ft")
    if ceil is not None and ceil < LOW_CEILING_FT:
        out.append(f"Low cloud ceiling (about {int(round(ceil, -1)):,} ft) at {where}; delays may be more likely")
    return out


# ---- fetching with per-airport caches ---------------------------------------------------------------------

class AviationWeatherClient:
    """METAR cached 10 min and TAF 30 min per airport; one comma-separated request per product."""

    def __init__(self, client: Optional[httpx.AsyncClient] = None):
        self.client = client or httpx.AsyncClient(timeout=TIMEOUT_SECONDS, headers={"User-Agent": USER_AGENT})
        self._cache: dict[tuple[str, str], tuple[float, Optional[dict]]] = {}
        self._down_until: dict[str, float] = {}

    async def _fetch(self, product: str, ids: list[str], ttl: int) -> dict[str, Optional[dict]]:
        now = time.monotonic()
        result: dict[str, Optional[dict]] = {}
        missing = []
        for i in ids:
            hit = self._cache.get((product, i))
            if hit and now - hit[0] < ttl:
                result[i] = hit[1]
            else:
                missing.append(i)
        if missing and self._down_until.get(product, 0) > now:
            for i in missing:
                result[i] = None
        elif missing:
            try:
                resp = None
                for attempt in range(2):  # the API occasionally stalls for seconds; one retry is usually instant
                    try:
                        resp = await self.client.get(f"{BASE_URL}/{product}",
                                                     params={"ids": ",".join(missing), "format": "json"})
                        break
                    except httpx.TimeoutException:
                        if attempt:
                            raise
                resp.raise_for_status()
                rows = resp.json() if resp.content else []
                by_id: dict[str, dict] = {}
                for r in rows if isinstance(rows, list) else []:
                    rid = (r or {}).get("icaoId")
                    if rid:
                        by_id.setdefault(rid, r)  # the API lists the most recent report first
                for i in missing:
                    self._cache[(product, i)] = (now, by_id.get(i))
                    result[i] = by_id.get(i)
            except Exception as e:  # network, HTTP, JSON: weather is optional
                logger.warning("aviationweather %s fetch failed: %r", product, e)
                self._down_until[product] = time.monotonic() + FAILURE_COOLDOWN_SECONDS
                for i in missing:
                    result[i] = None
        return result

    async def get_many(self, requests: list[tuple[str, Optional[datetime]]]) -> list[Optional[dict]]:
        """requests: [(icao, when)]; one weather dict (or None) per request, in order."""
        ids = list(dict.fromkeys(i for i, _ in requests))
        metars, tafs = await asyncio.gather(self._fetch("metar", ids, METAR_TTL_SECONDS),
                                            self._fetch("taf", ids, TAF_TTL_SECONDS))
        out = []
        for icao, when in requests:
            try:
                out.append(compose_weather(metars.get(icao), tafs.get(icao), when))
            except Exception as e:
                logger.warning("aviationweather compose failed for %s: %s", icao, e)
                out.append(None)
        return out
