"""
FlyRely Flight Delay Prediction API
====================================
FastAPI backend for predicting flight delay risk.

The delay and severity models are GradientBoosting classifiers trained on
BTS On-Time data (2024-2025) over 18 schedule/route features. They do not
use weather. Weather (free NOAA aviationweather.gov METAR/TAF by default, or a Tomorrow.io /
OpenWeatherMap key) only adds human-readable risk factors, and the response says whether it was available.

There is no fallback prediction: if a model is missing or inference fails,
/predict returns 503 with a machine-readable reason instead of a number.

Usage:
    pip install -r requirements.txt
    uvicorn main:app --reload

API Endpoints:
    GET  /                  - API info
    GET  /health            - Health check
    POST /predict           - Predict delay risk for a flight
    GET  /usage             - Prediction usage summary (admin only)
    GET  /usage/export      - Prediction usage log as CSV (admin only)
"""

import math
import os
import sys
import json
import csv
import time
import uuid
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Optional
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import joblib
import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_serializer

import codeshare
import prediction_log
import aviation_weather
from arrival_service import ArrivalService, arrival_block
from flyrely_dist.providers import LegSpec
from admin_auth import require_admin
from flyrely_features import stations  # read-only: ICAO ids for METAR/TAF lookups

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Weather API configuration (set via environment variables)
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")  # Tomorrow.io or OpenWeatherMap
# "aviationweather" (free NOAA METAR/TAF, no key; default when no key is set), "tomorrow" (default when a key is
# set) or "openweathermap". "none" turns weather off.
WEATHER_API_PROVIDER = os.getenv("WEATHER_API_PROVIDER", "")

# Frontend origins allowed to call the API (comma-separated).
# Localhost dev servers are always allowed; add deployed frontend origins via env.
DEV_ORIGINS = ["http://localhost:3000", "http://localhost:5173",
               "http://127.0.0.1:3000", "http://127.0.0.1:5173"]
ALLOWED_ORIGINS = DEV_ORIGINS + [
    o.strip() for o in os.getenv("FLYRELY_ALLOWED_ORIGINS", "").split(",") if o.strip()
]

# Model paths
MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "flight_delay_model.joblib"
FEATURES_PATH = MODEL_DIR / "feature_names.joblib"  # list of feature names, in model order
METADATA_PATH = MODEL_DIR / "model_metadata.json"
SEVERITY_MODEL_PATH = MODEL_DIR / "delay_severity_model.joblib"
SEVERITY_METADATA_PATH = MODEL_DIR / "delay_severity_metadata.json"
MODEL_VERSION = "bts18-gb-calib-v2"

# Usage log path (JSONL — one JSON record per line)
USAGE_LOG_PATH = Path(os.getenv("FLYRELY_USAGE_LOG", Path(__file__).parent / "usage_log.jsonl"))

# Weather API cost estimate for the keyed providers (Tomorrow.io free tier = 500 calls/day).
# The default aviationweather.gov provider is free: see WeatherService.cost_per_call.
WEATHER_COST_PER_CALL = 0.001  # USD

# Airport reference table (coords, names, time zones) built by scripts/build_airports_ref.py
AIRPORTS_REF_PATH = MODEL_DIR / "airports_ref.csv"


def _load_airports_ref(path: Path = AIRPORTS_REF_PATH) -> dict:
    with path.open(newline="") as f:
        return {r["iata"]: r for r in csv.DictReader(f)}


AIRPORTS_REF = _load_airports_ref()
# Airport coordinates for weather lookups and distance estimates
AIRPORT_COORDS = {c: (float(r["lat"]), float(r["lon"])) for c, r in AIRPORTS_REF.items()}

# Historical delay rates by airport (from training data)
AIRPORT_DELAY_RATES = {
    "ATL": 0.21, "ORD": 0.25, "DFW": 0.20, "DEN": 0.22, "LAX": 0.19, "JFK": 0.26,
    "SFO": 0.24, "SEA": 0.20, "MIA": 0.21, "PHX": 0.18, "LAS": 0.19, "MCO": 0.20,
    "CLT": 0.21, "EWR": 0.27, "BOS": 0.23, "MSP": 0.21, "DTW": 0.20, "PHL": 0.24,
    "LGA": 0.26, "DCA": 0.22, "IAH": 0.21, "SLC": 0.18, "SAN": 0.17, "TPA": 0.19,
    "PDX": 0.19, "BWI": 0.20, "FLL": 0.21, "MDW": 0.22, "BNA": 0.20, "AUS": 0.19,
}

# Historical delay rates by operating airline
AIRLINE_DELAY_RATES = {
    "AA": 0.21, "DL": 0.18, "UA": 0.22, "WN": 0.23, "AS": 0.17, "B6": 0.24,
    "NK": 0.28, "F9": 0.27, "G4": 0.26, "HA": 0.16, "SY": 0.25,
    # Regional operators flying for the majors (BTS 2025, departures delayed 15+ min)
    "OO": 0.185, "YX": 0.18, "MQ": 0.184, "OH": 0.256,
    # Six further regional operators, added once all-carrier BTS data (data/bts/parquet_v2, built from
    # the Marketing Carrier file) was available. Same definition and window as the regionals above:
    # calendar-2025 BTS flown flights by operating carrier, DepDelay >= 15 min (the window was verified
    # by reproducing OO/YX/MQ/OH exactly). 9E Endeavor, PT Piedmont, QX Horizon, G7 GoJet, C5 CommutAir, YV Mesa.
    "9E": 0.209, "PT": 0.174, "QX": 0.159, "G7": 0.250, "C5": 0.175, "YV": 0.171,
}

# Hub airports
HUB_AIRPORTS = {"ATL", "ORD", "DFW", "DEN", "LAX", "JFK", "SFO", "CLT", "LAS", "PHX",
                "MIA", "SEA", "EWR", "MCO", "BOS", "IAH", "MSP", "DTW", "PHL", "LGA"}

# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="FlyRely API",
    description="Flight delay prediction API (BTS-trained models)",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

import feedback  # noqa: E402
feedback.mount(app)
import flight_reports  # noqa: E402
flight_reports.mount(app)

# =============================================================================
# Usage Logging
# =============================================================================

class UsageLogger:
    """Logs each successful /predict call to a JSONL file."""

    def __init__(self, log_path: Path):
        self.log_path = log_path

    def record(self, entry: dict) -> None:
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"Usage log write failed: {e}")

    def read_all(self) -> list[dict]:
        if not self.log_path.exists():
            return []
        entries = []
        try:
            with open(self.log_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            logger.warning(f"Usage log read failed: {e}")
        return entries


usage_logger = UsageLogger(USAGE_LOG_PATH)

# Durable prediction log (SQLite, or Postgres when DATABASE_URL is set)
prediction_store = prediction_log.PredictionStore()

# =============================================================================
# Pydantic Models (canonical /predict contract)
# =============================================================================

class FlightRequest(BaseModel):
    """Request model for flight delay prediction."""
    flight_id: Optional[str] = Field(None, description="Client flight id, echoed back; generated if omitted")
    origin: str = Field(..., description="Origin airport code (e.g., 'JFK')", min_length=3, max_length=3)
    destination: str = Field(..., description="Destination airport code (e.g., 'LAX')", min_length=3, max_length=3)
    departure_time: datetime = Field(..., description="Scheduled local departure time (ISO, no offset)")
    airline: Optional[str] = Field(None, description="Ticketing airline code (e.g., 'AA', 'DL', 'UA')")
    flight_number: Optional[str] = Field(None, description="Ticketed flight number, e.g. '1234' or 'DL1234'; "
                                                           "used to find the operating carrier")
    arrival_time: Optional[datetime] = Field(None, description="Scheduled local arrival time (ISO, no offset). "
                                                               "Echoed back; the model does not use it")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "flight_id": "flight-1730000000000",
            "origin": "JFK",
            "destination": "LAX",
            "departure_time": "2026-11-15T14:30:00",
            "airline": "AA"
        }
    })


class WeatherData(BaseModel):
    """Weather conditions at an airport. The optional fields are only set by the aviationweather provider."""
    temperature_f: float
    wind_speed_mph: float
    visibility_miles: float
    conditions: str
    source: Optional[str] = Field(None, description="'current' (latest METAR) or 'forecast' (TAF block at departure)")
    as_of: Optional[str] = Field(None, description="UTC time of the observation, or the time the forecast is for")
    ceiling_ft: Optional[int] = Field(None, description="Lowest broken/overcast layer in feet, if any")
    wx: Optional[str] = Field(None, description="Raw weather phenomena, e.g. '-TSRA BR'")


class DelaySeverity(BaseModel):
    """Conditional delay severity breakdown — probabilities given a delay occurs."""
    minor_pct: float = Field(..., description="Probability of 15–44 min delay if delayed")
    moderate_pct: float = Field(..., description="Probability of 45–119 min delay if delayed")
    severe_pct: float = Field(..., description="Probability of 120+ min delay if delayed")
    expected_delay_label: str = Field(..., description="Most likely severity bucket if delayed")
    expected_delay_range: str = Field(..., description="Human-readable range, e.g. '15–44 min'")


class PredictionResponse(BaseModel):
    """Response model for flight delay prediction."""
    flight_id: str
    risk_level: str = Field(..., description="Risk level: 'low', 'medium', or 'high'")
    delay_probability: float = Field(..., description="Probability of a 15+ min delay (0-1)")
    confidence: float = Field(..., description="max(p, 1-p) — certainty of the binary call, not calibration")

    # Severity model output; null when the severity model is unavailable
    delay_severity: Optional[DelaySeverity] = None

    # Flight details
    origin: str
    destination: str
    departure_time: str
    airline: Optional[str]
    flight_number: Optional[str] = None
    # Who actually flies it (e.g. SkyWest for a Delta ticket); the prediction is made for this carrier
    operating_airline: Optional[str] = None
    operating_airline_name: Optional[str] = None
    operating_flight_number: Optional[str] = None

    # Weather is not a model input; it only feeds risk_factors
    weather_available: bool
    origin_weather: Optional[WeatherData] = None
    destination_weather: Optional[WeatherData] = None

    risk_factors: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)

    # Data-coverage caveats (e.g. airport not in the historical table)
    warnings: list[str] = Field(default_factory=list)

    model_version: str

    # Only present when the arrival flag is on (null then means the ladder is unavailable); arrival_time only when sent
    arrival_time: Optional[str] = None
    arrival: Optional[dict] = None

    @model_serializer(mode="wrap")
    def _omit_unset_arrival(self, handler):
        """Drop `arrival` / `arrival_time` unless they were set, so the output is byte-identical to before T6 when the
        flags are off. (Not `response_model_exclude_unset`: that also strips unset nested keys, e.g. WeatherData.source
        from the Tomorrow.io / OpenWeatherMap providers.)"""
        data = handler(self)
        if isinstance(data, dict):
            for k in ("arrival_time", "arrival"):
                if k not in self.model_fields_set:
                    data.pop(k, None)
        return data

# =============================================================================
# Weather Service
# =============================================================================

class WeatherService:
    """Service for fetching real-time weather data."""

    def __init__(self, api_key: str, provider: str = ""):
        self.api_key = api_key
        provider = (provider or "").strip().lower()
        if not provider:
            provider = "tomorrow" if api_key else "aviationweather"
        elif provider in ("tomorrow", "openweathermap") and not api_key:
            provider = "aviationweather"  # a keyed provider without a key: use the free one
        self.provider = provider
        self.client = httpx.AsyncClient(timeout=10.0)
        self.aviation = aviation_weather.AviationWeatherClient() if provider == "aviationweather" else None
        self._icao_table = None
        self._cache = {}  # Simple in-memory cache, keyed per airport per hour
        self._calls_day = None  # UTC date the call counter applies to
        self._calls_today = 0

    def _take_budget(self) -> bool:
        """Count one outbound provider call against the daily budget (FLYRELY_WEATHER_DAILY_BUDGET,
        default 450, under Tomorrow.io's 500/day free tier). False once spent: weather is then skipped."""
        today = datetime.now(timezone.utc).date()
        if self._calls_day != today:
            self._calls_day, self._calls_today = today, 0
        try:
            budget = int(os.getenv("FLYRELY_WEATHER_DAILY_BUDGET", "450"))
        except ValueError:
            budget = 450
        if self._calls_today >= budget:
            return False
        self._calls_today += 1
        return True

    @property
    def cost_per_call(self) -> float:
        """USD per outbound weather call: the NOAA provider is free."""
        return 0.0 if self.provider in ("aviationweather", "none") else WEATHER_COST_PER_CALL

    def icao(self, airport_code: str) -> str:
        if self._icao_table is None:
            self._icao_table = stations.load_table()
        return stations.icao_for(airport_code, self._icao_table if self._icao_table is not None else None)

    async def get_weather_pair(self, requests: list) -> list:
        """requests: [(airport_code, when_utc_or_None)] -> one weather dict (or None) each.

        aviationweather batches both airports into one METAR and one TAF request. Never raises."""
        if self.provider == "aviationweather":
            try:
                ids = [(self.icao(code), when) for code, when in requests]
                return await self.aviation.get_many(ids)
            except Exception as e:
                logger.error(f"Weather lookup error: {e}")
                return [None] * len(requests)
        if self.provider == "none":
            return [None] * len(requests)
        return [await self.get_weather(code) for code, _ in requests]

    async def get_weather(self, airport_code: str) -> Optional[dict]:
        """Get current weather for an airport from a keyed provider, or None if unavailable."""
        if self.provider in ("aviationweather", "none") or not self.api_key:
            return None

        coords = AIRPORT_COORDS.get(airport_code.upper())
        if not coords:
            return None

        cache_key = f"{airport_code}_{datetime.now().strftime('%Y%m%d%H')}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if not self._take_budget():
            logger.warning("Weather daily call budget spent; skipping weather for %s", airport_code)
            return None

        try:
            if self.provider == "tomorrow":
                weather = await self._fetch_tomorrow_io(coords)
            else:
                weather = await self._fetch_openweathermap(coords)

            if weather:
                self._cache[cache_key] = weather
            return weather

        except Exception as e:
            logger.error(f"Weather API error: {e}")
            return None

    async def _fetch_tomorrow_io(self, coords: tuple) -> Optional[dict]:
        """Fetch weather from Tomorrow.io API."""
        lat, lon = coords
        response = await self.client.get(
            "https://api.tomorrow.io/v4/weather/realtime",
            params={"location": f"{lat},{lon}", "apikey": self.api_key, "units": "imperial"},
        )
        response.raise_for_status()
        values = response.json().get("data", {}).get("values", {})
        if not {"temperature", "windSpeed", "visibility"} <= values.keys():
            return None
        return {
            "temperature_f": values["temperature"],
            "wind_speed_mph": values["windSpeed"],
            "visibility_miles": values["visibility"],
            "conditions": self._get_conditions_tomorrow(values),
        }

    async def _fetch_openweathermap(self, coords: tuple) -> Optional[dict]:
        """Fetch weather from OpenWeatherMap API."""
        lat, lon = coords
        response = await self.client.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"lat": lat, "lon": lon, "appid": self.api_key, "units": "imperial"},
        )
        response.raise_for_status()
        data = response.json()
        try:
            return {
                "temperature_f": data["main"]["temp"],
                "wind_speed_mph": data["wind"]["speed"],
                "visibility_miles": data["visibility"] / 1609.34,  # m to miles
                "conditions": data.get("weather", [{}])[0].get("description", "unknown"),
            }
        except (KeyError, TypeError):
            return None

    def _get_conditions_tomorrow(self, values: dict) -> str:
        """Convert Tomorrow.io weather code to description."""
        conditions = {
            1000: "clear", 1100: "mostly clear", 1101: "partly cloudy",
            1102: "mostly cloudy", 1001: "cloudy", 2000: "fog", 2100: "light fog",
            4000: "drizzle", 4001: "rain", 4200: "light rain", 4201: "heavy rain",
            5000: "snow", 5001: "flurries", 5100: "light snow", 5101: "heavy snow",
            6000: "freezing drizzle", 6001: "freezing rain", 7000: "ice pellets",
            7101: "heavy ice pellets", 7102: "light ice pellets", 8000: "thunderstorm"
        }
        return conditions.get(values.get("weatherCode"), "unknown")


weather_service = WeatherService(WEATHER_API_KEY, WEATHER_API_PROVIDER)

# =============================================================================
# Model Service
# =============================================================================

class PredictionUnavailable(Exception):
    """Raised when no real prediction can be produced."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason
        self.message = message


class ModelService:
    """Loads the delay + severity models and runs inference. No fallbacks."""

    def __init__(self):
        self.model = None
        self.feature_names: Optional[list[str]] = None
        self.severity_model = None
        self.load_error: Optional[str] = None
        self._load_model()

    def _load_model(self):
        try:
            self.feature_names = list(joblib.load(FEATURES_PATH))
            model = joblib.load(MODEL_PATH)
            if model.n_features_in_ != len(self.feature_names):
                raise ValueError(
                    f"model expects {model.n_features_in_} features, "
                    f"feature_names.joblib lists {len(self.feature_names)}"
                )
            self.model = model
            logger.info(f"Delay model loaded ({len(self.feature_names)} features)")
        except Exception as e:
            self.load_error = f"{type(e).__name__}: {e}"
            logger.error(f"Delay model unavailable: {self.load_error}")
            return

        try:
            severity = joblib.load(SEVERITY_MODEL_PATH)
            if severity.n_features_in_ != len(self.feature_names):
                raise ValueError("severity model feature count does not match delay model")
            self.severity_model = severity
            logger.info("Severity model loaded")
        except Exception as e:
            logger.error(f"Severity model unavailable: {type(e).__name__}: {e}")

    def _vector(self, features: dict) -> pd.DataFrame:
        missing = [n for n in self.feature_names if n not in features]
        if missing:
            raise PredictionUnavailable("prediction_failed", f"missing features: {missing}")
        # Named columns let sklearn verify them against the names seen in training
        return pd.DataFrame([[features[n] for n in self.feature_names]],
                            columns=self.feature_names, dtype=float)

    def predict(self, features: dict) -> tuple[float, str]:
        """Returns (probability of 15+ min delay, risk_level)."""
        if self.model is None:
            raise PredictionUnavailable("model_unavailable", self.load_error or "model not loaded")
        try:
            proba = float(self.model.predict_proba(self._vector(features))[0, 1])
        except PredictionUnavailable:
            raise
        except Exception as e:
            raise PredictionUnavailable("prediction_failed", f"{type(e).__name__}: {e}")
        return proba, self._probability_to_risk(proba)

    def predict_many(self, feature_rows: list) -> list:
        """Batch of predict(): one model call for all rows. Returns [(probability, risk_level), ...]."""
        if self.model is None:
            raise PredictionUnavailable("model_unavailable", self.load_error or "model not loaded")
        if not feature_rows:
            return []
        try:
            missing = [n for n in self.feature_names if any(n not in f for f in feature_rows)]
            if missing:
                raise PredictionUnavailable("prediction_failed", f"missing features: {missing}")
            frame = pd.DataFrame([[f[n] for n in self.feature_names] for f in feature_rows],
                                 columns=self.feature_names, dtype=float)
            probas = self.model.predict_proba(frame)[:, 1]
        except PredictionUnavailable:
            raise
        except Exception as e:
            raise PredictionUnavailable("prediction_failed", f"{type(e).__name__}: {e}")
        return [(float(p), self._probability_to_risk(float(p))) for p in probas]

    def _probability_to_risk(self, prob: float) -> str:
        """Calibration v2 thresholds: Low < 0.20, Medium 0.20–0.30, High >= 0.30"""
        if prob < 0.20:
            return "low"
        elif prob < 0.30:
            return "medium"
        else:
            return "high"

    def predict_severity(self, features: dict) -> Optional[DelaySeverity]:
        """
        Conditional severity distribution (given a delay occurs) from the
        4-class model [on_time, minor, moderate, severe]. None if unavailable.
        """
        if self.severity_model is None:
            return None
        try:
            proba = self.severity_model.predict_proba(self._vector(features).to_numpy())[0]  # fitted without names
        except Exception as e:
            logger.error(f"Severity prediction failed: {e}")
            return None

        delay_proba = proba[1:]  # [minor, moderate, severe]
        delay_total = delay_proba.sum()
        if delay_total < 1e-9:
            return None
        conditional = delay_proba / delay_total

        bucket_idx = int(np.argmax(delay_proba))
        return DelaySeverity(
            minor_pct=round(float(conditional[0]), 3),
            moderate_pct=round(float(conditional[1]), 3),
            severe_pct=round(float(conditional[2]), 3),
            expected_delay_label=["minor", "moderate", "severe"][bucket_idx],
            expected_delay_range=["15–44 min", "45–119 min", "120+ min"][bucket_idx],
        )


model_service = ModelService()

# Arrival ladder (plan: backtest/PLAN_ARRIVAL_AND_CONNECTIONS.md 2.3). Both flags are OFF by default, and the ladder is
# not even loaded unless one is on, so nothing about /predict changes until Sean turns them on after evaluation (T5).
ARRIVAL_FLAG = os.getenv("FLYRELY_FEATURE_ARRIVAL") == "1"
ITINERARY_FLAG = os.getenv("FLYRELY_FEATURE_ITINERARY") == "1"
arrival_service = ArrivalService(feature_fn=lambda *a: build_features(*a), enabled=ARRIVAL_FLAG or ITINERARY_FLAG)

# =============================================================================
# Helper Functions
# =============================================================================

def get_season(month: int) -> int:
    """Get season from month (0=winter, 1=spring, 2=summer, 3=fall)."""
    if month in [12, 1, 2]:
        return 0
    elif month in [3, 4, 5]:
        return 1
    elif month in [6, 7, 8]:
        return 2
    else:
        return 3


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n-th given weekday (Mon=0) of a month; n=-1 means the last one."""
    if n > 0:
        first = date(year, month, 1)
        return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))
    last = date(year, month + 1, 1) - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def us_major_holidays(year: int) -> list[date]:
    """Holidays used in training: New Year, MLK, Presidents, Memorial, July 4,
    Labor, Thanksgiving, Christmas."""
    return [
        date(year, 1, 1),
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _nth_weekday(year, 5, 0, -1),
        date(year, 7, 4),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 11, 3, 4),
        date(year, 12, 25),
    ]


def get_days_to_holiday(dt: datetime) -> int:
    """Calculate days to nearest major US holiday."""
    d = dt.date()
    holidays = [h for y in (d.year - 1, d.year, d.year + 1) for h in us_major_holidays(y)]
    return min(abs((d - h).days) for h in holidays)


def estimate_distance(origin: str, dest: str) -> float:
    """Great-circle (haversine) distance in statute miles, minimum 100.

    The longitude difference is normalised to [-180, 180] so routes that cross the
    antimeridian (e.g. GUM-HNL) are measured the short way round.
    """
    lat1, lon1 = AIRPORT_COORDS.get(origin.upper(), (39.8, -98.5))  # Default: center US
    lat2, lon2 = AIRPORT_COORDS.get(dest.upper(), (39.8, -98.5))
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians((lon2 - lon1 + 180) % 360 - 180)
    h = math.sin((phi2 - phi1) / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlon / 2) ** 2
    distance = 3958.8 * 2 * math.asin(min(1.0, math.sqrt(h)))
    return max(distance, 100)  # Minimum 100 miles


def build_features(origin: str, destination: str, dep_time: datetime, airline: Optional[str]) -> dict:
    """The 18 schedule/route features the BTS models were trained on."""
    origin_known = origin in AIRPORT_COORDS
    dest_known = destination in AIRPORT_COORDS
    # Unknown airports fall back to the airline's rate, then the global rate
    fallback_rate = AIRLINE_DELAY_RATES.get(airline, 0.20)
    origin_rate = AIRPORT_DELAY_RATES.get(origin, fallback_rate)
    dest_rate = AIRPORT_DELAY_RATES.get(destination, fallback_rate)
    days_to_holiday = get_days_to_holiday(dep_time)

    return {
        'dep_hour': dep_time.hour,
        'day_of_week': dep_time.isoweekday(),
        'month': dep_time.month,
        'is_weekend': 1 if dep_time.isoweekday() >= 6 else 0,
        'is_morning': 1 if 5 <= dep_time.hour < 12 else 0,
        'is_afternoon': 1 if 12 <= dep_time.hour < 18 else 0,
        'is_evening': 1 if 18 <= dep_time.hour < 22 else 0,
        'is_night': 1 if dep_time.hour >= 22 or dep_time.hour < 5 else 0,
        'season': get_season(dep_time.month),
        'days_to_holiday': days_to_holiday,
        'is_holiday_period': 1 if days_to_holiday <= 3 else 0,
        'origin_delay_rate': origin_rate,
        'dest_delay_rate': dest_rate,
        'route_delay_rate': (origin_rate + dest_rate) / 2,
        'airline_delay_rate': AIRLINE_DELAY_RATES.get(airline, 0.21) if airline else 0.21,
        'origin_is_hub': 1 if origin in HUB_AIRPORTS else 0,
        'dest_is_hub': 1 if destination in HUB_AIRPORTS else 0,
        'distance': estimate_distance(origin, destination) if origin_known and dest_known else 250,
    }


def time_of_day_reason(hour: int) -> Optional[str]:
    """Plain-language reason from departure hour. The model's delay probability rises through the day
    (lowest early morning, highest around 17-20h), so these only say that."""
    if 5 <= hour < 9:
        return "Early-morning flights are rarely delayed"
    if 17 <= hour <= 21:
        return "Evening flights are more often delayed, as delays build up during the day"
    if 13 <= hour < 17:
        return "Afternoon flights are somewhat more often delayed, as delays build up during the day"
    return None


def get_risk_factors(features: dict, weather_origin: Optional[dict], weather_dest: Optional[dict],
                     origin: Optional[str] = None, destination: Optional[str] = None) -> list[str]:
    """Reasons behind the prediction, most important first (weather, holiday, time of day, busy airport,
    airport history). Only the model's schedule features and the weather feed these."""
    factors = []

    if weather_origin:
        factors.extend(aviation_weather.weather_risk_factors(weather_origin, "origin"))
        if weather_origin['visibility_miles'] < 3:
            factors.append(f"Low visibility at origin ({weather_origin['visibility_miles']:.1f} mi)")
        if weather_origin['wind_speed_mph'] > 20:
            factors.append(f"High winds at origin ({weather_origin['wind_speed_mph']:.0f} mph)")
        if weather_origin['temperature_f'] < 32:
            factors.append(f"Freezing conditions at origin ({weather_origin['temperature_f']:.0f}°F)")

    if weather_dest:
        factors.extend(aviation_weather.weather_risk_factors(weather_dest, "destination"))
        if weather_dest['visibility_miles'] < 3:
            factors.append(f"Low visibility at destination ({weather_dest['visibility_miles']:.1f} mi)")
        if weather_dest['wind_speed_mph'] > 20:
            factors.append(f"High winds at destination ({weather_dest['wind_speed_mph']:.0f} mph)")

    if features.get('is_holiday_period'):
        factors.append("Holiday travel period (high traffic)")

    tod = time_of_day_reason(features.get('dep_hour', 12))
    if tod:
        factors.append(tod)

    busy = []
    if features.get('origin_is_hub'):
        busy.append(origin or "the departure airport")
    if features.get('dest_is_hub'):
        busy.append(destination or "the arrival airport")
    if busy:
        factors.append(f"{' and '.join(busy)} {'is a' if len(busy) == 1 else 'are'} busy "
                       f"{'airport' if len(busy) == 1 else 'airports'}, where delays spread more easily")

    origin_rate = features.get('origin_delay_rate', 0.20)
    if origin_rate > 0.24:
        factors.append("Origin airport has a high historical delay rate")

    return factors


def get_recommendations(risk_level: str, factors: list[str]) -> list[str]:
    """Generate recommendations based on risk level."""
    recommendations = []

    if risk_level == "high":
        recommendations.append("Consider booking an earlier flight as backup")
        recommendations.append("Allow extra time for connections (2+ hours)")
        recommendations.append("Sign up for flight status notifications")
    elif risk_level == "medium":
        recommendations.append("Monitor flight status closer to departure")
        recommendations.append("Have a backup plan for tight connections")
    else:
        recommendations.append("Flight conditions look favorable")

    if any("visibility" in f.lower() for f in factors):
        recommendations.append("Check for fog-related delays in morning hours")

    if any("wind" in f.lower() for f in factors):
        recommendations.append("Wind may cause turbulence - secure belongings")

    return recommendations

# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/")
async def root():
    """API information."""
    return {
        "name": "FlyRely API",
        "version": app.version,
        "model_version": MODEL_VERSION,
        "endpoints": {
            "POST /predict": "Predict delay risk for a flight",
            "POST /predict/itinerary": "Connection risk for a multi-leg trip (only when the itinerary flag is on)",
            "GET /health": "Health check",
            "GET /usage": "Prediction usage summary",
            "GET /usage/export": "Prediction usage log (CSV)",
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint. status is 'degraded' when /predict cannot serve."""
    model_loaded = model_service.model is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "severity_model_loaded": model_service.severity_model is not None,
        "arrival_model_loaded": arrival_service.loaded,
        "weather_api_configured": bool(WEATHER_API_KEY),
        "weather_provider": weather_service.provider,
        "model_version": MODEL_VERSION,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/usage")
async def get_usage(days: int = Query(default=30, ge=1, le=365), _admin: str = Depends(require_admin)):
    """Daily prediction counts, weather calls and estimated weather cost for the last `days` days. Admin only."""
    cutoff = datetime.now() - timedelta(days=days)
    entries = [e for e in usage_logger.read_all() if datetime.fromisoformat(e["ts"]) >= cutoff]

    daily: dict[str, dict] = {}
    route_counts: dict[str, int] = {}
    risk_counts: dict[str, int] = {"low": 0, "medium": 0, "high": 0}
    total_weather = 0

    for e in entries:
        day = e.get("date", e["ts"][:10])
        if day not in daily:
            daily[day] = {"date": day, "predictions": 0, "weather_calls": 0}
        daily[day]["predictions"] += 1
        wc = e.get("weather_calls", 0)
        daily[day]["weather_calls"] += wc
        total_weather += wc

        route = f"{e.get('origin', '?')}→{e.get('destination', '?')}"
        route_counts[route] = route_counts.get(route, 0) + 1

        rl = e.get("risk_level")
        if rl in risk_counts:
            risk_counts[rl] += 1

    top_routes = sorted(
        [{"route": r, "predictions": c} for r, c in route_counts.items()],
        key=lambda x: x["predictions"],
        reverse=True,
    )[:10]

    return {
        "period_days": days,
        "total_predictions": len(entries),
        "total_weather_calls": total_weather,
        "estimated_cost_usd": round(total_weather * weather_service.cost_per_call, 4),
        "avg_predictions_per_day": round(len(entries) / max(days, 1), 1),
        "daily": sorted(daily.values(), key=lambda x: x["date"], reverse=True),
        "top_routes": top_routes,
        "by_risk_level": risk_counts,
        "note": (
            "Weather provider: aviationweather.gov (NOAA), free: $0 per call."
            if weather_service.cost_per_call == 0 else
            f"Cost estimate based on ${weather_service.cost_per_call}/weather API call. "
            "Tomorrow.io free tier: 500 calls/day."),
    }


@app.get("/usage/export")
async def export_usage(_admin: str = Depends(require_admin)):
    """Download all usage logs as a CSV file. Admin only."""
    entries = usage_logger.read_all()
    if not entries:
        raise HTTPException(status_code=404, detail="No usage data to export")

    output = StringIO()
    fieldnames = ["ts", "date", "flight_id", "origin", "destination", "airline", "departure_time",
                  "risk_level", "delay_probability", "weather_calls", "model_version", "model_ms"]
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(entries)

    filename = f"flyrely_usage_{datetime.now().strftime('%Y%m%d')}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


def _weather_times(origin: str, destination: str, dep_local: datetime):
    """(departure, arrival) as aware UTC datetimes for the weather lookup, or (None, None) if the origin's time
    zone is unknown. The request has no scheduled block time, so arrival is departure plus a rough estimate
    (30 min taxi/approach + great-circle distance at ~450 mph); the response's `as_of` says the exact time used."""
    tz = (AIRPORTS_REF.get(origin) or {}).get("tz")
    if not tz:
        return None, None
    try:
        from zoneinfo import ZoneInfo
        dep_utc = dep_local.replace(tzinfo=ZoneInfo(tz)).astimezone(timezone.utc)
        block_min = 30 + estimate_distance(origin, destination) / 450 * 60
        return dep_utc, dep_utc + timedelta(minutes=block_min)
    except Exception:
        return None, None


@app.post("/predict", response_model=PredictionResponse)
async def predict_delay(request: FlightRequest, background_tasks: BackgroundTasks):
    """
    Predict flight delay risk.

    Returns 503 {"detail": {"reason": "model_unavailable" | "prediction_failed", "message": ...}}
    when no real prediction can be made.

    The optional `arrival` / `arrival_time` keys appear only when set (arrival flag on, or the request sent
    arrival_time); see PredictionResponse._omit_unset_arrival.
    """
    response, _ = await _predict_core(request, background_tasks)
    return response


async def _predict_core(request: FlightRequest, background_tasks: BackgroundTasks, need_forecast: bool = False):
    """One leg's prediction. Returns (PredictionResponse, LegForecast or None). `need_forecast` (itineraries) makes a
    missing ladder or a ladder failure a 503 instead of `arrival: null`."""
    flight_id = request.flight_id or f"srv-{uuid.uuid4().hex[:12]}"
    origin = request.origin.upper()
    destination = request.destination.upper()
    # Scheduled times are local wall-clock (as in BTS); drop any offset
    dep_time = request.departure_time.replace(tzinfo=None)
    airline, flight_number = codeshare.parse_flight(
        request.airline.upper() if request.airline else None, request.flight_number)

    # Predict for the carrier that flies the plane; the customer still sees the airline they booked
    operator = codeshare.resolve(airline, flight_number, origin, destination)
    model_airline = airline
    if operator and operator["operating_airline"] in AIRLINE_DELAY_RATES:
        model_airline = operator["operating_airline"]

    for label, code in (("origin", origin), ("destination", destination)):
        if code not in AIRPORT_COORDS:
            raise HTTPException(status_code=422, detail={
                "reason": "unknown_airport",
                "message": f"We don't recognise the {label} airport {code}. FlyRely covers US airports only."})
    if origin == destination:
        raise HTTPException(status_code=422, detail={
            "reason": "same_airport", "message": "Origin and destination must be different airports."})

    warnings = []
    if model_airline and model_airline not in AIRLINE_DELAY_RATES:
        warnings.append(f"Airline {model_airline} is not in FlyRely's historical data; using defaults")
    if operator and model_airline != operator["operating_airline"]:
        warnings.append(f"No history for operator {operator['operating_airline_name']}; "
                        f"using {airline}'s delay rate")

    features = build_features(origin, destination, dep_time, model_airline)

    start = time.perf_counter()
    try:
        probability, risk_level = model_service.predict(features)
    except PredictionUnavailable as e:
        logger.error(f"[predict] {flight_id}: {e.reason}: {e.message}")
        raise HTTPException(status_code=503, detail={"reason": e.reason, "message": e.message})
    delay_severity = model_service.predict_severity(features)
    model_ms = round((time.perf_counter() - start) * 1000)

    forecast = None
    arrival_set = False
    arrival = None
    if need_forecast or ARRIVAL_FLAG:
        try:
            if not arrival_service.loaded:
                raise PredictionUnavailable("model_unavailable",
                                            arrival_service.load_error or "arrival model not loaded")
            try:
                forecast = arrival_service.forecast([LegSpec(origin, destination, dep_time, model_airline)])[0]
            except PredictionUnavailable:
                raise
            except Exception as e:
                raise PredictionUnavailable("prediction_failed", f"{type(e).__name__}: {e}")
        except PredictionUnavailable as e:
            logger.error(f"[predict] {flight_id}: arrival {e.reason}: {e.message}")
            if need_forecast:
                raise HTTPException(status_code=503, detail={"reason": e.reason, "message": e.message})
            warnings.append("Arrival forecast unavailable")
        arrival_set = True
        if forecast is not None:
            arrival = arrival_block(forecast, arrival_service.version, model_service._probability_to_risk)

    dep_utc, arr_utc = _weather_times(origin, destination, dep_time)
    weather_origin, weather_dest = await weather_service.get_weather_pair(
        [(origin, dep_utc), (destination, arr_utc)])

    risk_factors = get_risk_factors(features, weather_origin, weather_dest, origin, destination)
    recommendations = get_recommendations(risk_level, risk_factors)

    usage_logger.record({
        "ts": datetime.now().isoformat(),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "flight_id": flight_id,
        "origin": origin,
        "destination": destination,
        "airline": airline,
        "departure_time": dep_time.isoformat(),
        "risk_level": risk_level,
        "delay_probability": round(probability, 3),
        "weather_calls": (weather_origin is not None) + (weather_dest is not None),
        "model_version": MODEL_VERSION,
        "model_ms": model_ms,
    })

    try:
        background_tasks.add_task(prediction_store.safe_write, prediction_log.build_record(
            flight_id=flight_id, origin=origin, dest=destination, sched_dep_local=dep_time,
            marketing_airline=airline, marketing_number=flight_number,
            operating_airline=operator["operating_airline"] if operator else None,
            operating_number=operator["operating_flight_number"] if operator else None,
            delay_probability=round(probability, 3), risk_level=risk_level,
            model_version=MODEL_VERSION, features=features,
            weather_available=weather_origin is not None and weather_dest is not None,
            arr_probability=round(forecast.arr.prob_late(15), 3) if forecast else None,
            cancel_probability=round(forecast.p_cancel, 4) if forecast else None,
            arr_model_version=arrival_service.version if forecast else None))
    except Exception as e:
        logger.warning(f"Prediction log record build failed: {e}")

    extra = {}
    if request.arrival_time is not None:
        extra["arrival_time"] = request.arrival_time.replace(tzinfo=None).isoformat()
    if arrival_set:
        extra["arrival"] = arrival
    response = PredictionResponse(
        flight_id=flight_id,
        risk_level=risk_level,
        delay_probability=round(probability, 3),
        confidence=round(max(probability, 1 - probability), 3),
        delay_severity=delay_severity,
        origin=origin,
        destination=destination,
        departure_time=dep_time.isoformat(),
        airline=airline,
        flight_number=flight_number,
        operating_airline=operator["operating_airline"] if operator else None,
        operating_airline_name=operator["operating_airline_name"] if operator else None,
        operating_flight_number=operator["operating_flight_number"] if operator else None,
        weather_available=weather_origin is not None and weather_dest is not None,
        origin_weather=WeatherData(**weather_origin) if weather_origin else None,
        destination_weather=WeatherData(**weather_dest) if weather_dest else None,
        risk_factors=risk_factors,
        recommendations=recommendations,
        warnings=warnings,
        model_version=MODEL_VERSION,
        **extra,
    )
    return response, forecast


# =============================================================================
# Disruption copilot (docs/PLAN_DISRUPTION_COPILOT.md): mounted only behind the flag
# =============================================================================

def _copilot_scorer():
    """Booking-time scorer for the copilot. Does not write to the prediction or usage logs."""
    def score(origin: str, destination: str, dep_time: datetime, airline: Optional[str]):
        return model_service.predict(build_features(origin, destination, dep_time, airline))
    return score


def _score_many(items: list) -> list:
    """items: [(origin, dest, dep_time, airline)] -> [(probability, risk_level)] in one model call.
    Like /predict, scores for the operating carrier when FlyRely has history for it. Raises PredictionUnavailable."""
    return model_service.predict_many([build_features(o, d, t, a) for o, d, t, a in items])


import flight_lookup  # noqa: E402   (imports copilot lazily, so the copilot flag-off sentinel holds)
def _search_factors(origin: str, destination: str, dep_time: datetime, airline: Optional[str]) -> list[str]:
    """Schedule-only reasons for a route-search row (no weather lookup per row)."""
    return get_risk_factors(build_features(origin, destination, dep_time, airline), None, None, origin, destination)


flight_lookup.mount(app, score_many=_score_many, scored_airlines=AIRLINE_DELAY_RATES, unavailable=PredictionUnavailable,
                    known_airports=set(AIRPORT_COORDS), factors_for=_search_factors)

if ITINERARY_FLAG:
    import itinerary
    itinerary.mount(app, sys.modules[__name__])

if os.getenv("FLYRELY_FEATURE_DISRUPTION_COPILOT") == "1":
    from copilot import mount as _copilot_mount
    _copilot_mount(app, score=_copilot_scorer())


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
