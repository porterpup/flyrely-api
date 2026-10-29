"""
FlyRely Flight Delay Prediction API
====================================
FastAPI backend for predicting flight delay risk.

The delay and severity models are GradientBoosting classifiers trained on
BTS On-Time data (2024-2025) over 18 schedule/route features. They do not
use weather. Weather (when a key is configured) only adds human-readable
risk factors, and the response says whether it was available.

There is no fallback prediction: if a model is missing or inference fails,
/predict returns 503 with a machine-readable reason instead of a number.

Usage:
    pip install -r requirements.txt
    uvicorn main:app --reload

API Endpoints:
    GET  /                  - API info
    GET  /health            - Health check
    POST /predict           - Predict delay risk for a flight
    GET  /usage             - Prediction usage summary
    GET  /usage/export      - Prediction usage log as CSV
"""

import os
import json
import csv
import time
import uuid
import logging
from datetime import date, datetime, timedelta
from typing import Optional
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import joblib
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Weather API configuration (set via environment variables)
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")  # Tomorrow.io or OpenWeatherMap
WEATHER_API_PROVIDER = os.getenv("WEATHER_API_PROVIDER", "tomorrow")  # "tomorrow" or "openweathermap"

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

# Weather API cost estimate: Tomorrow.io free tier = 500 calls/day
WEATHER_COST_PER_CALL = 0.001  # USD

# Airport coordinates for weather lookups and distance estimates
AIRPORT_COORDS = {
    "ATL": (33.6407, -84.4277), "ORD": (41.9742, -87.9073), "DFW": (32.8998, -97.0403),
    "DEN": (39.8561, -104.6737), "LAX": (33.9425, -118.4081), "JFK": (40.6413, -73.7781),
    "SFO": (37.6213, -122.3790), "SEA": (47.4502, -122.3088), "MIA": (25.7959, -80.2870),
    "PHX": (33.4373, -112.0078), "LAS": (36.0840, -115.1537), "MCO": (28.4312, -81.3081),
    "CLT": (35.2140, -80.9431), "EWR": (40.6895, -74.1745), "BOS": (42.3656, -71.0096),
    "MSP": (44.8848, -93.2223), "DTW": (42.2162, -83.3554), "PHL": (39.8744, -75.2424),
    "LGA": (40.7769, -73.8740), "DCA": (38.8512, -77.0402), "IAH": (29.9902, -95.3368),
    "SLC": (40.7899, -111.9791), "SAN": (32.7338, -117.1933), "TPA": (27.9756, -82.5333),
    "PDX": (45.5898, -122.5951), "BWI": (39.1774, -76.6684), "FLL": (26.0742, -80.1506),
    "MDW": (41.7868, -87.7522), "BNA": (36.1263, -86.6774), "AUS": (30.1975, -97.6664),
}

# Historical delay rates by airport (from training data)
AIRPORT_DELAY_RATES = {
    "ATL": 0.21, "ORD": 0.25, "DFW": 0.20, "DEN": 0.22, "LAX": 0.19, "JFK": 0.26,
    "SFO": 0.24, "SEA": 0.20, "MIA": 0.21, "PHX": 0.18, "LAS": 0.19, "MCO": 0.20,
    "CLT": 0.21, "EWR": 0.27, "BOS": 0.23, "MSP": 0.21, "DTW": 0.20, "PHL": 0.24,
    "LGA": 0.26, "DCA": 0.22, "IAH": 0.21, "SLC": 0.18, "SAN": 0.17, "TPA": 0.19,
    "PDX": 0.19, "BWI": 0.20, "FLL": 0.21, "MDW": 0.22, "BNA": 0.20, "AUS": 0.19,
}

# Historical delay rates by airline
AIRLINE_DELAY_RATES = {
    "AA": 0.21, "DL": 0.18, "UA": 0.22, "WN": 0.23, "AS": 0.17, "B6": 0.24,
    "NK": 0.28, "F9": 0.27, "G4": 0.26, "HA": 0.16, "SY": 0.25,
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

# =============================================================================
# Pydantic Models (canonical /predict contract)
# =============================================================================

class FlightRequest(BaseModel):
    """Request model for flight delay prediction."""
    flight_id: Optional[str] = Field(None, description="Client flight id, echoed back; generated if omitted")
    origin: str = Field(..., description="Origin airport code (e.g., 'JFK')", min_length=3, max_length=3)
    destination: str = Field(..., description="Destination airport code (e.g., 'LAX')", min_length=3, max_length=3)
    departure_time: datetime = Field(..., description="Scheduled local departure time (ISO, no offset)")
    airline: Optional[str] = Field(None, description="Airline code (e.g., 'AA', 'DL', 'UA')")

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
    """Weather conditions at an airport."""
    temperature_f: float
    wind_speed_mph: float
    visibility_miles: float
    conditions: str


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

    # Weather is not a model input; it only feeds risk_factors
    weather_available: bool
    origin_weather: Optional[WeatherData] = None
    destination_weather: Optional[WeatherData] = None

    risk_factors: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)

    # Data-coverage caveats (e.g. airport not in the historical table)
    warnings: list[str] = Field(default_factory=list)

    model_version: str

# =============================================================================
# Weather Service
# =============================================================================

class WeatherService:
    """Service for fetching real-time weather data."""

    def __init__(self, api_key: str, provider: str = "tomorrow"):
        self.api_key = api_key
        self.provider = provider
        self.client = httpx.AsyncClient(timeout=10.0)
        self._cache = {}  # Simple in-memory cache, keyed per airport per hour

    async def get_weather(self, airport_code: str) -> Optional[dict]:
        """Get current weather for an airport, or None if unavailable."""
        if not self.api_key:
            return None

        coords = AIRPORT_COORDS.get(airport_code.upper())
        if not coords:
            return None

        cache_key = f"{airport_code}_{datetime.now().strftime('%Y%m%d%H')}"
        if cache_key in self._cache:
            return self._cache[cache_key]

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
    """Estimate flight distance in miles."""
    lat1, lon1 = AIRPORT_COORDS.get(origin.upper(), (39.8, -98.5))  # Default: center US
    lat2, lon2 = AIRPORT_COORDS.get(dest.upper(), (39.8, -98.5))
    # Rough approximation: 1 degree ≈ 69 miles
    distance = ((abs(lat2 - lat1) * 69) ** 2 + (abs(lon2 - lon1) * 69 * 0.7) ** 2) ** 0.5
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


def get_risk_factors(features: dict, weather_origin: Optional[dict], weather_dest: Optional[dict]) -> list[str]:
    """Identify risk factors for the flight."""
    factors = []

    hour = features.get('dep_hour', 12)
    if 17 <= hour <= 21:
        factors.append("Evening departure (peak congestion time)")

    if features.get('is_holiday_period'):
        factors.append("Holiday travel period (high traffic)")

    if weather_origin:
        if weather_origin['visibility_miles'] < 3:
            factors.append(f"Low visibility at origin ({weather_origin['visibility_miles']:.1f} mi)")
        if weather_origin['wind_speed_mph'] > 20:
            factors.append(f"High winds at origin ({weather_origin['wind_speed_mph']:.0f} mph)")
        if weather_origin['temperature_f'] < 32:
            factors.append(f"Freezing conditions at origin ({weather_origin['temperature_f']:.0f}°F)")

    if weather_dest:
        if weather_dest['visibility_miles'] < 3:
            factors.append(f"Low visibility at destination ({weather_dest['visibility_miles']:.1f} mi)")
        if weather_dest['wind_speed_mph'] > 20:
            factors.append(f"High winds at destination ({weather_dest['wind_speed_mph']:.0f} mph)")

    origin_rate = features.get('origin_delay_rate', 0.20)
    if origin_rate > 0.24:
        factors.append(f"Origin airport has high historical delay rate ({origin_rate*100:.0f}%)")

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
        "weather_api_configured": bool(WEATHER_API_KEY),
        "model_version": MODEL_VERSION,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/usage")
async def get_usage(days: int = Query(default=30, ge=1, le=365)):
    """Daily prediction counts, weather calls and estimated weather cost for the last `days` days."""
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
        "estimated_cost_usd": round(total_weather * WEATHER_COST_PER_CALL, 4),
        "avg_predictions_per_day": round(len(entries) / max(days, 1), 1),
        "daily": sorted(daily.values(), key=lambda x: x["date"], reverse=True),
        "top_routes": top_routes,
        "by_risk_level": risk_counts,
        "note": f"Cost estimate based on ${WEATHER_COST_PER_CALL}/weather API call. Tomorrow.io free tier: 500 calls/day.",
    }


@app.get("/usage/export")
async def export_usage():
    """Download all usage logs as a CSV file."""
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


@app.post("/predict", response_model=PredictionResponse)
async def predict_delay(request: FlightRequest):
    """
    Predict flight delay risk.

    Returns 503 {"detail": {"reason": "model_unavailable" | "prediction_failed", "message": ...}}
    when no real prediction can be made.
    """
    flight_id = request.flight_id or f"srv-{uuid.uuid4().hex[:12]}"
    origin = request.origin.upper()
    destination = request.destination.upper()
    # Scheduled times are local wall-clock (as in BTS); drop any offset
    dep_time = request.departure_time.replace(tzinfo=None)
    airline = request.airline.upper() if request.airline else None

    warnings = []
    for label, code in (("Origin", origin), ("Destination", destination)):
        if code not in AIRPORT_COORDS:
            warnings.append(f"{label} airport {code} is not in FlyRely's historical data; using defaults")
    if airline and airline not in AIRLINE_DELAY_RATES:
        warnings.append(f"Airline {airline} is not in FlyRely's historical data; using defaults")

    features = build_features(origin, destination, dep_time, airline)

    start = time.perf_counter()
    try:
        probability, risk_level = model_service.predict(features)
    except PredictionUnavailable as e:
        logger.error(f"[predict] {flight_id}: {e.reason}: {e.message}")
        raise HTTPException(status_code=503, detail={"reason": e.reason, "message": e.message})
    delay_severity = model_service.predict_severity(features)
    model_ms = round((time.perf_counter() - start) * 1000)

    weather_origin = await weather_service.get_weather(origin)
    weather_dest = await weather_service.get_weather(destination)

    risk_factors = get_risk_factors(features, weather_origin, weather_dest)
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

    return PredictionResponse(
        flight_id=flight_id,
        risk_level=risk_level,
        delay_probability=round(probability, 3),
        confidence=round(max(probability, 1 - probability), 3),
        delay_severity=delay_severity,
        origin=origin,
        destination=destination,
        departure_time=dep_time.isoformat(),
        airline=airline,
        weather_available=weather_origin is not None and weather_dest is not None,
        origin_weather=WeatherData(**weather_origin) if weather_origin else None,
        destination_weather=WeatherData(**weather_dest) if weather_dest else None,
        risk_factors=risk_factors,
        recommendations=recommendations,
        warnings=warnings,
        model_version=MODEL_VERSION,
    )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
