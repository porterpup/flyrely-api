"""
FlyRely Flight Delay Prediction API
====================================
FastAPI backend for predicting flight delay risk.

Features:
- Real-time weather integration (Tomorrow.io or OpenWeatherMap)
- ML-based delay probability prediction
- Risk level classification (Low/Medium/High)

Usage:
    pip install fastapi uvicorn httpx python-dotenv joblib pandas numpy scikit-learn
    uvicorn main:app --reload

API Endpoints:
    GET  /                  - API info
    GET  /health            - Health check
    POST /predict           - Predict delay risk for a flight
    GET  /airports          - List supported airports
"""

import os
import json
import csv
import time
import logging
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path
from io import StringIO
import re

import numpy as np
import pandas as pd
import joblib
import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Weather API configuration (set via environment variables)
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")  # Tomorrow.io or OpenWeatherMap
WEATHER_API_PROVIDER = os.getenv("WEATHER_API_PROVIDER", "tomorrow")  # "tomorrow" or "openweathermap"

# Email notification configuration
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
AVIATION_STACK_KEY = os.getenv("AVIATION_STACK_KEY", "")
NOTIFY_FROM_EMAIL = os.getenv("NOTIFY_FROM_EMAIL", "alerts@flyrely.app")

# Model paths
MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "flight_delay_model.joblib"
FEATURES_PATH = MODEL_DIR / "feature_names.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
SEVERITY_MODEL_PATH = MODEL_DIR / "delay_severity_model.joblib"
SEVERITY_METADATA_PATH = MODEL_DIR / "delay_severity_metadata.json"

# Usage log path (JSONL — one JSON record per line)
USAGE_LOG_PATH = Path(__file__).parent / "usage_log.jsonl"

# Weather API cost estimate: Tomorrow.io free tier = 500 calls/day
# Paid plan ~$0.001 per call above free tier (rough estimate)
WEATHER_COST_PER_CALL = 0.001  # USD

# Airport coordinates for weather lookups
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
    description="Flight delay prediction API with real-time weather integration",
    version="1.0.0",
)

# CORS middleware for frontend integration
# Note: allow_credentials=True is incompatible with allow_origins=["*"] per CORS spec.
# Using allow_credentials=False so the wildcard origin works correctly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
# =============================================================================
# Usage Logging
# =============================================================================

class UsageLogger:
    """Logs each /predict call to a JSONL file for cost forecasting."""

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

    def clear(self) -> None:
        if self.log_path.exists():
            self.log_path.unlink()


usage_logger = UsageLogger(USAGE_LOG_PATH)


class UsageTrackingMiddleware(BaseHTTPMiddleware):
    """Times every /predict request and records it to the usage log."""

    async def dispatch(self, request: Request, call_next):
        if request.url.path != "/predict" or request.method != "POST":
            return await call_next(request)

        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - start) * 1000)

        # We can't re-read the body after it's been consumed, so we log
        # what we can from the request scope + a flag on the response.
        # Detailed fields (origin, dest, etc.) are logged inside /predict itself.
        logger.info(f"[usage] POST /predict completed in {elapsed_ms}ms — status={response.status_code}")
        return response


# Register middleware now that the class is defined
app.add_middleware(UsageTrackingMiddleware)

# =============================================================================
# Pydantic Models
# =============================================================================

class FlightRequest(BaseModel):
    """Request model for flight delay prediction."""
    origin: str = Field(..., description="Origin airport code (e.g., 'JFK')", min_length=3, max_length=3)
    destination: str = Field(..., description="Destination airport code (e.g., 'LAX')", min_length=3, max_length=3)
    departure_time: datetime = Field(..., description="Scheduled departure time (ISO format)")
    airline: Optional[str] = Field(None, description="Airline code (e.g., 'AA', 'DL', 'UA')")

    class Config:
        json_schema_extra = {
            "example": {
                "origin": "JFK",
                "destination": "LAX",
                "departure_time": "2025-03-15T14:30:00",
                "airline": "AA"
            }
        }


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
    risk_level: str = Field(..., description="Risk level: 'low', 'medium', or 'high'")
    delay_probability: float = Field(..., description="Probability of any delay (0-1)")
    confidence: float = Field(..., description="Model confidence (0-1)")

    # Delay severity (new) — only meaningful when delay_probability > 0
    delay_severity: Optional[DelaySeverity] = Field(
        None, description="Conditional breakdown of delay severity if a delay occurs"
    )

    # Flight details
    origin: str
    destination: str
    departure_time: str
    airline: Optional[str]

    # Weather data (if available)
    origin_weather: Optional[WeatherData] = None
    destination_weather: Optional[WeatherData] = None

    # Factors contributing to risk
    risk_factors: list[str] = Field(default_factory=list)

    # Recommendations
    recommendations: list[str] = Field(default_factory=list)


# =============================================================================
# Weather Service
# =============================================================================


class NotifyRequest(BaseModel):
    flight_number: str
    origin: str
    destination: str
    scheduled_departure: str   # ISO string e.g. "2026-03-01T14:30:00"
    airline: Optional[str] = None
    risk_level: str
    delay_probability: float
    delay_minutes: Optional[int] = None
    delay_reason: Optional[str] = None
    recipient_email: str


class FlightLookupResponse(BaseModel):
    found: bool
    flight_number: str
    airline_name: Optional[str] = None
    airline_code: Optional[str] = None
    origin: Optional[str] = None
    destination: Optional[str] = None
    scheduled_departure: Optional[str] = None   # "HH:MM" local
    scheduled_arrival: Optional[str] = None
    status: Optional[str] = None
    error: Optional[str] = None

class WeatherService:
    """Service for fetching real-time weather data."""

    def __init__(self, api_key: str, provider: str = "tomorrow"):
        self.api_key = api_key
        self.provider = provider
        self.client = httpx.AsyncClient(timeout=10.0, proxy=None)
        self._cache = {}  # Simple in-memory cache
        self._cache_ttl = 1800  # 30 minutes

    async def get_weather(self, airport_code: str) -> Optional[dict]:
        """Get current weather for an airport."""
        if not self.api_key:
            logger.warning("No weather API key configured")
            return None

        coords = AIRPORT_COORDS.get(airport_code.upper())
        if not coords:
            logger.warning(f"Unknown airport: {airport_code}")
            return None

        # Check cache
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
        url = "https://api.tomorrow.io/v4/weather/realtime"
        params = {
            "location": f"{lat},{lon}",
            "apikey": self.api_key,
            "units": "imperial"
        }

        response = await self.client.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        values = data.get("data", {}).get("values", {})
        return {
            "temperature_f": values.get("temperature", 70),
            "wind_speed_mph": values.get("windSpeed", 10),
            "visibility_miles": values.get("visibility", 10),
            "conditions": self._get_conditions_tomorrow(values)
        }

    async def _fetch_openweathermap(self, coords: tuple) -> Optional[dict]:
        """Fetch weather from OpenWeatherMap API."""
        lat, lon = coords
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "imperial"
        }

        response = await self.client.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        return {
            "temperature_f": data.get("main", {}).get("temp", 70),
            "wind_speed_mph": data.get("wind", {}).get("speed", 10),
            "visibility_miles": data.get("visibility", 10000) / 1609.34,  # m to miles
            "conditions": data.get("weather", [{}])[0].get("description", "clear")
        }

    def _get_conditions_tomorrow(self, values: dict) -> str:
        """Convert Tomorrow.io weather code to description."""
        code = values.get("weatherCode", 1000)
        conditions = {
            1000: "clear", 1100: "mostly clear", 1101: "partly cloudy",
            1102: "mostly cloudy", 1001: "cloudy", 2000: "fog", 2100: "light fog",
            4000: "drizzle", 4001: "rain", 4200: "light rain", 4201: "heavy rain",
            5000: "snow", 5001: "flurries", 5100: "light snow", 5101: "heavy snow",
            6000: "freezing drizzle", 6001: "freezing rain", 7000: "ice pellets",
            7101: "heavy ice pellets", 7102: "light ice pellets", 8000: "thunderstorm"
        }
        return conditions.get(code, "unknown")


# Initialize weather service
weather_service = WeatherService(WEATHER_API_KEY, WEATHER_API_PROVIDER)

# =============================================================================
# Model Service
# =============================================================================

class ModelService:
    """Service for loading and running the ML model."""

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.metadata = None
        self.severity_model = None
        self.severity_metadata = None
        self._load_model()

    def _load_model(self):
        """Load the trained model and metadata."""
        try:
            if MODEL_PATH.exists():
                self.model = joblib.load(MODEL_PATH)
                logger.info(f"Model loaded from {MODEL_PATH}")
            else:
                logger.warning(f"Model not found at {MODEL_PATH}")

            if FEATURES_PATH.exists():
                self.feature_names = joblib.load(FEATURES_PATH)
                logger.info(f"Feature names loaded: {len(self.feature_names)} features")

            if METADATA_PATH.exists():
                with open(METADATA_PATH) as f:
                    self.metadata = json.load(f)
                logger.info(f"Model metadata loaded")

            if SEVERITY_MODEL_PATH.exists():
                self.severity_model = joblib.load(SEVERITY_MODEL_PATH)
                logger.info(f"Severity model loaded from {SEVERITY_MODEL_PATH}")
            else:
                logger.warning(f"Severity model not found at {SEVERITY_MODEL_PATH}")

            if SEVERITY_METADATA_PATH.exists():
                with open(SEVERITY_METADATA_PATH) as f:
                    self.severity_metadata = json.load(f)
                logger.info(f"Severity metadata loaded")

        except Exception as e:
            logger.error(f"Error loading model: {e}")

    def predict(self, features: dict) -> tuple[float, str]:
        """
        Make a prediction.

        Returns:
            tuple: (probability, risk_level)
        """
        if self.model is None:
            # Fallback to simple heuristic if model not loaded
            return self._heuristic_predict(features)

        # Build feature vector
        feature_vector = self._build_feature_vector(features)

        # Get probability
        proba = self.model.predict_proba([feature_vector])[0, 1]

        # Convert to risk level
        risk_level = self._probability_to_risk(proba)

        return proba, risk_level

    def _build_feature_vector(self, features: dict) -> list:
        """Build feature vector in the correct order."""
        if self.feature_names is None:
            # Default feature order
            self.feature_names = [
                'dep_hour', 'day_of_week', 'month', 'is_weekend',
                'is_morning', 'is_afternoon', 'is_evening', 'is_night',
                'season', 'is_holiday_period', 'days_to_holiday',
                'origin_delay_rate', 'dest_delay_rate', 'route_delay_rate',
                'airline_delay_rate', 'origin_is_hub', 'dest_is_hub', 'distance',
                'origin_temp_f', 'origin_wind_mph', 'origin_visibility',
                'low_visibility', 'high_wind', 'freezing_temp', 'severe_weather'
            ]

        vector = []
        for name in self.feature_names:
            vector.append(features.get(name, 0))

        return vector

    def _probability_to_risk(self, prob: float) -> str:
        """Convert probability to risk level."""
        if prob < 0.25:
            return "low"
        elif prob < 0.50:
            return "medium"
        else:
            return "high"

    def _heuristic_predict(self, features: dict) -> tuple[float, str]:
        """Simple heuristic prediction as fallback."""
        prob = 0.15  # Base rate

        # Adjust for time of day
        hour = features.get('dep_hour', 12)
        if 17 <= hour <= 21:
            prob += 0.10  # Evening peak
        elif 6 <= hour <= 9:
            prob += 0.05  # Morning rush

        # Adjust for weather
        if features.get('low_visibility', 0):
            prob += 0.15
        if features.get('high_wind', 0):
            prob += 0.10
        if features.get('freezing_temp', 0):
            prob += 0.12

        # Adjust for airport
        prob += features.get('origin_delay_rate', 0.20) - 0.20

        prob = min(max(prob, 0), 1)
        risk_level = self._probability_to_risk(prob)

        return prob, risk_level

    def predict_severity(self, features: dict) -> Optional["DelaySeverity"]:
        """
        Predict delay severity distribution using the multi-class model.

        Returns a DelaySeverity with conditional probabilities (given a delay occurs):
          minor_pct, moderate_pct, severe_pct sum to 1.0 across the three delay buckets.
        Returns None if the severity model is not loaded.
        """
        if self.severity_model is None:
            return None

        try:
            # Build 18-feature vector (severity model uses same features as binary model)
            severity_feature_names = [
                'dep_hour', 'day_of_week', 'month', 'is_weekend',
                'is_morning', 'is_afternoon', 'is_evening', 'is_night',
                'season', 'is_holiday_period', 'days_to_holiday',
                'origin_delay_rate', 'dest_delay_rate', 'route_delay_rate',
                'airline_delay_rate', 'origin_is_hub', 'dest_is_hub', 'distance',
            ]
            vector = [features.get(name, 0) for name in severity_feature_names]

            # classes: [on_time, minor, moderate, severe]
            proba = self.severity_model.predict_proba([vector])[0]  # shape (4,)

            # Conditional distribution among the three delay buckets (indices 1,2,3)
            delay_proba = proba[1:]  # [minor, moderate, severe]
            delay_total = delay_proba.sum()

            if delay_total < 1e-9:
                # Model thinks almost no chance of delay — return uniform conditional
                minor_pct = moderate_pct = severe_pct = round(1 / 3, 3)
            else:
                conditional = delay_proba / delay_total
                minor_pct = round(float(conditional[0]), 3)
                moderate_pct = round(float(conditional[1]), 3)
                severe_pct = round(float(conditional[2]), 3)

            # Most likely severity bucket
            bucket_idx = int(np.argmax(delay_proba))  # 0=minor,1=moderate,2=severe
            bucket_labels = ["minor", "moderate", "severe"]
            bucket_ranges = ["15–44 min", "45–119 min", "120+ min"]
            expected_label = bucket_labels[bucket_idx]
            expected_range = bucket_ranges[bucket_idx]

            return DelaySeverity(
                minor_pct=minor_pct,
                moderate_pct=moderate_pct,
                severe_pct=severe_pct,
                expected_delay_label=expected_label,
                expected_delay_range=expected_range,
            )

        except Exception as e:
            logger.error(f"Error in predict_severity: {e}")
            return None


# Initialize model service
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


def get_days_to_holiday(date: datetime) -> int:
    """Calculate days to nearest major US holiday."""
    holidays_2025 = [
        datetime(2025, 1, 1), datetime(2025, 1, 20), datetime(2025, 2, 17),
        datetime(2025, 5, 26), datetime(2025, 7, 4), datetime(2025, 9, 1),
        datetime(2025, 11, 27), datetime(2025, 12, 25),
    ]
    holidays_2026 = [
        datetime(2026, 1, 1), datetime(2026, 1, 19), datetime(2026, 2, 16),
        datetime(2026, 5, 25), datetime(2026, 7, 4), datetime(2026, 9, 7),
        datetime(2026, 11, 26), datetime(2026, 12, 25),
    ]
    holidays = holidays_2025 + holidays_2026

    min_days = min(abs((date - h).days) for h in holidays)
    return min_days


def estimate_distance(origin: str, dest: str) -> float:
    """Estimate flight distance in miles."""
    # Simple approximation based on coordinates
    coords1 = AIRPORT_COORDS.get(origin.upper(), (39.8, -98.5))  # Default: center US
    coords2 = AIRPORT_COORDS.get(dest.upper(), (39.8, -98.5))

    # Haversine formula (simplified)
    lat1, lon1 = coords1
    lat2, lon2 = coords2

    dlat = abs(lat2 - lat1)
    dlon = abs(lon2 - lon1)

    # Rough approximation: 1 degree ≈ 69 miles
    distance = ((dlat * 69) ** 2 + (dlon * 69 * 0.7) ** 2) ** 0.5

    return max(distance, 100)  # Minimum 100 miles


def get_risk_factors(features: dict, weather_origin: dict, weather_dest: dict) -> list[str]:
    """Identify risk factors for the flight."""
    factors = []

    # Time-based factors
    hour = features.get('dep_hour', 12)
    if 17 <= hour <= 21:
        factors.append("Evening departure (peak congestion time)")

    if features.get('is_holiday_period'):
        factors.append("Holiday travel period (high traffic)")

    # Weather factors
    if weather_origin:
        if weather_origin.get('visibility_miles', 10) < 3:
            factors.append(f"Low visibility at origin ({weather_origin['visibility_miles']:.1f} mi)")
        if weather_origin.get('wind_speed_mph', 0) > 20:
            factors.append(f"High winds at origin ({weather_origin['wind_speed_mph']:.0f} mph)")
        if weather_origin.get('temperature_f', 70) < 32:
            factors.append(f"Freezing conditions at origin ({weather_origin['temperature_f']:.0f}°F)")

    if weather_dest:
        if weather_dest.get('visibility_miles', 10) < 3:
            factors.append(f"Low visibility at destination ({weather_dest['visibility_miles']:.1f} mi)")
        if weather_dest.get('wind_speed_mph', 0) > 20:
            factors.append(f"High winds at destination ({weather_dest['wind_speed_mph']:.0f} mph)")

    # Airport factors
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

    # Weather-specific recommendations
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
        "version": "1.0.0",
        "description": "Flight delay prediction with real-time weather",
        "endpoints": {
            "POST /predict": "Predict delay risk for a flight",
            "GET /airports": "List supported airports",
            "GET /health": "Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_service.model is not None,
        "weather_api_configured": bool(WEATHER_API_KEY),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/airports")
async def list_airports():
    """List supported airports."""
    airports = []
    for code, coords in AIRPORT_COORDS.items():
        airports.append({
            "code": code,
            "delay_rate": AIRPORT_DELAY_RATES.get(code, 0.20),
            "is_hub": code in HUB_AIRPORTS
        })

    return {
        "count": len(airports),
        "airports": sorted(airports, key=lambda x: x["code"])
    }


@app.get("/usage")
async def get_usage(days: int = Query(default=30, ge=1, le=365)):
    """
    API usage summary for cost forecasting.

    Returns daily call counts, total weather API calls, and estimated cost
    for the last `days` days (default 30).
    """
    all_entries = usage_logger.read_all()
    if not all_entries:
        return {
            "period_days": days,
            "total_predictions": 0,
            "total_weather_calls": 0,
            "estimated_cost_usd": 0.0,
            "daily": [],
            "by_route": [],
            "by_risk_level": {},
            "message": "No usage data recorded yet. Make some predictions first.",
        }

    cutoff = datetime.now() - timedelta(days=days)
    entries = [e for e in all_entries if datetime.fromisoformat(e["ts"]) >= cutoff]

    # Daily aggregation
    daily: dict[str, dict] = {}
    route_counts: dict[str, int] = {}
    risk_counts: dict[str, int] = {"low": 0, "medium": 0, "high": 0}
    total_weather = 0

    for e in entries:
        date = e.get("date", e["ts"][:10])
        if date not in daily:
            daily[date] = {"date": date, "predictions": 0, "weather_calls": 0}
        daily[date]["predictions"] += 1
        wc = e.get("weather_calls", 0)
        daily[date]["weather_calls"] += wc
        total_weather += wc

        route = f"{e.get('origin', '?')}→{e.get('destination', '?')}"
        route_counts[route] = route_counts.get(route, 0) + 1

        rl = e.get("risk_level", "unknown")
        if rl in risk_counts:
            risk_counts[rl] += 1

    # Sort daily descending
    daily_list = sorted(daily.values(), key=lambda x: x["date"], reverse=True)

    # Top routes
    top_routes = sorted(
        [{"route": r, "predictions": c} for r, c in route_counts.items()],
        key=lambda x: x["predictions"],
        reverse=True,
    )[:10]

    # Cost estimate: weather calls above free tier (500/day) cost ~$0.001 each
    # Simple estimate: total weather calls * cost per call (ignores free tier for simplicity)
    estimated_cost = round(total_weather * WEATHER_COST_PER_CALL, 4)

    return {
        "period_days": days,
        "total_predictions": len(entries),
        "total_weather_calls": total_weather,
        "estimated_cost_usd": estimated_cost,
        "avg_predictions_per_day": round(len(entries) / max(days, 1), 1),
        "daily": daily_list,
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
    fieldnames = ["ts", "date", "origin", "destination", "airline", "departure_time",
                  "risk_level", "delay_probability", "weather_fetched", "weather_calls", "model_ms"]
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(entries)

    output.seek(0)
    filename = f"flyrely_usage_{datetime.now().strftime('%Y%m%d')}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )




# =============================================================================
# Notification & Flight Lookup
# =============================================================================

@app.post("/notify")
async def send_notification(req: NotifyRequest):
    """
    Send a delay alert email via Resend.
    Called by the frontend when a flight's risk is medium or high.
    """
    if not RESEND_API_KEY:
        raise HTTPException(status_code=503, detail="Email notifications not configured")

    import resend
    resend.api_key = RESEND_API_KEY

    risk_emoji = "🔴" if req.risk_level == "high" else "🟡"
    risk_label = "High Risk" if req.risk_level == "high" else "Medium Risk"
    dep_dt = datetime.fromisoformat(req.scheduled_departure)
    dep_formatted = dep_dt.strftime("%b %-d at %-I:%M %p")

    delay_line = ""
    if req.delay_minutes and req.delay_minutes > 0:
        delay_line = f"<p style='margin:8px 0;color:#92400e;'><strong>Expected delay:</strong> {req.delay_minutes}–{req.delay_minutes+30} minutes</p>"

    reason_line = ""
    if req.delay_reason:
        reason_line = f"<p style='margin:8px 0;color:#78350f;font-size:14px;'>{req.delay_reason}</p>"

    html = f"""
    <div style='font-family:sans-serif;max-width:480px;margin:0 auto;'>
      <div style='background:#1e3a5f;padding:24px;border-radius:12px 12px 0 0;'>
        <h1 style='color:white;margin:0;font-size:20px;'>✈️ FlyRely Alert</h1>
        <p style='color:rgba(255,255,255,0.7);margin:4px 0 0;font-size:14px;'>Flight delay prediction</p>
      </div>
      <div style='background:white;padding:24px;border:1px solid #e2e8f0;border-radius:0 0 12px 12px;'>
        <div style='display:flex;align-items:center;gap:12px;margin-bottom:16px;'>
          <span style='font-size:32px;'>{risk_emoji}</span>
          <div>
            <p style='margin:0;font-size:18px;font-weight:700;color:#0f172a;'>{req.flight_number}</p>
            <p style='margin:2px 0 0;color:#64748b;font-size:14px;'>{req.airline_name or req.airline_code or "Unknown airline"}</p>
          </div>
        </div>
        <div style='background:#f8fafc;border-radius:8px;padding:16px;margin-bottom:16px;'>
          <p style='margin:0 0 8px;font-weight:600;color:#0f172a;'>{req.origin} → {req.destination}</p>
          <p style='margin:8px 0;color:#475569;'>Scheduled: {dep_formatted}</p>
          {delay_line}
          {reason_line}
        </div>
        <div style='background:#fef3c7;border:1px solid #fde68a;border-radius:8px;padding:12px;margin-bottom:16px;'>
          <p style='margin:0;font-weight:600;color:#92400e;'>{risk_label} · {round(req.delay_probability * 100)}% delay probability</p>
        </div>
        <p style='color:#94a3b8;font-size:12px;margin:0;'>
          This alert was sent by FlyRely. Predictions are based on historical data and current weather conditions.
        </p>
      </div>
    </div>
    """

    try:
        params = {
            "from": NOTIFY_FROM_EMAIL,
            "to": [req.recipient_email],
            "subject": f"{risk_emoji} {req.flight_number} – {risk_label} of Delay ({dep_formatted})",
            "html": html,
        }
        resend.Emails.send(params)
        return {"sent": True, "to": req.recipient_email}
    except Exception as e:
        logger.error(f"Resend error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send email: {str(e)}")


@app.get("/flight-lookup", response_model=FlightLookupResponse)
async def lookup_flight(
    flight_number: str = Query(..., description="e.g. UA1071"),
    date: str = Query(..., description="YYYY-MM-DD"),
):
    """
    Look up a flight by number and date using AviationStack.
    Returns origin, destination, scheduled departure time.
    """
    if not AVIATION_STACK_KEY:
        return FlightLookupResponse(
            found=False,
            flight_number=flight_number,
            error="Flight lookup not configured"
        )

    # Parse airline code and flight number
    match = re.match(r'^([A-Z]{2})(\d+)$', flight_number.upper())
    if not match:
        return FlightLookupResponse(found=False, flight_number=flight_number, error="Invalid flight number format")

    airline_iata, flight_num = match.groups()

    try:
        # AviationStack real-time flights endpoint
        # Free plan: HTTP only, no flight_date filter (date filtering is a paid feature)
        # We search by flight_iata and return the most recent result
        async with httpx.AsyncClient(timeout=10.0, proxy=None) as client:
            resp = await client.get(
                "http://api.aviationstack.com/v1/flights",
                params={
                    "access_key": AVIATION_STACK_KEY,
                    "flight_iata": flight_number.upper(),
                    "limit": 1,
                },
            )
            data = resp.json()

        # AviationStack returns error info in JSON even when HTTP status is non-200
        if "error" in data:
            err = data["error"]
            err_msg = err.get("info", str(err))
            logger.error(f"AviationStack API error: {err_msg}")
            return FlightLookupResponse(found=False, flight_number=flight_number, error=err_msg)

        flights = data.get("data", [])
        if not flights:
            return FlightLookupResponse(found=False, flight_number=flight_number, error="Flight not found")

        f = flights[0]
        dep = f.get("departure", {})
        arr = f.get("arrival", {})
        airline = f.get("airline", {})

        # Parse scheduled departure time (HH:MM)
        sched_dep_iso = dep.get("scheduled", "")
        sched_arr_iso = arr.get("scheduled", "")

        dep_time = None
        if sched_dep_iso:
            try:
                dep_time = datetime.fromisoformat(sched_dep_iso.replace("Z", "+00:00")).strftime("%H:%M")
            except Exception:
                dep_time = sched_dep_iso[11:16] if len(sched_dep_iso) >= 16 else None

        arr_time = None
        if sched_arr_iso:
            try:
                arr_time = datetime.fromisoformat(sched_arr_iso.replace("Z", "+00:00")).strftime("%H:%M")
            except Exception:
                arr_time = sched_arr_iso[11:16] if len(sched_arr_iso) >= 16 else None

        return FlightLookupResponse(
            found=True,
            flight_number=flight_number.upper(),
            airline_name=airline.get("name"),
            airline_code=airline.get("iata", airline_iata),
            origin=dep.get("iata"),
            destination=arr.get("iata"),
            scheduled_departure=dep_time,
            scheduled_arrival=arr_time,
            status=f.get("flight_status"),
        )

    except httpx.HTTPStatusError as e:
        logger.error(f"AviationStack HTTP error: {e} — body: {e.response.text[:500]}")
        return FlightLookupResponse(found=False, flight_number=flight_number, error="Lookup service error")
    except Exception as e:
        logger.error(f"Flight lookup error: {e}")
        return FlightLookupResponse(found=False, flight_number=flight_number, error=str(e))


@app.post("/predict", response_model=PredictionResponse)
async def predict_delay(request: FlightRequest):
    """
    Predict flight delay risk.

    Takes origin, destination, departure time, and optional airline.
    Returns risk level, probability, and contributing factors.
    """
    origin = request.origin.upper()
    destination = request.destination.upper()
    dep_time = request.departure_time
    airline = request.airline.upper() if request.airline else None

    # Validate airports
    if origin not in AIRPORT_COORDS:
        raise HTTPException(status_code=400, detail=f"Unknown origin airport: {origin}")
    if destination not in AIRPORT_COORDS:
        raise HTTPException(status_code=400, detail=f"Unknown destination airport: {destination}")

    # Get weather data
    weather_origin = await weather_service.get_weather(origin)
    weather_dest = await weather_service.get_weather(destination)

    # Build features
    features = {
        # Time features
        'dep_hour': dep_time.hour,
        'day_of_week': dep_time.isoweekday(),
        'month': dep_time.month,
        'is_weekend': 1 if dep_time.isoweekday() >= 6 else 0,
        'is_morning': 1 if 5 <= dep_time.hour < 12 else 0,
        'is_afternoon': 1 if 12 <= dep_time.hour < 18 else 0,
        'is_evening': 1 if 18 <= dep_time.hour < 22 else 0,
        'is_night': 1 if dep_time.hour >= 22 or dep_time.hour < 5 else 0,
        'season': get_season(dep_time.month),
        'days_to_holiday': get_days_to_holiday(dep_time),
        'is_holiday_period': 1 if get_days_to_holiday(dep_time) <= 3 else 0,

        # Airport features
        'origin_delay_rate': AIRPORT_DELAY_RATES.get(origin, 0.20),
        'dest_delay_rate': AIRPORT_DELAY_RATES.get(destination, 0.20),
        'route_delay_rate': (AIRPORT_DELAY_RATES.get(origin, 0.20) + AIRPORT_DELAY_RATES.get(destination, 0.20)) / 2,
        'airline_delay_rate': AIRLINE_DELAY_RATES.get(airline, 0.21) if airline else 0.21,
        'origin_is_hub': 1 if origin in HUB_AIRPORTS else 0,
        'dest_is_hub': 1 if destination in HUB_AIRPORTS else 0,
        'distance': estimate_distance(origin, destination),
    }

    # Add weather features
    if weather_origin:
        features['origin_temp_f'] = weather_origin['temperature_f']
        features['origin_wind_mph'] = weather_origin['wind_speed_mph']
        features['origin_visibility'] = weather_origin['visibility_miles']
        features['low_visibility'] = 1 if weather_origin['visibility_miles'] < 3 else 0
        features['high_wind'] = 1 if weather_origin['wind_speed_mph'] > 20 else 0
        features['freezing_temp'] = 1 if weather_origin['temperature_f'] < 32 else 0
        features['severe_weather'] = 1 if (features['low_visibility'] or features['high_wind'] or features['freezing_temp']) else 0
    else:
        # Defaults if no weather
        features.update({
            'origin_temp_f': 60, 'origin_wind_mph': 10, 'origin_visibility': 10,
            'low_visibility': 0, 'high_wind': 0, 'freezing_temp': 0, 'severe_weather': 0
        })

    # Make prediction
    _predict_start = time.perf_counter()
    probability, risk_level = model_service.predict(features)
    _predict_ms = round((time.perf_counter() - _predict_start) * 1000)

    # Predict delay severity distribution
    delay_severity = model_service.predict_severity(features)

    # Get risk factors and recommendations
    risk_factors = get_risk_factors(features, weather_origin, weather_dest)
    recommendations = get_recommendations(risk_level, risk_factors)

    # Log this prediction for usage/cost tracking
    usage_logger.record({
        "ts": datetime.now().isoformat(),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "origin": origin,
        "destination": destination,
        "airline": airline,
        "departure_time": dep_time.isoformat(),
        "risk_level": risk_level,
        "delay_probability": round(probability, 3),
        "weather_fetched": weather_origin is not None or weather_dest is not None,
        "weather_calls": (1 if weather_origin is not None else 0) + (1 if weather_dest is not None else 0),
        "model_ms": _predict_ms,
    })

    # Build response
    response = PredictionResponse(
        risk_level=risk_level,
        delay_probability=round(probability, 3),
        confidence=round(max(probability, 1 - probability), 3),
        delay_severity=delay_severity,
        origin=origin,
        destination=destination,
        departure_time=dep_time.isoformat(),
        airline=airline,
        origin_weather=WeatherData(**weather_origin) if weather_origin else None,
        destination_weather=WeatherData(**weather_dest) if weather_dest else None,
        risk_factors=risk_factors,
        recommendations=recommendations
    )

    return response


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
