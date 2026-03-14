"""
Real-Time Weather Integration Module
=====================================

Integrates live weather data from Tomorrow.io API for flight delay predictions.
Provides weather feature extraction and caching for low-latency predictions.

API: Tomorrow.io (free tier - 500 calls/day)
Features: 12 weather dimensions per flight route
"""

import os
import requests
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import hashlib
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeatherCache:
    """Simple in-memory cache for weather data."""
    
    def __init__(self, ttl_seconds=3600):
        self.ttl_seconds = ttl_seconds
        self.cache = {}
    
    def get(self, key: str) -> Optional[Dict]:
        """Retrieve cached weather data if valid."""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.utcnow().timestamp() - timestamp < self.ttl_seconds:
                return data
            del self.cache[key]
        return None
    
    def set(self, key: str, data: Dict) -> None:
        """Cache weather data with timestamp."""
        self.cache[key] = (data, datetime.utcnow().timestamp())
    
    def clear(self) -> None:
        """Clear all cached data."""
        self.cache.clear()


class TomorrowIOWeatherClient:
    """Tomorrow.io API client for real-time weather data."""
    
    BASE_URL = 'https://api.tomorrow.io/v4/weather/realtime'
    
    def __init__(self, api_key: Optional[str] = None, cache_ttl: int = 3600):
        """
        Initialize weather client.
        
        Args:
            api_key: Tomorrow.io API key (or set TOMORROW_IO_API_KEY env var)
            cache_ttl: Cache time-to-live in seconds
        """
        self.api_key = api_key or os.getenv('TOMORROW_IO_API_KEY')
        if not self.api_key:
            logger.warning("No API key provided. Using mock data mode.")
        
        self.cache = WeatherCache(ttl_seconds=cache_ttl)
        self.call_count = 0
        self.daily_limit = 500
    
    def get_weather(self, lat: float, lon: float) -> Optional[Dict]:
        """
        Fetch real-time weather for coordinates.
        
        Args:
            lat: Latitude
            lon: Longitude
        
        Returns:
            dict: Weather data or None on error
        """
        cache_key = f"{lat:.4f}_{lon:.4f}"
        
        # Try cache first
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        # Check API limits
        if self.call_count >= self.daily_limit:
            logger.warning("Daily API limit (500) reached. Using cached/mock data.")
            return None
        
        if not self.api_key:
            return self._get_mock_weather()
        
        try:
            params = {
                'location': f'{lat},{lon}',
                'apikey': self.api_key,
                'fields': ','.join([
                    'temperature', 'windSpeed', 'windGust', 'windDirection',
                    'humidity', 'dewPoint', 'pressure', 'visibility',
                    'precipitationIntensity', 'thunderstormProbability',
                    'cloudCover', 'weatherCode'
                ])
            }
            
            response = requests.get(self.BASE_URL, params=params, timeout=5)
            response.raise_for_status()
            
            self.call_count += 1
            data = response.json()
            
            # Cache the result
            self.cache.set(cache_key, data)
            
            return data
        
        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
    
    def _get_mock_weather(self) -> Dict:
        """Return mock weather data for testing."""
        return {
            'data': {
                'values': {
                    'temperature': 72.5,
                    'windSpeed': 8.3,
                    'windGust': 15.2,
                    'windDirection': 270,
                    'humidity': 65,
                    'dewPoint': 58.0,
                    'pressure': 1013.25,
                    'visibility': 9.8,
                    'precipitationIntensity': 0.0,
                    'thunderstormProbability': 5,
                    'cloudCover': 35,
                    'weatherCode': 1000
                }
            }
        }


class WeatherFeatureExtractor:
    """Extract ML-ready features from weather data."""
    
    # Weather code to severity mapping
    WEATHER_SEVERITY = {
        1000: 0,  # Clear
        1100: 0,  # Mostly Clear
        1101: 1,  # Partly Cloudy
        1102: 1,  # Mostly Cloudy
        1001: 1,  # Cloudy
        2000: 2,  # Fog
        2100: 2,  # Light Fog
        3000: 3,  # Light Wind
        3001: 3,  # Wind
        3002: 4,  # Strong Wind
        4000: 4,  # Drizzle
        4001: 4,  # Rain
        4200: 5,  # Light Rain
        4201: 5,  # Rain
        5000: 6,  # Snow
        5001: 7,  # Flurries
        5100: 7,  # Light Snow
        5101: 7,  # Snow
        6000: 8,  # Freezing Drizzle
        6001: 8,  # Freezing Rain
        6200: 8,  # Light Freezing Rain
        6201: 8,  # Freezing Rain
        7000: 9,  # Ice Pellets
        7101: 9,  # Heavy Ice Pellets
        7102: 9,  # Ice Pellets
        8000: 9,  # Thunderstorm
    }
    
    def extract_features(self, weather_data: Dict) -> Dict[str, float]:
        """
        Extract 12 weather features for ML model.
        
        Args:
            weather_data: Raw API response
        
        Returns:
            dict: Feature values keyed by name
        """
        if not weather_data or 'data' not in weather_data:
            return self._get_default_features()
        
        values = weather_data['data']['values']
        
        # Normalize temperature to Fahrenheit if needed
        temp = float(values.get('temperature', 70))
        
        # Extract weather code severity
        weather_code = int(values.get('weatherCode', 1000))
        severity = self.WEATHER_SEVERITY.get(weather_code, 0)
        
        # Determine thunderstorm probability
        thunderstorm_prob = float(values.get('thunderstormProbability', 0))
        
        return {
            'temperature': temp,
            'wind_speed': float(values.get('windSpeed', 0)),
            'visibility': float(values.get('visibility', 10)),
            'precipitation': float(values.get('precipitationIntensity', 0)),
            'thunderstorm_probability': thunderstorm_prob,
            'humidity': float(values.get('humidity', 50)),
            'pressure': float(values.get('pressure', 1013.25)),
            'dew_point': float(values.get('dewPoint', 50)),
            'wind_gust': float(values.get('windGust', 0)),
            'cloud_coverage': float(values.get('cloudCover', 0)),
            'ceiling': max(0, float(values.get('visibility', 10)) * 1000),  # Estimate
            'weather_severity': severity
        }
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default/neutral feature values."""
        return {
            'temperature': 70.0,
            'wind_speed': 0.0,
            'visibility': 10.0,
            'precipitation': 0.0,
            'thunderstorm_probability': 0.0,
            'humidity': 50.0,
            'pressure': 1013.25,
            'dew_point': 50.0,
            'wind_gust': 0.0,
            'cloud_coverage': 0.0,
            'ceiling': 10000.0,
            'weather_severity': 0
        }


class RealTimeWeatherService:
    """High-level service for weather-integrated predictions."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.weather_client = TomorrowIOWeatherClient(api_key=api_key)
        self.feature_extractor = WeatherFeatureExtractor()
    
    def get_weather_features(self, lat: float, lon: float) -> Dict[str, float]:
        """
        Get weather features for a location.
        
        Args:
            lat: Latitude
            lon: Longitude
        
        Returns:
            dict: 12 weather features ready for ML model
        """
        weather_data = self.weather_client.get_weather(lat, lon)
        return self.feature_extractor.extract_features(weather_data)
    
    def get_features_for_flight(self, flight_info: Dict) -> Dict[str, float]:
        """
        Get weather features for a flight's departure airport.
        
        Args:
            flight_info: Dict with 'origin_lat', 'origin_lon' keys
        
        Returns:
            dict: Weather features
        """
        lat = flight_info.get('origin_lat')
        lon = flight_info.get('origin_lon')
        
        if lat is None or lon is None:
            raise ValueError("Flight info must contain origin_lat and origin_lon")
        
        return self.get_weather_features(lat, lon)
    
    def get_cache_status(self) -> Dict:
        """Get cache and API usage statistics."""
        return {
            'cached_locations': len(self.weather_client.cache.cache),
            'api_calls_today': self.weather_client.call_count,
            'daily_limit': self.weather_client.daily_limit,
            'remaining_calls': self.weather_client.daily_limit - self.weather_client.call_count
        }


# Module-level singleton for convenience
_service = None


def get_service(api_key: Optional[str] = None) -> RealTimeWeatherService:
    """Get or create module-level weather service."""
    global _service
    if _service is None:
        _service = RealTimeWeatherService(api_key=api_key)
    return _service
def get_weather_features(lat: float, lon: float) -> Dict[str, float]:
    """Convenience function to get weather features."""
    return get_service().get_weather_features(lat, lon)
