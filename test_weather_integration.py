"""
Comprehensive Test Suite for Weather Integration
================================================

Tests for phase1_retraining.py and real_time_weather.py modules.
Includes unit tests, integration tests, and mock data validation.
"""

import unittest
import json
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

# Import modules under test
from phase1_retraining import Phase1RetrainingPipeline
from real_time_weather import (
    WeatherCache, TomorrowIOWeatherClient, WeatherFeatureExtractor,
    RealTimeWeatherService
)
class TestWeatherCache(unittest.TestCase):
    """Test WeatherCache functionality."""
    
    def setUp(self):
        self.cache = WeatherCache(ttl_seconds=2)
    
    def test_cache_set_get(self):
        """Test setting and getting cached values."""
        data = {'temp': 72, 'humidity': 65}
        self.cache.set('test_key', data)
        
        retrieved = self.cache.get('test_key')
        self.assertEqual(retrieved, data)
    
    def test_cache_miss(self):
        """Test cache miss returns None."""
        result = self.cache.get('nonexistent')
        self.assertIsNone(result)
    
    def test_cache_expiration(self):
        """Test cache TTL expiration."""
        import time
        data = {'temp': 72}
        self.cache.set('expiring', data)
        
        # Still valid
        self.assertIsNotNone(self.cache.get('expiring'))
        
        # Wait for expiration
        time.sleep(2.1)
        self.assertIsNone(self.cache.get('expiring'))
    
    def test_cache_clear(self):
        """Test clearing cache."""
        self.cache.set('key1', {'data': 1})
        self.cache.set('key2', {'data': 2})
        self.cache.clear()
        
        self.assertIsNone(self.cache.get('key1'))
        self.assertIsNone(self.cache.get('key2'))


class TestWeatherFeatureExtractor(unittest.TestCase):
    """Test weather feature extraction."""
    
    def setUp(self):
        self.extractor = WeatherFeatureExtractor()
    
    def test_extract_features_valid_data(self):
        """Test feature extraction from valid API response."""
        weather_data = {
            'data': {
                'values': {
                    'temperature': 75.5,
                    'windSpeed': 10.0,
                    'visibility': 9.5,
                    'precipitationIntensity': 0.1,
                    'thunderstormProbability': 15,
                    'humidity': 60,
                    'pressure': 1012.5,
                    'dewPoint': 55.0,
                    'windGust': 18.0,
                    'cloudCover': 40,
                    'weatherCode': 1101
                }
            }
        }
        
        features = self.extractor.extract_features(weather_data)
        
        self.assertEqual(features['temperature'], 75.5)
        self.assertEqual(features['wind_speed'], 10.0)
        self.assertEqual(features['visibility'], 9.5)
        self.assertEqual(features['precipitation'], 0.1)
        self.assertEqual(features['thunderstorm_probability'], 15)
        self.assertIn('weather_severity', features)
        self.assertEqual(len(features), 12)
    
    def test_extract_features_empty_data(self):
        """Test feature extraction with missing data."""
        features = self.extractor.extract_features(None)
        
        # Should return defaults
        self.assertEqual(features['temperature'], 70.0)
        self.assertEqual(features['wind_speed'], 0.0)
        self.assertEqual(len(features), 12)
    
    def test_weather_severity_mapping(self):
        """Test weather code to severity mapping."""
        test_cases = [
            (1000, 0),  # Clear
            (2000, 2),  # Fog
            (4000, 4),  # Drizzle
            (8000, 9),  # Thunderstorm
        ]
        
        for code, expected_severity in test_cases:
            weather_data = {
                'data': {'values': {'weatherCode': code}}
            }
            features = self.extractor.extract_features(weather_data)
            self.assertEqual(features['weather_severity'], expected_severity,
                           f"Code {code} should map to severity {expected_severity}")
class TestTomorrowIOWeatherClient(unittest.TestCase):
    """Test Tomorrow.io API client."""
    
    def setUp(self):
        self.client = TomorrowIOWeatherClient(api_key='test_key')
    
    def test_mock_weather_mode(self):
        """Test mock data generation when no API key."""
        client = TomorrowIOWeatherClient(api_key=None)
        data = client._get_mock_weather()
        
        self.assertIn('data', data)
        self.assertIn('values', data['data'])
        self.assertIn('temperature', data['data']['values'])
    
    @patch('requests.get')
    def test_get_weather_success(self, mock_get):
        """Test successful API call."""
        mock_response = Mock()
        mock_response.json.return_value = {
            'data': {
                'values': {
                    'temperature': 72,
                    'windSpeed': 8,
                    'humidity': 65
                }
            }
        }
        mock_get.return_value = mock_response
        
        result = self.client.get_weather(40.7128, -74.0060)
        
        self.assertIsNotNone(result)
        self.assertIn('data', result)
        self.assertEqual(self.client.call_count, 1)
    
    @patch('requests.get')
    def test_get_weather_caching(self, mock_get):
        """Test weather data caching."""
        mock_response = Mock()
        mock_response.json.return_value = {
            'data': {'values': {'temperature': 72}}
        }
        mock_get.return_value = mock_response
        
        # First call
        result1 = self.client.get_weather(40.7128, -74.0060)
        # Second call (should use cache)
        result2 = self.client.get_weather(40.7128, -74.0060)
        
        # Only one API call should have been made
        self.assertEqual(mock_get.call_count, 1)
        self.assertEqual(result1, result2)
    
    def test_api_limit_enforcement(self):
        """Test daily API call limit."""
        self.client.call_count = 500  # Max limit
        
        result = self.client.get_weather(40.7128, -74.0060)
        
        # Should return None due to limit
        self.assertIsNone(result)


class TestRealTimeWeatherService(unittest.TestCase):
    """Test high-level weather service."""
    
    def setUp(self):
        self.service = RealTimeWeatherService(api_key='test_key')
    
    def test_get_weather_features(self):
        """Test getting weather features for location."""
        # Mock the API call
        self.service.weather_client._get_mock_weather = lambda: {
            'data': {
                'values': {
                    'temperature': 72,
                    'windSpeed': 8,
                    'visibility': 10,
                    'precipitationIntensity': 0,
                    'thunderstormProbability': 0,
                    'humidity': 65,
                    'pressure': 1013.25,
                    'dewPoint': 58,
                    'windGust': 12,
                    'cloudCover': 25,
                    'weatherCode': 1000
                }
            }
        }
        
        features = self.service.get_weather_features(40.7128, -74.0060)
        
        self.assertIsInstance(features, dict)
        self.assertEqual(len(features), 12)
        self.assertIn('temperature', features)
    
    def test_get_features_for_flight(self):
        """Test getting weather for flight info dict."""
        flight_info = {
            'origin_lat': 40.7128,
            'origin_lon': -74.0060,
            'flight_number': 'AA123'
        }
        
        features = self.service.get_features_for_flight(flight_info)
        
        self.assertIsInstance(features, dict)
        self.assertEqual(len(features), 12)
    
    def test_get_features_for_flight_missing_coords(self):
        """Test error handling for missing coordinates."""
        flight_info = {'flight_number': 'AA123'}
        
        with self.assertRaises(ValueError):
            self.service.get_features_for_flight(flight_info)
    
    def test_cache_status(self):
        """Test cache status reporting."""
        status = self.service.get_cache_status()
        
        self.assertIn('cached_locations', status)
        self.assertIn('api_calls_today', status)
        self.assertIn('daily_limit', status)
        self.assertIn('remaining_calls', status)


class TestPhase1RetrainingPipeline(unittest.TestCase):
    """Test model retraining pipeline."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, 'test_model.joblib')
        self.pipeline = Phase1RetrainingPipeline(model_path=self.model_path)
        
        # Create test data
        self.X_test = self._create_test_features(100)
        self.y_test = np.random.randint(0, 2, 100)
    
    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_features(self, n_samples):
        """Create dummy feature matrix."""
        feature_dict = {}
        for name in self.pipeline.feature_names:
            feature_dict[name] = np.random.randn(n_samples) * 10
        return pd.DataFrame(feature_dict)
    
    def test_feature_names(self):
        """Test feature list is complete."""
        self.assertGreater(len(self.pipeline.feature_names), 0)
        self.assertIn('temperature', self.pipeline.feature_names)
        self.assertIn('wind_speed', self.pipeline.feature_names)
        self.assertEqual(len(self.pipeline.weather_features), 12)
    
    def test_training_pipeline(self):
        """Test full training workflow."""
        X = self._create_test_features(200)
        y = np.random.randint(0, 2, 200)
        
        metrics = self.pipeline.train(X, y)
        
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        self.assertGreaterEqual(metrics['precision'], 0)
        self.assertLessEqual(metrics['precision'], 1)
    
    def test_save_load_model(self):
        """Test model persistence."""
        # Train
        X = self._create_test_features(100)
        y = np.random.randint(0, 2, 100)
        self.pipeline.train(X, y)
        
        # Save
        self.pipeline.save_model()
        self.assertTrue(os.path.exists(self.model_path))
        
        # Load into new pipeline
        pipeline2 = Phase1RetrainingPipeline(model_path=self.model_path)
        pipeline2.load_model()
        
        self.assertIsNotNone(pipeline2.model)
        self.assertEqual(pipeline2.feature_names, self.pipeline.feature_names)
    
    def test_predict_batch(self):
        """Test batch prediction."""
        X = self._create_test_features(100)
        y = np.random.randint(0, 2, 100)
        self.pipeline.train(X, y)
        
        # Predict on new data
        X_new = self._create_test_features(10)
        predictions = self.pipeline.predict_batch(X_new)
        
        self.assertEqual(len(predictions), 10)
        self.assertTrue(all(p in [0, 1] for p in predictions))
    
    def test_predict_proba(self):
        """Test probability predictions."""
        X = self._create_test_features(100)
        y = np.random.randint(0, 2, 100)
        self.pipeline.train(X, y)
        
        X_new = self._create_test_features(5)
        probs = self.pipeline.predict_proba_batch(X_new)
        
        self.assertEqual(probs.shape, (5, 2))
        self.assertTrue(all(0 <= p <= 1 for row in probs for p in row))
    
    def test_feature_importance(self):
        """Test feature importance extraction."""
        X = self._create_test_features(100)
        y = np.random.randint(0, 2, 100)
        self.pipeline.train(X, y)
        
        importance = self.pipeline.get_feature_importance()
        
        self.assertEqual(len(importance), len(self.pipeline.feature_names))
        self.assertIn('feature', importance.columns)
        self.assertIn('importance', importance.columns)


class TestBackwardCompatibility(unittest.TestCase):
    """Test v2 is backward compatible with v1."""
    
    def test_feature_list_includes_base_features(self):
        """Test that v2 includes all v1 base features."""
        pipeline = Phase1RetrainingPipeline()
        
        base_features = [
            'departure_hour', 'day_of_week', 'month',
            'airline_id', 'aircraft_type', 'route_distance',
            'scheduled_duration', 'airport_congestion'
        ]
        
        for feature in base_features:
            self.assertIn(feature, pipeline.feature_names,
                         f"Missing v1 feature: {feature}")
    
    def test_model_artifact_naming(self):
        """Test model artifact follows expected naming."""
        pipeline = Phase1RetrainingPipeline()
        self.assertIn('v2_weather', pipeline.model_path)


class TestIntegration(unittest.TestCase):
    """End-to-end integration tests."""
    
    def test_weather_to_model_pipeline(self):
        """Test weather features flow into model."""
        # Get weather features
        service = RealTimeWeatherService(api_key=None)
        weather_features = service.get_weather_features(40.7128, -74.0060)
        
        # Should have all 12 weather features
        self.assertEqual(len(weather_features), 12)
        
        # Create a minimal feature set for model
        base_features = {
            'departure_hour': 14,
            'day_of_week': 3,
            'month': 3,
            'airline_id': 1,
            'aircraft_type': 1,
            'route_distance': 500,
            'scheduled_duration': 120,
            'airport_congestion': 0.6
        }
        
        # Combine
        all_features = {**base_features, **weather_features}
        self.assertEqual(len(all_features), 20)


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], verbosity=2, exit=False)
if __name__ == '__main__':
    run_tests()
