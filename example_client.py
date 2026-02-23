#!/usr/bin/env python3
"""
FlyRely API Client Example
==========================
Demonstrates how to use the FlyRely prediction API.

Usage:
    pip install httpx
    python example_client.py
"""

import httpx
from datetime import datetime, timedelta

API_URL = "http://localhost:8000"


def predict_flight(origin: str, destination: str, departure_time: datetime, airline: str = None):
    """Make a flight delay prediction."""
    payload = {
        "origin": origin,
        "destination": destination,
        "departure_time": departure_time.isoformat(),
    }
    if airline:
        payload["airline"] = airline

    response = httpx.post(f"{API_URL}/predict", json=payload, timeout=10)
    response.raise_for_status()
    return response.json()


def print_prediction(result: dict):
    """Pretty print a prediction result."""
    risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}

    print("\n" + "=" * 60)
    print(f"Flight: {result['origin']} → {result['destination']}")
    print(f"Departure: {result['departure_time']}")
    if result.get('airline'):
        print(f"Airline: {result['airline']}")
    print("=" * 60)

    # Risk level
    emoji = risk_emoji.get(result['risk_level'], "⚪")
    print(f"\n{emoji} Risk Level: {result['risk_level'].upper()}")
    print(f"   Delay Probability: {result['delay_probability']*100:.0f}%")
    print(f"   Confidence: {result['confidence']*100:.0f}%")

    # Weather
    if result.get('origin_weather'):
        w = result['origin_weather']
        print(f"\n🌤️  Origin Weather ({result['origin']}):")
        print(f"   {w['temperature_f']:.0f}°F, {w['wind_speed_mph']:.0f} mph wind, {w['visibility_miles']:.1f} mi visibility")
        print(f"   Conditions: {w['conditions']}")

    if result.get('destination_weather'):
        w = result['destination_weather']
        print(f"\n🌤️  Destination Weather ({result['destination']}):")
        print(f"   {w['temperature_f']:.0f}°F, {w['wind_speed_mph']:.0f} mph wind, {w['visibility_miles']:.1f} mi visibility")
        print(f"   Conditions: {w['conditions']}")

    # Risk factors
    if result.get('risk_factors'):
        print("\n⚠️  Risk Factors:")
        for factor in result['risk_factors']:
            print(f"   • {factor}")

    # Recommendations
    if result.get('recommendations'):
        print("\n💡 Recommendations:")
        for rec in result['recommendations']:
            print(f"   • {rec}")

    print()


def main():
    print("FlyRely API Client Example")
    print("=" * 60)

    # Check API health
    try:
        health = httpx.get(f"{API_URL}/health", timeout=5).json()
        print(f"✓ API Status: {health['status']}")
        print(f"  Model loaded: {health['model_loaded']}")
        print(f"  Weather API: {'configured' if health['weather_api_configured'] else 'not configured'}")
    except Exception as e:
        print(f"✗ Cannot connect to API at {API_URL}")
        print(f"  Error: {e}")
        print("\nMake sure the API is running: uvicorn main:app --reload")
        return

    # Example predictions
    examples = [
        # Morning flight, good conditions expected
        {
            "origin": "SFO",
            "destination": "LAX",
            "departure_time": datetime.now() + timedelta(days=7, hours=8),
            "airline": "DL"
        },
        # Evening JFK flight (historically more delays)
        {
            "origin": "JFK",
            "destination": "MIA",
            "departure_time": datetime.now() + timedelta(days=7, hours=18),
            "airline": "AA"
        },
        # Chicago in winter (potential weather)
        {
            "origin": "ORD",
            "destination": "DEN",
            "departure_time": datetime.now() + timedelta(days=14, hours=14),
            "airline": "UA"
        },
    ]

    print("\n" + "-" * 60)
    print("Running example predictions...")

    for ex in examples:
        try:
            result = predict_flight(**ex)
            print_prediction(result)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
