#!/usr/bin/env python3
"""
FlyRely Flight Delay Prediction API
Updated with prediction logging to CSV
"""

import os
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("Warning: FastAPI not installed. Install with: pip install fastapi uvicorn")
    FastAPI = None


# ============================================================================
# Configuration
# ============================================================================

# Paths
REPO_ROOT = Path(__file__).parent.parent
PREDICTIONS_CSV = REPO_ROOT / "predictions.csv"
MODEL_VERSION = "v2.0.0"

# CSV Columns
CSV_HEADERS = [
    "timestamp",
    "flight_id",
    "origin",
    "destination",
    "departure_time",
    "prediction",
    "probability",
    "model_version"
]


# ============================================================================
# Pydantic Models
# ============================================================================

class FlightPredictionRequest(BaseModel):
    """Request schema for prediction endpoint"""
    flight_id: str
    origin: str
    destination: str
    departure_time: str  # ISO 8601 format expected
    airline: Optional[str] = None
    aircraft: Optional[str] = None


class FlightPredictionResponse(BaseModel):
    """Response schema for prediction endpoint"""
    flight_id: str
    origin: str
    destination: str
    prediction: str  # "on-time" or "delayed"
    probability: float  # Confidence score 0.0-1.0
    model_version: str
# ============================================================================
# CSV Logging
# ============================================================================

def ensure_csv_header():
    """Create CSV file with header if it doesn't exist"""
    if not PREDICTIONS_CSV.exists():
        with open(PREDICTIONS_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()
        print(f"Created predictions log: {PREDICTIONS_CSV}")


def log_prediction(
    flight_id: str,
    origin: str,
    destination: str,
    departure_time: str,
    prediction: str,
    probability: float
):
    """
    Log a prediction to the CSV file
    
    Args:
        flight_id: Flight identifier
        origin: IATA code for origin airport
        destination: IATA code for destination airport
        departure_time: ISO 8601 formatted departure time
        prediction: "on-time" or "delayed"
        probability: Confidence score (0.0-1.0)
    """
    ensure_csv_header()
    
    row = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "flight_id": flight_id,
        "origin": origin,
        "destination": destination,
        "departure_time": departure_time,
        "prediction": prediction,
        "probability": round(probability, 4),
        "model_version": MODEL_VERSION
    }
    
    with open(PREDICTIONS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writerow(row)


# ============================================================================
# Mock Prediction Model
# ============================================================================

def predict_delay(
    origin: str,
    destination: str,
    departure_time: str,
    airline: Optional[str] = None,
    aircraft: Optional[str] = None
) -> Tuple[str, float]:
    """
    Mock prediction function. Replace with actual ML model.
    
    Returns:
        Tuple of (prediction, probability)
        prediction: "on-time" or "delayed"
        probability: Confidence score 0.0-1.0
    """
    import hashlib
    import datetime as dt
    
    route = f"{origin}-{destination}"
    try:
        dep_dt = dt.datetime.fromisoformat(departure_time.replace("Z", "+00:00"))
        hour = dep_dt.hour
    except:
        hour = 12
    
    seed = hash(f"{route}-{hour}") % 100
    
    if seed > 60:
        return "delayed", 0.65 + (seed % 30) / 100
    else:
        return "on-time", 0.70 + (seed % 25) / 100


# ============================================================================
# FastAPI App
# ============================================================================
if FastAPI:
    app = FastAPI(
        title="FlyRely Prediction API",
        description="Flight delay prediction service with monitoring",
        version="2.0.0"
    )

    @app.get("/health")
    def health_check():
        """Health check endpoint"""
        return {"status": "ok", "service": "FlyRely"}

    @app.post("/predict", response_model=FlightPredictionResponse)
    def predict(request: FlightPredictionRequest):
        """Predict flight delay"""
        try:
            prediction, probability = predict_delay(
                origin=request.origin,
                destination=request.destination,
                departure_time=request.departure_time,
                airline=request.airline,
                aircraft=request.aircraft
            )
            
            log_prediction(
                flight_id=request.flight_id,
                origin=request.origin,
                destination=request.destination,
                departure_time=request.departure_time,
                prediction=prediction,
                probability=probability
            )
            
            return FlightPredictionResponse(
                flight_id=request.flight_id,
                origin=request.origin,
                destination=request.destination,
                prediction=prediction,
                probability=probability,
                model_version=MODEL_VERSION
            )
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    @app.get("/predictions/stats")
    def get_stats():
        """Get statistics on predictions logged so far"""
        if not PREDICTIONS_CSV.exists():
            return {"total": 0, "on_time": 0, "delayed": 0}
        
        on_time_count = 0
        delayed_count = 0
        
        with open(PREDICTIONS_CSV, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row and row.get("prediction") == "on-time":
                    on_time_count += 1
                elif row and row.get("prediction") == "delayed":
                    delayed_count += 1
        
        total = on_time_count + delayed_count
        return {
            "total": total,
            "on_time": on_time_count,
            "delayed": delayed_count,
            "on_time_pct": round(on_time_count / total * 100, 1) if total > 0 else 0
        }


# ============================================================================
# CLI Entry Point
# ============================================================================
if __name__ == "__main__":
    if FastAPI:
        ensure_csv_header()
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
    else:
        print("Error: FastAPI not installed")
        print("Install with: pip install fastapi uvicorn")
        exit(1)
