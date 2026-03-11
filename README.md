# FlyRely API

Flight delay prediction API with real-time weather integration.

... (unchanged)

## Model thresholds (Calibration v2)

The API maps model output probabilities to risk levels using the following calibrated thresholds:
- Low: probability < 0.20
- Medium: 0.20 ≤ probability < 0.30
- High: probability ≥ 0.30

This matches the frontend `toRiskLevel` mapping.
