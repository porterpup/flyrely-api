# FlyRely API

Flight delay prediction API with real-time weather integration.

... (unchanged)

## Model thresholds

The API maps model output probabilities to risk levels using the following calibrated thresholds:
- Low: probability < 0.15
- Medium: 0.15 ≤ probability < 0.25
- High: probability ≥ 0.25

This matches the frontend `toRiskLevel` mapping.
