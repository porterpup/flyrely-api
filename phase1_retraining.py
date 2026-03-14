"""
Phase 1 Model Retraining Pipeline with Weather Features
========================================================

Full retraining pipeline for flight delay prediction model v2.
Incorporates 12 weather features for improved precision and recall.

Features:
- Temperature, wind speed, visibility
- Precipitation, thunderstorm indicators
- Humidity, pressure, dew point
- Random Forest model with class weighting
- Backward compatible with v1 API
"""

import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class Phase1RetrainingPipeline:
"""Full retraining pipeline with weather feature integration."""

def __init__(self, model_path='flight_delay_model_v2_weather.joblib'):
self.model_path = model_path
self.model = None
self.scaler = StandardScaler()
self.feature_names = self._get_feature_names()
self.weather_features = [
'temperature', 'wind_speed', 'visibility',
'precipitation', 'thunderstorm_probability',
'humidity', 'pressure', 'dew_point',
'wind_gust', 'cloud_coverage', 'ceiling', 'weather_severity'
]

def _get_feature_names(self):
"""Define all feature names for model."""
base_features = [
'departure_hour', 'day_of_week', 'month',
'airline_id', 'aircraft_type', 'route_distance',
'scheduled_duration', 'airport_congestion'
]
return base_features + self.weather_features

def load_training_data(self, filepath):
"""Load and validate training data."""
logger.info(f"Loading training data from {filepath}")
df = pd.read_csv(filepath)
# Validate required columns
missing = set(self.feature_names) - set(df.columns)
if missing:
raise ValueError(f"Missing columns: {missing}")

# Separate features and target
X = df[self.feature_names]
y = df['is_delayed'] # Binary: 0=on-time, 1=delayed

logger.info(f"Loaded {len(df)} samples, {len(self.feature_names)} features")
return X, y

def train(self, X, y, test_size=0.2, random_state=42):
"""
Train RandomForest with class weighting for imbalanced data.

Args:
X: Feature matrix
y: Target vector
test_size: Validation split ratio
random_state: Reproducibility seed

Returns:
dict: Training metrics
"""
logger.info("Starting model training...")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=test_size, random_state=random_state, stratify=y
)

# Scale features
X_train_scaled = self.scaler.fit_transform(X_train)
X_test_scaled = self.scaler.transform(X_test)

# Train with class weighting for imbalanced dataset
self.model = RandomForestClassifier(
n_estimators=200,
max_depth=20,
min_samples_split=10,
min_samples_leaf=5,
class_weight='balanced',
random_state=random_state,
n_jobs=-1,
verbose=1
)

self.model.fit(X_train_scaled, y_train)
logger.info("Training complete")

# Evaluate
y_pred = self.model.predict(X_test_scaled)

metrics = {
'precision': precision_score(y_test, y_pred),
'recall': recall_score(y_test, y_pred),
'f1': f1_score(y_test, y_pred),
'train_samples': len(X_train),
'test_samples': len(X_test),
'timestamp': datetime.utcnow().isoformat()
}

logger.info(f"Precision: {metrics['precision']:.4f}")
logger.info(f"Recall: {metrics['recall']:.4f}")
logger.info(f"F1-Score: {metrics['f1']:.4f}")

return metrics

def save_model(self):
"""Save trained model and scaler."""
if self.model is None:
raise RuntimeError("No trained model. Call train() first.")
artifact = {
'model': self.model,
'scaler': self.scaler,
'feature_names': self.feature_names,
'weather_features': self.weather_features,
'version': 'v2_weather',
'timestamp': datetime.utcnow().isoformat()
}

joblib.dump(artifact, self.model_path)
logger.info(f"Model saved to {self.model_path}")

def load_model(self):
"""Load previously trained model."""
artifact = joblib.load(self.model_path)
self.model = artifact['model']
self.scaler = artifact['scaler']
self.feature_names = artifact['feature_names']
self.weather_features = artifact['weather_features']
logger.info(f"Model loaded from {self.model_path}")
return self.model

def predict_batch(self, X):
"""
Predict delays for batch of flights.

Args:
X: Feature matrix (pandas DataFrame or numpy array)

Returns:
numpy array: Predictions (0=on-time, 1=delayed)
"""
if self.model is None:
raise RuntimeError("No model loaded. Call load_model() first.")

if isinstance(X, pd.DataFrame):
X = X[self.feature_names].values

X_scaled = self.scaler.transform(X)
return self.model.predict(X_scaled)

def predict_proba_batch(self, X):
"""
Get delay probability for batch of flights.
Args:
X: Feature matrix

Returns:
numpy array: Probabilities [[on_time_prob, delay_prob], ...]
"""
if self.model is None:
raise RuntimeError("No model loaded. Call load_model() first.")

if isinstance(X, pd.DataFrame):
X = X[self.feature_names].values

X_scaled = self.scaler.transform(X)
return self.model.predict_proba(X_scaled)
def get_feature_importance(self):
"""Get feature importance rankings."""
if self.model is None:
raise RuntimeError("No model loaded.")

importances = self.model.feature_importances_
return pd.DataFrame({
'feature': self.feature_names,
'importance': importances
}).sort_values('importance', ascending=False)


def main():
"""Example retraining workflow."""
pipeline = Phase1RetrainingPipeline()
# Load training data
X, y = pipeline.load_training_data('training_data.csv')

# Train model
metrics = pipeline.train(X, y)

# Save model artifact
pipeline.save_model()

# Print feature importance
importance = pipeline.get_feature_importance()
print("\nTop 15 Features by Importance:")
print(importance.head(15))
return metrics


if __name__ == '__main__':
main()
