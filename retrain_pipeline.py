#!/usr/bin/env python3
"""
FlyRely Model Retraining Pipeline
Fetches fresh flight data and retrains the ML model
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import subprocess

def run_opensky_fetch():
    """Fetch fresh data from OpenSky API"""
    print("\n" + "="*60)
    print("STEP 1: Fetching fresh flight data from OpenSky")
    print("="*60)
    
    result = subprocess.run(['python3', 'opensky_fetcher.py'], cwd='data')
    return result.returncode == 0


def load_training_data():
    """Load OpenSky data + any existing training data"""
    print("\n" + "="*60)
    print("STEP 2: Loading training data")
    print("="*60)
    
    data_dir = Path('data')
    dfs = []
    
    # Load OpenSky recent data
    opensky_file = data_dir / 'opensky_flights_recent.csv'
    if opensky_file.exists():
        df_opensky = pd.read_csv(opensky_file)
        print(f"Loaded {len(df_opensky)} flights from OpenSky")
        dfs.append(df_opensky)
    
    # Load any historical BTS data
    bts_file = data_dir / 'bts_flights_2024_2025.csv'
    if bts_file.exists():
        df_bts = pd.read_csv(bts_file)
        print(f"Loaded {len(df_bts)} flights from BTS historical data")
        dfs.append(df_bts)
    
    if not dfs:
        print("ERROR: No training data found!")
        return None
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total training samples: {len(df)}")
    return df


def prepare_features(df):
    """Extract features from flight data"""
    print("\n" + "="*60)
    print("STEP 3: Preparing features")
    print("="*60)
    
    features = []
    
    # Extract temporal features
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
    else:
        df['hour'] = 12
        df['day_of_week'] = 0
        df['month'] = 3
    
    # Airline encoding
    airline_map = {'United': 0, 'American': 1, 'Delta': 2}
    df['airline_id'] = df.get('airline', 'United').map(airline_map).fillna(0)
    
    # Build feature matrix
    feature_cols = [
        'hour', 'day_of_week', 'month', 'airline_id'
    ]
    
    X = df[feature_cols].fillna(0).values
    y = df['is_delayed'].values
    
    print(f"Features: {feature_cols}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y.astype(int))}")
    
    return X, y, feature_cols


def train_model(X, y):
    """Train GradientBoosting model"""
    print("\n" + "="*60)
    print("STEP 4: Training model")
    print("="*60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {len(X_train)}")
    print(f"Test set: {len(X_test)}")
    
    # Train model
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    print("Training GradientBoostingClassifier...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print("\n=== Model Metrics ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc_roc': float(auc),
        'trained_at': datetime.utcnow().isoformat(),
        'samples': len(X_train)
    }
    
    return model, metrics


def save_model(model, metrics, feature_cols):
    """Save trained model and metadata"""
    print("\n" + "="*60)
    print("STEP 5: Saving model")
    print("="*60)
    
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # Save model
    model_file = models_dir / 'flight_delay_model.joblib'
    joblib.dump(model, model_file)
    print(f"Saved model to {model_file}")
    
    # Save feature names
    features_file = models_dir / 'feature_names.joblib'
    joblib.dump(feature_cols, features_file)
    print(f"Saved feature names to {features_file}")
    
    # Save metrics
    metrics_file = models_dir / 'model_metadata.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_file}")
    
    return True


def main():
    print("\n" + "="*70)
    print("FlyRely Model Retraining Pipeline")
    print("="*70)
    
    # Step 1: Fetch fresh data
    if not run_opensky_fetch():
        print("WARNING: OpenSky fetch failed (may need valid credentials)")
    
    # Step 2: Load data
    df = load_training_data()
    if df is None:
        return False
    
    # Step 3: Prepare features
    X, y, feature_cols = prepare_features(df)
    
    # Step 4: Train model
    model, metrics = train_model(X, y)
    
    # Step 5: Save model
    save_model(model, metrics, feature_cols)
    
    print("\n" + "="*70)
    print("✅ Retraining complete!")
    print("="*70)
    print("\nNext steps:")
    print("1. Commit model changes to GitHub")
    print("2. Railway auto-deploys on push")
    print("3. Verify with: curl https://your-api/health")
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
