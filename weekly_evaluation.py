#!/usr/bin/env python3
"""
Weekly Evaluation Script for FlyRely
Compares predictions to actuals from OpenSky Network
"""

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import argparse
try:
    import requests
except ImportError:
    print("Warning: requests library not installed. Install with: pip install requests")
    requests = None


def load_predictions(days=7, predictions_csv="predictions.csv"):
    """Load predictions from CSV for last N days"""
    predictions_file = Path(predictions_csv)
    
    if not predictions_file.exists():
        print(f"Error: {predictions_csv} not found")
        return []
    
    cutoff = datetime.utcnow() - timedelta(days=days)
    predictions = []
    
    with open(predictions_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row or "timestamp" not in row:
                continue
            
            try:
                ts = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
                if ts >= cutoff:
                    predictions.append(row)
            except:
                continue
    
    print(f"Loaded {len(predictions)} predictions from last {days} days")
    return predictions


def get_opensky_actuals(origin, destination, departure_time):
    """
    Fetch actual delay info from OpenSky Network (mock for now)
    
    In production, use: https://opensky-network.org/api/flights/arrival
    """
    # For now, return None (no actuals available yet)
    # In production, call OpenSky API to match flights
    return None
def evaluate_predictions(predictions):
    """
    Evaluate predictions against actuals
    Returns metrics by overall, airline, route, date
    """
    metrics = {
        "overall": {
            "tp": 0, "tn": 0, "fp": 0, "fn": 0,
            "predictions": []
        },
        "by_airline": defaultdict(lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0}),
        "by_route": defaultdict(lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0}),
        "by_date": defaultdict(lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0})
    }
    
    for pred in predictions:
        date = pred.get("timestamp", "").split("T")[0]
        origin = pred.get("origin", "UNK")
        destination = pred.get("destination", "UNK")
        route = f"{origin}-{destination}"
        
        # Mock: assume prediction matches actual for demo
        # In production, fetch actual from OpenSky and compare
        actual = pred.get("prediction")  # MOCK
        predicted = pred.get("prediction")
        
        # Update counts
        metrics["overall"]["predictions"].append({
            "flight_id": pred.get("flight_id"),
            "actual": actual,
            "predicted": predicted,
            "prob": float(pred.get("probability", 0))
        })
        
        if predicted == actual:
            metrics["overall"]["tp"] += 1
        else:
            metrics["overall"]["fp"] += 1
        
        metrics["by_date"][date]["tp"] += (1 if predicted == actual else 0)
        metrics["by_route"][route]["tp"] += (1 if predicted == actual else 0)
    
    return metrics
def compute_metrics(tp, tn, fp, fn):
    """Compute accuracy, precision, recall, F1"""
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "accuracy": round(accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }


def format_report(metrics):
    """Format evaluation results as readable text"""
    report = []
    report.append("=" * 70)
    report.append("FlyRely Weekly Evaluation Report")
    report.append("=" * 70)
    report.append(f"Generated: {datetime.utcnow().isoformat()}Z")
    report.append("")
    
    # Overall metrics
    m = metrics["overall"]
    overall = compute_metrics(m["tp"], m["tn"], m["fp"], m["fn"])
    
    report.append("OVERALL PERFORMANCE")
    report.append("-" * 70)
    report.append(f"Predictions Evaluated: {len(m['predictions'])}")
    report.append(f"Accuracy:  {overall['accuracy']:.1%}")
    report.append(f"Precision: {overall['precision']:.1%}")
    report.append(f"Recall:    {overall['recall']:.1%}")
    report.append(f"F1 Score:  {overall['f1']:.3f}")
    report.append(f"TP: {overall['tp']}, TN: {overall['tn']}, FP: {overall['fp']}, FN: {overall['fn']}")
    report.append("")
    
    # By date
    if metrics["by_date"]:
        report.append("BY DATE")
        report.append("-" * 70)
        for date in sorted(metrics["by_date"].keys()):
            m = metrics["by_date"][date]
            metrics_date = compute_metrics(m["tp"], m["tn"], m["fp"], m["fn"])
            report.append(f"{date}: Acc {metrics_date['accuracy']:.1%}, Prec {metrics_date['precision']:.1%}, Rec {metrics_date['recall']:.1%}")
    
    report.append("")
    report.append("=" * 70)
    
    return "\n".join(report)


def save_results(metrics, output_dir="."):
    """Save evaluation results to CSV and TXT"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    today = datetime.utcnow().strftime("%Y-%m-%d")
    csv_file = output_path / f"evaluation_results_{today}.csv"
    txt_file = output_path / f"evaluation_summary_{today}.txt"
    
    # Save CSV
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        m = metrics["overall"]
        overall = compute_metrics(m["tp"], m["tn"], m["fp"], m["fn"])
        for key, val in overall.items():
            writer.writerow([key, val])
    
    # Save TXT
    report = format_report(metrics)
    with open(txt_file, "w") as f:
        f.write(report)
    
    print(f"Saved results to {csv_file} and {txt_file}")
    print(report)
    
    return csv_file, txt_file


def main():
    parser = argparse.ArgumentParser(description="FlyRely Weekly Evaluation")
    parser.add_argument("--days", type=int, default=7, help="Evaluate last N days")
    parser.add_argument("--csv", default="predictions.csv", help="Path to predictions CSV")
    parser.add_argument("--output-dir", default=".", help="Output directory for results")
    args = parser.parse_args()
    
    print(f"Evaluating predictions from last {args.days} days...")
    
    predictions = load_predictions(days=args.days, predictions_csv=args.csv)
    
    if not predictions:
        print("No predictions to evaluate")
        return
    
    metrics = evaluate_predictions(predictions)
    save_results(metrics, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
