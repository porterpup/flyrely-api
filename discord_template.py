#!/usr/bin/env python3
"""
Discord formatting templates for FlyRely evaluation results
"""

from datetime import datetime


def format_evaluation_results(metrics, metrics_baseline=None):
    """
    Format evaluation results as Discord message
    
    Args:
        metrics: Metrics dict from weekly_evaluation.py
        metrics_baseline: Optional baseline metrics for comparison
    
    Returns:
        Formatted Discord markdown string
    """
    m = metrics["overall"]
    tp, tn, fp, fn = m["tp"], m["tn"], m["fp"], m["fn"]
    total = tp + tn + fp + fn
    
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    msg = f"""
**FlyRely Weekly Evaluation** 📊

**Performance:**
- Accuracy:  `{accuracy:.1%}`
- Precision: `{precision:.1%}`
- Recall:    `{recall:.1%}`
- F1 Score:  `{f1:.3f}`

**Confusion Matrix:**
- TP: {tp} | TN: {tn}
- FP: {fp} | FN: {fn}
- Total: {total}
Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
"""
    return msg.strip()


def format_quick_snapshot(metrics):
    """One-liner status update"""
    m = metrics["overall"]
    tp, tn, fp, fn = m["tp"], m["tn"], m["fp"], m["fn"]
    total = tp + tn + fp + fn
    
    if total == 0:
        return "No predictions logged yet"
    
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return f"✈️ FlyRely: {total} predictions | Acc {accuracy:.0%} | Prec {precision:.0%} | Rec {recall:.0%}"


def format_alerts_only(metrics, thresholds=None):
    """Post only if metrics are below thresholds"""
    if thresholds is None:
        thresholds = {"precision": 0.35, "recall": 0.55}
    
    m = metrics["overall"]
    tp, tn, fp, fn = m["tp"], m["tn"], m["fp"], m["fn"]
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    alerts = []
    
    if precision < thresholds["precision"]:
        alerts.append(f"⚠️ Precision LOW: {precision:.1%} (threshold: {thresholds['precision']:.0%})")
    
    if recall < thresholds["recall"]:
        alerts.append(f"⚠️ Recall LOW: {recall:.1%} (threshold: {thresholds['recall']:.0%})")
    
    if not alerts:
        return None  # All good
    
    return "**FlyRely Alerts** 🚨\n" + "\n".join(alerts)
if __name__ == "__main__":
    # Example usage
    mock_metrics = {
        "overall": {
            "tp": 68, "tn": 1108, "fp": 594, "fn": 216,
            "predictions": []
        },
        "by_airline": {},
        "by_route": {},
        "by_date": {}
    }
    
    print("Full Report:")
    print(format_evaluation_results(mock_metrics))
    print("\nQuick Snapshot:")
    print(format_quick_snapshot(mock_metrics))
    print("\nAlerts (if any):")
    alerts = format_alerts_only(mock_metrics)
    print(alerts if alerts else "✅ All metrics healthy")
