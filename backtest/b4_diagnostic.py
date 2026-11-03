"""
B4 diagnostic: re-score the committed 18-feature model on all-carrier 2026-01..08 with the
operating carrier mapped correctly, and compare the six carriers the old Reporting Carrier
data omitted (9E, PT, QX, G7, C5, YV) against the rest.

    python backtest/b4_diagnostic.py     [out_filename]   # writes backtest/results/<out_filename> (default b4_allcarrier_diagnostic.json)

Diagnostic only: nothing is retrained and models/ is untouched.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest as bt  # noqa: E402

MISSING = ["9E", "PT", "QX", "G7", "C5", "YV"]


def metrics(df, p):
    flown = ~df["cancelled"].to_numpy()
    y, pf = df["delayed"].to_numpy()[flown], np.asarray(p)[flown]
    ece, _ = bt.ece_and_deciles(y, pf)
    return {"n_flown": int(flown.sum()), "observed_delay_rate": round(float(y.mean()), 4),
            "mean_predicted": round(float(pf.mean()), 4),
            "auc": round(float(roc_auc_score(y, pf)), 4),
            "brier": round(float(brier_score_loss(y, pf)), 4), "ece": round(float(ece), 4)}


def main():
    hold = bt.load_months("2026-01", "2026-08")
    feats = bt.committed_features(hold)
    p = bt.main.model_service.model.predict_proba(feats)[:, 1]
    is_missing = hold["Reporting_Airline"].isin(MISSING).to_numpy()
    thirty = (hold["Origin"].isin(bt.main.AIRPORT_DELAY_RATES) & hold["Dest"].isin(bt.main.AIRPORT_DELAY_RATES)).to_numpy()
    out = {
        "note": "committed 18-feature model, raw predict_proba, airline = operating carrier (Reporting_Airline); data = parquet_v2",
        "period": "2026-01..2026-08", "n_flights": int(len(hold)),
        "share_previously_missing": round(float(is_missing.mean()), 4),
        "parity_mismatches_of_2000": bt.check_parity(hold, feats),
        "overall": metrics(hold, p),
        "previously_missing_carriers": metrics(hold[is_missing], p[is_missing]),
        "other_carriers": metrics(hold[~is_missing], p[~is_missing]),
        "thirty_airport_routes": metrics(hold[thirty], p[thirty]),
        "by_carrier": {c: metrics(hold[hold["Reporting_Airline"] == c], p[(hold["Reporting_Airline"] == c).to_numpy()])
                       for c in sorted(hold["Reporting_Airline"].unique())},
    }
    name = sys.argv[1] if len(sys.argv) > 1 else "b4_allcarrier_diagnostic.json"  # registered file is never the default target of re-runs
    path = bt.RESULTS / name
    path.write_text(json.dumps(out, indent=2))
    for k in ("overall", "thirty_airport_routes", "previously_missing_carriers", "other_carriers"):
        print(k, out[k])
    print("parity mismatches:", out["parity_mismatches_of_2000"])


if __name__ == "__main__":
    main()
