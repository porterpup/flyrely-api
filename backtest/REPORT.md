# Booking-time backtest: results

Protocol: `PROTOCOL.md` (commit `f6678e2`). Holdout 2026-01 → 2026-08, 4.73M flights,
scored once on 2026-10-29. Raw output: `results/results.json`, `results/calibration.png`.

## Verdict: ITERATE

Neither booking-time model clears the bar. The retrained model meets the ITERATE
condition (AUC ≥ 0.66 and criteria 3–4 pass).

| Criterion | Bar | Committed (production) | Booking (retrained) | Day-of reference |
|---|---|---|---|---|
| 1. ROC-AUC, delayed | ≥ 0.70 | 0.677 ✗ | 0.682 ✗ | 0.820 ✓ |
| 2. ECE / worst decile gap | ≤ 0.02 / ≤ 0.03 | 0.007 / 0.023 ✓ | 0.045 / 0.072 ✗ | 0.023 / 0.050 ✗ |
| 3. High ≥ 2× Low, ordered every month | | 2.88× ✓ | 2.78× ✓ | 6.11× ✓ |
| 4. Riskiest 20% hold ≥ 35% of severe delays | | 37.8% ✓ | 38.9% ✓ | 65.2% ✓ |
| 5. Riskiest 20% hold ≥ 30% of cancellations | | 27.0% ✗ | 27.5% ✗ | 46.5% ✓ |
| 6. Monthly AUC never < 0.65 | | 0.620 ✗ | 0.618 ✗ | 0.795 ✓ |

Holdout base rates: 24.0% delayed ≥ 15 min, 3.6% delayed ≥ 2 h, 2.2% cancelled.

## What went wrong

- **2026 was a worse year than the training data.** The retrained model predicts
  too low in every decile, by 3–5 points, because its historical rates come from
  2025. Biggest misses: Southwest (31% observed vs 20% predicted), Spirit,
  JetBlue, American, and March and July. That is a level shift, not a ranking
  problem.
- **Ranking is weakest in winter.** Monthly AUC climbs from 0.62 in January to 0.70
  from June on. Winter delays are weather-driven and can't be seen at booking.
- **The production model's good calibration is partly luck.** Its hard-coded
  airline and airport rates happen to sit near 2026 levels. It ranks slightly worse.

## Deviation from the protocol

The BTS Reporting Carrier files have no marketing-airline column, so the
production model's airline lookup used the operating airline. Regional flights
(SkyWest, Endeavor, Envoy, Republic, PSA) fell back to the default 0.21 rate.
That affects one of 18 features for about a quarter of flights.

## Next step

The protocol's ITERATE step was a weather experiment at booking lead time.
Forecasts don't reach weeks ahead, so that is unlikely to help. The two fixes
the evidence points to:

1. **Recent-performance features at booking time** (trailing 30/90-day airline,
   airport and route rates), so level shifts come through.
2. **Rolling recalibration** on the most recent months.

These need a new protocol commit and a fresh holdout. 2026-09 onward isn't
published yet. See `REPORT_RUNUP.md` for the horizons closer to departure.
