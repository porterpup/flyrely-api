# Run-up backtest: results

Protocol: `PROTOCOL_RUNUP.md` (commit `4a5ae73`), code at `2a5b164`. Holdout
2026-01 → 2026-08, 4.73M flights, scored once on 2026-10-29. Raw output:
`results/runup.json`.

## What this does and doesn't show

The comparison is against **Flighty proxies built from BTS data**, not Flighty's
own predictions. Those aren't available as data. Real Flighty also sees
airline-posted delays and FAA programs, so it is likely stronger than the proxy,
especially 3 h and 1 h out. **This result does not show FlyRely beats Flighty.**
It shows FlyRely beats what a flight-history plus "where's my plane" tracker
can do. The live head-to-head (`headtohead/h2h.py`) is the real test.

## Results (holdout, target: departure delay ≥ 15 min)

| Horizon | FlyRely AUC | Proxy AUC | FlyRely Brier | Proxy Brier | Brier diff (95% CI) | FlyRely ECE | Verdict |
|---|---|---|---|---|---|---|---|
| Booking (~T−30d) | 0.682 | 0.597 | 0.1711 | 0.1790 | −0.0079 [−0.0092, −0.0066] | 0.045 | Better than proxy, **miscalibrated** |
| T−7d | 0.693 | 0.629 | 0.1676 | 0.1757 | −0.0081 [−0.0093, −0.0069] | 0.028 ✗ | **Fails** (calibration) |
| T−1d | 0.702 | 0.643 | 0.1655 | 0.1740 | −0.0085 [−0.0099, −0.0072] | 0.0200 | Passes, but ECE is exactly at the bar |
| T−3h | 0.821 | 0.750 | 0.1245 | 0.1406 | −0.0162 [−0.0173, −0.0152] | 0.004 ✓ | **Passes** all four |
| T−1h | 0.853 | 0.787 | 0.1085 | 0.1243 | −0.0158 [−0.0168, −0.0148] | 0.006 ✓ | **Passes** all four |

- AUC rises at every step toward departure, as it should (the leakage check
  passes).
- Every horizon beats the booking-time model on Brier.
- Weather (archived forecasts, 54 of 100 airports fetched) lowered validation
  Brier at every horizon, so all four horizon models use it.
- 3 h and 1 h are steady month to month (AUC 0.80–0.86). 7 d and 1 d dip to
  0.64 in January.
- On "disrupted" (delayed, cancelled or diverted), AUC is 0.68 / 0.70 / 0.83 /
  0.86 at 7 d / 1 d / 3 h / 1 h.

## Reading it

- **Close to departure (3 h, 1 h):** knowing where the aircraft is, plus how the
  origin, destination and airline are running that day, gives a large and
  well-calibrated gain over a plane-tracker rule. This is the strongest result
  and the most direct contest with Flighty.
- **Days out (7 d, 1 d):** a real but smaller gain. Calibration suffers from the
  same 2025→2026 level shift as the booking model.
- **Booking time:** ranks flights much better than on-time history (AUC 0.68 vs
  0.60). But it under-predicts 2026 by 3–5 points, so the probabilities it
  shows would be too low.

## Next steps toward accuracy

1. Fix calibration drift at booking, 7 d and 1 d: rolling recalibration and
   longer trailing-rate windows. Pre-register it and test on 2026-09+ when BTS
   publishes it.
2. Finish the weather fetch for all 100 airports.
3. Put the 3 h and 1 h models behind the API. They need a live feed of same-day
   departures and aircraft assignments (AeroAPI or OpenSky).
4. Start logging real flights in `headtohead/h2h.py`. Only that answers "more
   accurate than Flighty".
