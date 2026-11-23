# Arrival re-test (Amendment 1): September 2026 results

**Status:** this was run as pre-registered in `PROTOCOL_ARRIVAL_CONNECTIONS.md` Amendment 1 (approved by Sean 2026-11-10 and 2026-11-22). There was one scoring attempt. Raw numbers are in `backtest/results/arrival_retest_2026-09.json`, and the attempt record is `arrival_retest_2026-09.attempts.json`.

**Outcome:** `ship_nothing_arrival`. A1, A2 and A4 fail; A3 passes. The C confirmation was not run (per D4), so the final connection outcome stays `no_connection_numbers_arrival_ladder_not_shipped`. **Per D6, there are no further attempts on 2026 data.** Any next arrival attempt needs a new protocol, evaluated on 2027-01 or later. Both API flags stay off.

## What was tested

- **Arm L-S** is the frozen arrival ladder (arm L) with a level-only logit offset `a_t` per threshold, slope 1, followed by rearrangement.
- **Fitting:** the offsets were fitted on 2026-03..08 (3,979,495 flown rows) and frozen before any September arrival file existed. Nothing else was fitted.
- **Offsets:** they range from +0.08 (t = -15) to +0.36 (t = 300); t = 15 is +0.265.
- **Evaluation:** 2026-09, with 624,486 flown arrivals.

## Gates (thresholds unchanged from the original protocol)

| Gate | Test | September result | Pass? |
|---|---|---|---|
| A1 | ECE at t=15 ≤ 0.02 | 0.0407 | ✗ |
| | Max decile gap at t=15 ≤ 0.04 | 0.0577 | ✗ |
| | Monthly ECE ≤ 0.04 | 0.0407 | ✗ |
| | ECE at t=45 ≤ 0.03 | 0.0309 | ✗ |
| | ECE at t=120 ≤ 0.03 | 0.0126 | ✓ |
| A2 | Brier below production departure baseline (CI < 0) | -0.00223 [-0.00808, +0.00360] | ✗ |
| | Brier below climatology (CI < 0) | -0.00648 [-0.01112, -0.00166] | ✓ |
| A3 | High tier ≥ 1.8× low tier, ordered every month | 2.68× (12.0% / 20.7% / 32.3%), ordered | ✓ |
| A4 | RPS skill > 0 (CI excludes 0) | 0.028 [0.011, 0.043] | ✓ |
| | q10–q90 coverage within 0.80 ± 0.04 | 0.850 | ✗ |

At t=15:
- AUC 0.648 and Brier 0.1693.
- Mean predicted late rate **26.8%** against an observed rate of **22.7%**.
- The production departure baseline used as arrival is AUC 0.658, Brier 0.1715 and ECE 0.066, with mean prediction 16.2%.

## What happened (diagnosis, post hoc and non-gating)

The re-test overcorrected:
- Jan–Aug failed by predicting too low: 19.5% predicted against 23.6% observed.
- The nudge was learned on March–August, which includes the late-running spring and summer months. It raised the level, as it was designed to.
- September 2026 ran markedly *less* late (22.7% at t=15), so L-S now predicts too high by about 4 points.

**The miss looks seasonal, not a fixed 2026 shift.** A single constant offset learned on one season cannot track that. This reading is drawn after seeing the result, so it is a hypothesis for any 2027 protocol, not a finding. A3 passing again confirms that the ranking is sound. The problem remains calibration.

## What this means for the product

- **Arrival percentages stay off,** and so do connection-miss percentages.
- **The time-only tight-connection warning** (live since app PR #33/#36, extended to the flight page in app PR #49) is what users see.
- **The departure prediction stays the headline** everywhere.
- **D5** (the October take-away check) has nothing to take away, so it does not run.
- **Next attempt:** per D6, any next arrival protocol is evaluated on 2027-01 or later. A design that handles seasonality (e.g. month-of-year level terms fitted on 2025, or a rolling recalibration with an explicit seasonal term) would need to be pre-registered first.

## Provenance

| Commit | What |
|---|---|
| `b717936` | Lock-move (PR #70). September becomes ordinary data. |
| `8495499` | PR #72 merged: Amendment 1 and `eval_arrival_retest.py`. This is the code freeze (`84954994`). |
| `ab5a1de` | Freeze file: offsets fitted on 2026-03..08, with model hashes. Committed before any September arrival file was built. |
| `1c9aa0f` | Check passed (11/11): 630,177 rows, 30 dates, 1,048,025 pairs. |
| `b59a1f0` | Score results: one attempt at code commit `1c9aa0f`. |

Model hashes match the amendment:
- ladder `eec7f6b9…`
- production `fa854d75…`
- dependence `0c513c96…`
- MCT `ea889882…`

FLYRELY_HOLDOUT_UNLOCK was never set. No September arrival outcome was computed or printed before the score.

## Disclosures

These are the ones listed in the amendment:
- **(a)** H1 scored departure models on September, and Sean was shown the September departure booking-model calibration (22.7% predicted vs 22.4% observed late departures).
- **(b)** The amendment exists because the Jan–Aug evaluation failed.
- **(c)** The weekly schedule refresh includes September schedules.
- **(d)** Weather gaps at 4 airports are irrelevant: the arrival path uses no weather.

One more for this report: the arrival ladder joblib was restored from the `flyrely-api-wt-arr` worktree into `models/`. Its sha256 matched the manifest before the freeze.
