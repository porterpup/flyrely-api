# FlyRely run-up backtest protocol (pre-registered)

Written and committed **before** any run-up result was computed. Companion to
`PROTOCOL.md` (booking time). The pass/fail bar below must not be changed after
seeing results. A new protocol needs a new commit and a fresh holdout.

## Question

Between booking and departure, can FlyRely give a delay probability that is
**more accurate than what a Flighty-style tracker shows at the same moment**,
and does that accuracy improve as departure gets closer?

## Data

Same BTS On-Time Reporting data, months and strict time split as `PROTOCOL.md`:

| Set | Rows | Used for |
|---|---|---|
| Fit | 2025-01 → 2025-06 | Fitting models |
| Validation | 2025-07 → 2025-12 | Model selection, calibration, weather go/no-go |
| **Holdout** | **2026-01 → 2026-08** | **Reported once, at the end** |

Long-run historical rates come from the same trailing windows as `PROTOCOL.md`.
The booking-time holdout result (`PROTOCOL.md`) will be seen before this one.
The booking features are reused unchanged, and nothing in this protocol is tuned
on holdout data.

Weather: Open-Meteo *Previous Runs* archive, i.e. forecasts as they were issued
1 or 7 days earlier, never observed weather. Free, no key.

## Horizons and what is known at each

The cutoff is `c = scheduled departure − h`. All times are converted to one
clock using each airport's UTC offset, derived from BTS itself
(`CRSArrTime − CRSDepTime − CRSElapsedTime` per route, per month).

| Horizon | Features (each row adds to the one above) |
|---|---|
| Booking (`PROTOCOL.md` model 2) | Schedule, calendar, long-run airport/airline/route/hour rates |
| **T−7d** | + recent performance: trailing 7-day delay and cancel rates for origin, dest, airline, route, flight number, ending the day before the cutoff date |
| **T−1d** | same, with the window ending the day before the cutoff date |
| **T−3h** | + same-day network state at `c`: departures so far, share delayed, mean delay, flights scheduled but not yet gone, known cancellations, at origin, at dest and for the airline system-wide; + aircraft rotation: latest leg of the same tail with scheduled departure ≤ `c`, its delay as known at `c` (actual if it has left, else a lower bound), and the remaining scheduled slack before this flight |
| **T−1h** | same as T−3h, at the later cutoff |
| Weather variant (W) | each horizon + forecast precipitation, snowfall, wind gusts, CAPE and weather code at origin and dest for the scheduled hour; `previous_day7` run for T−7d, `previous_day1` run for T−1d, T−3h and T−1h |

A delay is "known at `c`" only if it is derivable from scheduled times and
departures that happened at or before `c`. The flight being predicted is never
part of its own features.

## Targets

`delayed` (DepDelay ≥ 15, cancellations excluded) is primary. `disrupted`
(delayed, cancelled or diverted) is reported alongside.

## Flighty proxies (the baselines to beat)

Flighty's own predictions aren't available as data, so each horizon has a
documented proxy for what a Flighty user sees at that moment:

- **Booking, T−7d, T−1d — flight history.** The flight number's on-time record
  over the trailing 90 days (what Flighty shows as "on-time %"), smoothed
  toward the airline's rate with 10 pseudo-flights and calibrated on
  validation.
- **T−3h, T−1h — "where's my plane".** The inbound aircraft's known delay
  at `c` minus the scheduled slack in the turn, i.e. the delay a tracker
  would project for this flight, plus that flight-history rate, combined by a
  logistic regression fit on fit data and calibrated on validation.

Real Flighty also sees airline-posted delays and FAA programs, which BTS
doesn't record. The proxies are therefore probably weaker than Flighty close to
departure. The live head-to-head (`headtohead/`) settles that, not this test.

## Models

One `HistGradientBoostingClassifier` per horizon (same settings and seed as
`PROTOCOL.md`), isotonic-calibrated on validation. Variant W is the same
model with weather features. W is adopted only if it lowers validation Brier
at a horizon. Otherwise that horizon reports the no-weather model. The
choice is made on validation and recorded before the holdout is scored.

## Pass / fail bar (holdout, per horizon T−7d, T−1d, T−3h, T−1h)

FlyRely **beats Flighty** at a horizon only if **all** hold:

1. **Better than the proxy:** Brier score lower than the Flighty proxy for that
   horizon, with the 95% day-block bootstrap interval of the difference
   (1,000 resamples of holdout days, seed 42) entirely below zero.
2. **Ranks better:** ROC-AUC for `delayed` higher than the proxy.
3. **Calibrated:** ECE (10 equal-count bins) ≤ **0.02**.
4. **Adds value over booking:** Brier lower than the booking-time model's.

Across horizons, AUC must not fall as departure approaches
(7d ≤ 1d ≤ 3h ≤ 1h, tolerance 0.002). Otherwise the report flags a leak or bug
before it says anything else.

The booking-time comparison against the flight-history proxy (criteria 1–2
only) is reported too. That is the "at time of purchase" claim.

## Outcomes

- **Beats proxy at all four horizons and at booking:** build the run-up models
  into the API and start the live head-to-head with beta users.
- **Beats proxy at some horizons:** ship those horizons only. The report
  names what is missing at the others.
- **Beats proxy nowhere:** say so plainly. Don't tune until it passes.

## Production note

The T−3h and T−1h features need live same-day departures and aircraft
assignments (AeroAPI or OpenSky), and T−7d/T−1d need recent actuals. Weather
needs a forecast feed. The report states each feature's live source.

## Reproducibility

`backtest/runup.py`, seed 42. Weather cache in `data/weather/` (gitignored).
Results go in `backtest/results/runup.json` and `backtest/REPORT_RUNUP.md`.
