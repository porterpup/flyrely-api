# FlyRely backtest protocol (pre-registered)

Written and committed **before** any holdout result was computed. The pass/fail
bar below must not be changed after seeing results. A new protocol needs a new
commit and a fresh holdout.

## Question

Can a prediction made at **booking time** (schedule information only) separate
risky flights from safe ones well enough to justify FlyRely's thesis that
pre-booking insight is worth paying for?

## Data

- BTS On-Time Reporting (Carrier On-Time Performance), free public data,
  Jan 2024 – Aug 2026 (latest month published as of 2026-10-29).
- Scope: all reporting US carriers, all airports. No AeroAPI, no Tomorrow.io,
  no paid data.

## Strict time split (no random split)

| Set | Months | Used for |
|---|---|---|
| Train | 2024-01 → 2025-06 | Fitting models; computing all historical delay rates |
| Validation | 2025-07 → 2025-12 | Model selection and calibration only |
| **Holdout** | **2026-01 → 2026-08** | **Reported once, at the end** |

Historical-rate features (airport, airline, route, hour-of-day rates) are
computed **only from months before the prediction month's set**: train rates
for validation, train+validation rates for the holdout. No feature uses
anything known after the scheduled departure.

## Targets

| Target | Definition | Notes |
|---|---|---|
| `delayed` | DepDelay ≥ 15 min | Cancelled flights excluded (they have no departure delay) |
| `severe` | DepDelay ≥ 120 min | Same population as `delayed` |
| `cancelled` | Cancelled = 1 | Reported separately, over all flights |
| `disrupted` | delayed OR cancelled OR diverted | What a traveller actually experiences |

## Models evaluated

1. **Committed model** (`models/flight_delay_model.joblib`), with features built
   exactly the way the production API builds them (`main.build_features`).
   This measures what users would actually get today.
2. **Retrained booking-time model**: same model family, features computed
   properly from training data (real airport/airline/route/hour rates,
   great-circle distance).
3. **Day-of reference** (the lead-time comparison): the booking-time features
   plus what a day-of tracker like Flighty knows a few hours before
   departure, namely the departure delay of the same aircraft's previous leg
   (BTS `Tail_Number`, previous leg scheduled to depart before this flight).
   FlyRely can't know this at booking. The gap between models 2 and 3 is the
   accuracy given up by predicting early.

Flighty's own predictions are not available as data, so model 3 is the
documented proxy for "what Flighty surfaces day-of".

## Pass / fail bar (holdout, booking-time model 2)

The thesis gets a **PROCEED** only if **all** of these hold:

1. **Discrimination:** ROC-AUC for `delayed` ≥ **0.70**.
2. **Calibration:** expected calibration error (10 equal-count bins) ≤ **0.02**,
   and every decile's predicted vs observed delay rate within **±0.03**.
3. **Tier separation:** with risk tiers set on validation data,
   observed `delayed` rate in High ≥ **2×** Low, tiers in the right order
   (Low < Medium < High) in **every** holdout month.
4. **Severe delays:** the riskiest 20% of flights contain ≥ **35%** of `severe`
   delays.
5. **Cancellations:** the riskiest 20% of flights contain ≥ **30%** of
   cancellations.
6. **Stability:** monthly ROC-AUC never below **0.65**.

The committed model (1) is held to the same bar and reported side by side.

## Outcomes

- **PROCEED:** all six pass. Next step: beta would-pay test.
- **ITERATE:** fails, but AUC ≥ 0.66 and criteria 3–4 pass. Next step: one
  pre-registered weather experiment (Open-Meteo forecasts as available at
  booking lead time), with a fresh protocol commit.
- **STOP / PIVOT:** anything worse. The report says so plainly and does not
  tune the test until it passes.

## Reproducibility

`backtest/` holds the scripts. Data is downloaded by `backtest/fetch_bts.sh`
into `data/bts/` (gitignored). Random seed 42. Results, tables and the
recommendation go in `backtest/REPORT.md`.
