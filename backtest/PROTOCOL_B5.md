# Protocol B5: do observed METAR and FAA GS/GDP programs improve run-up delay prediction?

Pre-registered. This file is committed BEFORE any model with the new features is trained or scored
(`backtest/experiments_b5.py` is written and run only after this commit). Modelled on B1
(`experiments_v3.py`, `REPORT_V3_VALIDATION.md`).

## Question

Over the run-up models that exist today (the `bts18-gb-calib-v2` booking model plus the T-1d / T-3h / T-1h
models of `PROTOCOL_RUNUP.md`), does adding either free source improve accuracy?

* **A. Observed airport weather (METAR)**, Iowa Environmental Mesonet ASOS archive.
* **B. FAA traffic-management programs**, ground stops (GS) and ground delay programs (GDP), ATCSCC advisories database.

The product owner adds a source only if it helps.

## Data (fetched with cached, resumable, sequential scripts)

* METAR: `backtest/fetch_metar.py`. Routine and special reports, UTC, 2024-01-01 to 2026-08-31, for the 100 busiest
  airports in `parquet_v2` over 2025-2026 (the same list as `fetch_weather.py`). One request per airport-year, a
  2 s pause, back-off on 429/503, raw CSV cached in `data/metar/`.
* FAA: `backtest/fetch_faa_adv.py`. One list page per day (all GS and GDP advisories of the day: number, airport,
  title, send time in UTC), 974 pages cached in `data/faa_adv/`. Historical data IS retrievable and parseable
  (27,045 GS/GDP advisories 2024-01..2026-08). The per-advisory detail pages (program period) hold ~17,000 pages
  for the 100 airports and the server needs ~7 s per page, so they are NOT fetched in full. Amendment recorded
  before any evaluation: every advisory gets the same documented default period (below), and a random sample of
  detail pages is fetched only to check those defaults.
* Nothing dated 2026-09 or later is fetched or used (both scripts refuse it without `FLYRELY_HOLDOUT_UNLOCK=1`;
  `experiments_b5.py` hard-fails on a month >= 2026-09). 2026-09 onward stays the clean holdout.

## Splits (identical to the run-up backtest, `backtest.PERIODS`)

| Role | Flights | Trailing-rate window |
|---|---|---|
| Fit (model weights) | 2025-01 .. 2025-06 | 2024-01 .. 2024-12 |
| Calibration (isotonic) | 2025-07 .. 2025-12 | 2024-07 .. 2025-06 |
| **Evaluation** | **2026-01 .. 2026-08** | 2025-01 .. 2025-12 |

2026-01..08 is validation, not an untouched holdout (`PROTOCOL_ADDENDUM_2026-11.md` item 3). A positive result here is
therefore a recommendation to confirm on the clean holdout (2026-09 onward), not a final proof. Cancelled flights are
excluded from AUC/Brier/ECE (the target is `DepDelay >= 15`), as in `runup.score`.

## Horizons and availability

The cutoff is `c = scheduled departure - h`. Only information stamped at or before `c` is used.

| Horizon | A. METAR | B. FAA programs |
|---|---|---|
| Booking, T-7d | not tested: the weather "now" at departure is unknown | not tested: no program is in force a week ahead |
| T-1d | not tested: same reason | tested (exploratory): programs advised by `c` that cover the flight |
| T-3h | tested | tested |
| T-1h | tested | tested |

Live sources for serving: aviationweather.gov METAR API (A), FAA NAS Status `airport-status-information` (B). The
training definitions use the same fields (`flyrely_features.metar.from_aviationweather` is the mapping).

## Feature definitions (exact; code in `flyrely_features/metar.py` and `faa.py`)

### A. Observed weather at the cutoff, per side `o` (origin) and `d` (destination)

For airport X, take the latest METAR/SPECI with observation time `<= c`. If the report is older than 180 minutes, or X
has no station in the fetched set, every A feature of that side is NaN (never a "calm" default). Features
(`mt_{o,d}_*`):

* `ceiling_ft`: lowest base of a BKN, OVC or VV layer; 12000 if none.
* `vis_mi`: surface visibility in statute miles.
* `wind_kt`, `gust_kt` (0 if no gust reported).
* `precip`: 1 if any present-weather token contains RA, DZ, SN, GR, GS, PL, SG, IC or UP (vicinity `VC` tokens excluded).
* `snow`: 1 if SN or SG. `thunder`: 1 if TS (vicinity thunderstorms included). `freezing`: 1 if FZ. `fog`: 1 if FG (not VC).
* `ifr`: 1 if ceiling < 1000 ft or visibility < 3 mi.
* `p01i`: last-hour precipitation (inches, 0 if missing). `age_min`: minutes since the report.

Same cutoff for origin and destination: the destination feature is "weather at the destination now", not at arrival.

### B. FAA programs known at the cutoff, per side (`faa_{o,d}_*`)

An advisory is visible from its send time (UTC). Per (airport, kind) the newest advisory sent at or before `c` is in
force unless it is a cancellation (CNX); a later advisory of the same kind replaces it. Its program
period is, for every advisory alike, from its send time for 90 min (GS) or 480 min (GDP); an earlier ending
(cancellation, or a replacing advisory) takes precedence. (The detail-page period is not used; the sample check
reports how well these defaults match the real periods.) Features:

* `gs_now`: a ground stop in force whose period contains `c`.
* `gdp_now`: a non-proposed GDP in force whose period contains `c`.
* `gdp_prop`: a PROPOSED GDP in force whose anticipated period contains the flight's scheduled time at that airport
  (departure for `o`, arrival for `d`).
* `gdp_cover`: a non-proposed GDP in force whose period contains the flight's scheduled time at that airport.

Airports outside the fetched 100 get NaN. GS/GDP restrict arrivals into the control element, so `d` is the direct side
and `o` a congestion signal.

## Models and arms

Per horizon the base model is the existing run-up horizon model: booking features (committed 18) + trailing 7-day
recent-performance + (T-3h/T-1h) network pending/rotation features + archived-forecast weather (approved for run-up
models; when `data/weather` is present), `HistGradientBoostingClassifier` and isotonic calibration exactly as
`backtest.fit_calibrated` (seed 42). Arms: **base**, **base + A**, **base + B**, **base + A + B**, each refit and
recalibrated on the same fit/calibration months. T-1d has only base and base + B. Also reported: the committed
booking model (`bts18-gb-calib-v2` features, refit as in `runup.py`) on the same evaluation rows.

## Metrics

On flown evaluation flights: AUC for `DepDelay >= 15`, Brier, ECE (10 equal-count bins as `backtest.ece_and_deciles`),
per-month Brier. Paired difference `Brier(arm) - Brier(base)` with a 95% day-block bootstrap (1,000 resamples,
seed 42; `runup.brier_diff`).

## Decision rule (fixed now)

A source is **adopted at a horizon** if and only if both hold on the evaluation months:

1. the 95% bootstrap interval of `Brier(base + source) - Brier(base)` is entirely below 0; and
2. `ECE(base + source) <= ECE(base) + 0.005` (calibration has not worsened by more than half a percentage point).

A source is **recommended overall** if it is adopted at one or more horizons; it is then used only at those horizons.
Reported but not gating: AUC change, months improved out of 8, the A+B arm, and the size of the gain (a CI that
excludes 0 on millions of flights can still be tiny; the report states the effect in plain language). If either
condition fails the verdict is "do not adopt". There is no re-tuning after seeing results: feature definitions and
thresholds above are final, and any change is recorded as a new protocol.

## Amendment 2 (before any evaluation): default periods

A sample of 230 detail pages (157 GS, 73 GDP, random active advisories, `--sample 250`) showed the first defaults
(GS 180 min, GDP 720 min) were far too long: real GS periods run a median 64 min from send (90th percentile 80, max 143),
real GDPs a median 426 min (mean 477, 90th percentile 715). The defaults are changed to GS 90 min and GDP 480 min. This
uses only the advisory text (no delay outcome), and no model has been trained.
