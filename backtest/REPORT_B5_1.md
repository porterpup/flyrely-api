# B5.1 results: B5 re-run with the review fixes

**Status:** `backtest/experiments_b5.py` ran as pre-registered in `PROTOCOL_B5_1.md`, which builds on `PROTOCOL_B5.md`
and was committed before any code change or evaluation. No amendments were needed. Raw numbers are in
`backtest/results/b5_1_metar_faa.json`.

**Validation only.** The evaluation months are 2026-01..08. Nothing dated 2026-09 or later was loaded, and
`FLYRELY_HOLDOUT_UNLOCK` was never set. There are 5,227,724 evaluation flights, of which 5,108,811 were flown
(the same as B5).

## Fixes applied

(a) **FAA cancellations.** `classify()` now treats CNX, CANCEL*, CANX and CNCL wording as a cancellation.
- 775 of 27,045 advisories were reclassified from new program to cancellation, 611 of them at the 100 modelled airports.
- New status counts: GS active 10,605, GS cancel 4,239; GDP active 5,076, GDP cancel 3,849.
- The output is written to `programs_v2.parquet`, and the old file is kept.

(b) **METAR station IDs.** These now come from `data/ref/airports.csv`, through `flyrely_features/stations.py`.
- Seven airports had empty files: ANC, HNL, OGG, KOA, LIH, SJU and PBI. They were fetched again for 2024-01..2026-08
  (21 requests, all returning data).
- IEM serves PBI under its new station ID, DJT (ICAO code KDJT).
- The empty files were moved to `data/metar/_stale_empty_b5/`.
- The live mapping no longer uses `icaoId[-3:]`.

(c) **METAR publication lag.** A report becomes usable 10 minutes after its observation time.

(d) **Clock offsets.** Each airport's offset now comes from the tz database for each day, using the time zones in
`models/airports_ref.csv`.
  This also fixes Guam and Saipan, whose offsets were 24 hours wrong in B5. Their cutoffs had been a day late (about
  2,900 flights), which was a tiny leak in B5's base features.

(e) **Holdout guards.** `busiest_airports` ignores months from 2026-09 onward, and `experiments_b5.py` refuses to run
if any METAR time falls on or after 2026-09-01.

## Decision rule (unchanged)

Adopt at a horizon only if both conditions hold:
- the 95% day-block bootstrap CI of Brier(base+source) minus Brier(base) is entirely below 0;
- ECE(base+source) is no more than ECE(base) + 0.005.

If either condition fails, do not adopt.

## Verdict

| Source | T-1d | T-3h | T-1h | Changed from B5? |
|---|---|---|---|---|
| A. METAR | not tested | adopt | adopt | no |
| B. FAA GS/GDP | adopt | adopt | adopt | no |

## B5.1 vs B5, every horizon and arm

Each cell shows B5 -> B5.1. A negative Brier difference is better. Every row improved in 8 of 8 months.

| Horizon | Arm | AUC | Brier | ECE | Brier diff vs base [95% CI, B5.1] | Adopt |
|---|---|---|---|---|---|---|
| T-1d | base | 0.6985 -> 0.6985 | 0.16437 -> 0.16446 | 0.0182 -> 0.0202 | | |
| T-1d | +B | 0.7013 -> 0.7033 | 0.16387 -> 0.16348 | 0.0194 -> 0.0196 | -0.000504 -> -0.000986 [-0.001165, -0.000800] | yes -> yes |
| T-3h | base | 0.8220 -> 0.8222 | 0.12299 -> 0.12281 | 0.0057 -> 0.0066 | | |
| T-3h | +A | 0.8242 -> 0.8240 | 0.12243 -> 0.12239 | 0.0062 -> 0.0057 | -0.000564 -> -0.000424 [-0.000508, -0.000343] | yes -> yes |
| T-3h | +B | 0.8274 -> 0.8278 | 0.12087 -> 0.12063 | 0.0037 -> 0.0057 | -0.002127 -> -0.002180 [-0.002378, -0.001976] | yes -> yes |
| T-3h | +A+B (info) | 0.8283 -> 0.8287 | 0.12071 -> 0.12047 | 0.0044 -> 0.0045 | -0.002280 -> -0.002343 [-0.002537, -0.002140] | n/a |
| T-1h | base | 0.8552 -> 0.8556 | 0.10667 -> 0.10642 | 0.0080 -> 0.0079 | | |
| T-1h | +A | 0.8572 -> 0.8571 | 0.10613 -> 0.10608 | 0.0078 -> 0.0088 | -0.000531 -> -0.000345 [-0.000427, -0.000265] | yes -> yes |
| T-1h | +B | 0.8620 -> 0.8619 | 0.10408 -> 0.10397 | 0.0062 -> 0.0064 | -0.002581 -> -0.002457 [-0.002694, -0.002205] | yes -> yes |
| T-1h | +A+B (info) | 0.8632 -> 0.8631 | 0.10382 -> 0.10368 | 0.0066 -> 0.0077 | -0.002848 -> -0.002743 [-0.002970, -0.002498] | n/a |

The base rows show the effect of the clock fix (d). The committed booking model is unchanged: AUC 0.6774,
Brier 0.17005, ECE 0.0453.

**What changed, in plain terms:**
- **FAA:** the T-3h and T-1h gains are essentially unchanged. At T-1d the gain roughly doubles (from -0.0005 to
  -0.0010), because cancellations now end the programs they cancel instead of starting fake ones.
- **METAR:** the gain shrinks by about a quarter, which fits the 10-minute lag removing the freshest reports. It is
  still small and consistent (8 of 8 months), and ECE stays within tolerance. The largest ECE worsening among the
  adopted arms is T-1h +A, at +0.0009.

## Coverage after the fix

- **METAR:** all 100 airports (was 93). The origin is covered for 89.5% of evaluation flights (was 86.9%), and both
  ends for 79.7% (was 75.7%).
- **FAA:** all 100 airports; origin 89.5%, both ends 79.8% (unchanged).
- **Remaining gap:** flights to or from airports outside the top 100, which get missing values by design.

## Caveats

1. **Validation, not holdout.** These are validation months; confirm on 2026-09 onward.
2. **Default FAA periods.** Programs still use the documented defaults (GS 90 min, GDP 480 min), and production must
   use the same.
3. **Clock offsets at local noon.** On the fall-back day, flights between midnight and 2am get a cutoff 60 minutes
   late. No clock-change day falls in the evaluation months, so the effect here is nil.
4. **Advisory wording edge cases.**
   - A title naming two kinds of program is given only one kind, GS. All such titles are at Canadian airports.
   - Partial cancellations such as "CANCELLED FOR ALL BUT DAL" end the whole ground stop (a handful of rows).
   - Two misspelled Canadian titles are missed.
5. **What the gains mean.** The FAA gain still partly reflects a "system already disrupted" signal. METAR's gain is
   small, so whether to depend on live aviationweather.gov data is a product decision.

## Reproduce

```
venv/bin/python backtest/fetch_metar.py --airports ANC,HNL,OGG,KOA,LIH,SJU,PBI
venv/bin/python backtest/fetch_faa_adv.py build
venv/bin/python backtest/experiments_b5.py
```
