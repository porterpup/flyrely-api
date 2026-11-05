# Protocol B5.1: fixes to the B5 pipeline and a re-run

Pre-registered. This file is committed BEFORE any evaluation code for B5.1 is changed or run. It follows the
"Known issues" section of `REPORT_B5_METAR_FAA.md` (found in final review of B5, PR #12). None of the issues is leakage;
they are defects in feature construction and in holdout guards. B5 stays as published; B5.1 is a new run with a new
protocol, as `PROTOCOL_B5.md` requires for any change.

## Fixes (exactly these five, plus a wording fix)

**(a) Cancellation advisories.** `fetch_faa_adv.classify()` treats a title as a cancellation when it contains the
word CNX, or any word starting with CANCEL (CANCELLATION, CANCELLED, CANCELED, CANCELATION), or CANX / CNCL. Before,
only "CNX" counted, so about 777 advisories titled like "... GROUND STOP CANCELLATION" were treated as new programs.
`programs.parquet` is rebuilt into a NEW file, `data/faa_adv/programs_v2.parquet`; the old file is kept. The report
states how many advisories changed status. A title naming both kinds (for example "GS/GDP CANCELLED") still gets
one kind, as before (the first match; a handful of rows).

**(b) METAR station codes.** The IEM station id is derived from a real mapping (`data/ref/airports.csv`, columns
`ident`, `icao_code`, `iata_code`, `gps_code`), not by prepending K: an airport is found by IATA code, or else by the
ICAO ident `K` + code. Its ICAO code is `icao_code` (else `gps_code`, else `ident`); the IEM station is the ICAO code
without its leading K when it is a 4-letter K code, otherwise the ICAO code itself (PANC, PHNL, PHOG, PHKO, PHLI,
TJSJ). For KPBI the table lists the airport under its new codes (KDJT / DJT), which is the id IEM serves (checked
by one request: PBI returns nothing, DJT returns reports). The 7 airports that have empty files (ANC, HNL, OGG, KOA,
LIH, PBI, SJU) are fetched again for 2024-01-01 to 2026-08-31 only (3 requests per airport, 2 s pause, cached,
the existing back-off); the empty files are moved aside (not deleted) to `data/metar/_stale_empty_b5/`. Nothing
else is re-fetched. The loader labels each file's rows by the airport code in its file name. The live mapping in
`flyrely_features/metar.py` (`icaoId[-3:]`, which turned TJSJ into JSJ) is replaced by the same mapping
(`flyrely_features/stations.py`).

**(c) METAR publication lag.** A report becomes usable at observation time + 10 minutes: for a cutoff `c`, only
reports with observation time `<= c - 10 min` are used. `age_min` and the 180-minute staleness limit are still
measured from the observation time to `c`.

**(d) Clock offsets.** `runup.airport_offsets` is replaced. Each airport's offset from the ATL reference clock is
computed per day from the tz database (`zoneinfo`) using the airport's IANA zone from `models/airports_ref.csv`
(offset of the airport at local noon of that date minus offset of America/New_York at local noon). The old route-graph
derivation (most common route difference per month, rounded to hours) is dropped. Airports without daylight saving
time (PHX, HNL, SJU and others) now change correctly relative to ATL on the clock-change days. The residual: flights
scheduled in the few hours before the change time on the two change days use the post-change offset. This changes the
base model's cutoffs, so the base arm is re-run and reported before (from B5) and after.

**(e) Holdout guards.** `fetch_metar.busiest_airports` ignores months from 2026-09 on (the airport list is
computed from 2025-01..2026-08 only). `experiments_b5.py` hard-fails if any METAR observation time is on or after
2026-09-01, in addition to the existing guards.

**Wording.** `PROTOCOL_B5.md` line 116 "If neither condition is met" now reads "If either condition fails".

## Unchanged from PROTOCOL_B5

Everything else: the question, the fit / calibration / evaluation splits and trailing windows, the horizons and which
sources are tested at each (METAR at T-3h and T-1h; FAA at T-1d, T-3h and T-1h), the arms (base, +A, +B, +A+B; T-1d
base and +B), the feature definitions (the default FAA program periods of 90 min for GS and 480 min for GDP included),
the model and calibration (`backtest.fit_calibrated`, seed 42), the metrics (AUC, Brier, ECE with 10 equal-count bins,
per-month Brier), the paired day-block bootstrap (1,000 resamples, seed 42), and the decision rule:

> A source is adopted at a horizon if and only if the 95% bootstrap interval of `Brier(base + source) - Brier(base)`
> is entirely below 0 AND `ECE(base + source) <= ECE(base) + 0.005`. If either condition fails the verdict is "do
> not adopt".

Evaluation months stay 2026-01..08 (validation, not holdout). Nothing dated 2026-09 or later is loaded, and
`FLYRELY_HOLDOUT_UNLOCK` is never set. No re-tuning after seeing results.

## Outputs

`backtest/results/b5_1_metar_faa.json`, `backtest/REPORT_B5_1.md`. Features and predictions are cached in
`data/cache/b5_1/` (the B5 cache is left alone). The report compares B5.1 with B5 for each horizon and arm, applies
the rule, reports the base model before and after (d), the number of advisories reclassified as cancellations, and the
METAR coverage after the fix.
