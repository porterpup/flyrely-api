# Protocol H1: B5.1 holdout confirmation on September 2026

Pre-registered. This file is committed **before** any data dated 2026-09 or later is downloaded, and before any code
for the holdout run is written. It is roadmap step 3 (H1) of `PLAN_ROADMAP_2026-11.md`, section 6. The roadmap calls
this file `PROTOCOL_B5_HOLDOUT.md`; it is named `PROTOCOL_B5_1_HOLDOUT.md` because it confirms the B5.1 run.

Builds on `PROTOCOL_B5.md` and `PROTOCOL_B5_1.md`. Results it confirms: `REPORT_B5_1.md`,
`results/b5_1_metar_faa.json`. Written 2026-11-05. No data calls were made to write it.

**In one sentence:** take the exact B5.1 models, trained on 2025 data only, score them once on September 2026 (data no
model or person has seen), apply the same decision rule, and report whatever comes out.

---

## 1. Question

Do the five B5.1 adoptions hold on a month nobody has looked at?

| ID | Source | Horizon | B5.1 Brier diff vs base [95% CI] (validation 2026-01..08) |
|---|---|---|---|
| H1-B-1d | B. FAA GS/GDP | T-1d | -0.000986 [-0.001165, -0.000800] |
| H1-A-3h | A. METAR | T-3h | -0.000424 [-0.000508, -0.000343] |
| H1-B-3h | B. FAA GS/GDP | T-3h | -0.002180 [-0.002378, -0.001976] |
| H1-A-1h | A. METAR | T-1h | -0.000345 [-0.000427, -0.000265] |
| H1-B-1h | B. FAA GS/GDP | T-1h | -0.002457 [-0.002694, -0.002205] |

These five are the only gated tests. Nothing else is tested for adoption.

## 2. What is frozen (no re-tuning)

Everything in `PROTOCOL_B5.md` and `PROTOCOL_B5_1.md` stays exactly as written:

- **Feature code:** `flyrely_features/` (`metar.py`, `faa.py`, `stations.py`, `booking.py`, `weather.py`, `_common.py`),
  `backtest/runup.py` feature functions, `backtest/backtest.py` feature and fitting functions, and
  `backtest/fetch_faa_adv.classify()` / `parse_list()`. **Code base:** master at `c153739` (its code is identical to the
  B5.1 merge `2825135`; only a plan file was added since). The H1 changes in section 8 may add a mode, paths, guards and
  checks. They may not change any feature definition, model setting or metric.
- **Feature definitions:** the B5.1 ones, including the 10-minute METAR publication lag, the 180-minute staleness limit,
  the tz-database clock offsets, cancellation wording, and the **default FAA program periods (GS 90 min, GDP 480 min)**.
  Detail pages are not used.
- **Airport sets:** the same 100 METAR/FAA airports as B5.1 (`busiest_airports(100)` over 2025-01..2026-08; September
  is never used to choose them), and the same archived-forecast weather airports.
- **Arms:** per horizon, as B5.1. T-1d: base, +B. T-3h and T-1h: base, +A, +B, +A+B (+A+B is information only).
  Context: the committed booking model (`bts18-gb-calib-v2` features) on the same rows.
- **Model and calibration:** `backtest.fit_calibrated` (HistGradientBoostingClassifier + isotonic), seed 42.
- **Training data (all before the holdout):**

  | Role | Flights | Trailing-rate window |
  |---|---|---|
  | Fit (model weights) | 2025-01 .. 2025-06 | 2024-01 .. 2024-12 |
  | Calibration (isotonic) | 2025-07 .. 2025-12 | 2024-07 .. 2025-06 |
  | **Holdout evaluation** | **2026-09** (all days, flight date) | **2025-09 .. 2026-08** |

  These are the B5.1 models, byte for byte (section 5, stage 1 checks this). They are **not refit** on 2026 data: a
  refit for production is step P2 of the roadmap, after H1, under its own plan.
  The holdout's trailing-rate window follows the rule every split already uses: the 12 months just before the first
  evaluation month. (B5.1's evaluation used 2025-01..12 for 2026-01..08.) As a non-gating sensitivity, the report also
  gives the results with the B5.1 window, 2025-01..2025-12.
- **Same-day features** (network, rotation, trailing 7-day recent performance) are built from BTS September rows as
  of each flight's cutoff, exactly as in validation. This is how every horizon model has always been evaluated.
- **Population:** every BTS Marketing Carrier row with flight date in 2026-09. Cancelled flights are excluded from AUC,
  Brier and ECE (target `DepDelay >= 15`), as in `runup.score`.

**Known, accepted quirk:** `month` is a booking feature, and the fit months (Jan to Jun) never contain September. This
was equally true of July and August in validation, and it affects every arm the same way. It is not changed.

## 3. Metrics and bootstrap (unchanged)

On flown holdout flights, per arm: AUC (`DepDelay >= 15`), Brier, ECE (10 equal-count bins, `backtest.ece_and_deciles`),
max decile gap, AUC for "disrupted".

Per gated test: the paired difference `Brier(base + source) - Brier(base)` with a 95% day-block bootstrap,
**1,000 resamples, seed 42** (`runup.brier_diff`). The blocks are the 30 flight dates of September.

Reported but not gating: AUC change, the +A+B arm, the per-week Brier difference (4 full weeks plus the remainder,
replacing "months improved", which needs more than one month), the comparison with the booking model, the coverage of
the new features, the share of flights with a GS/GDP flag (to compare with validation), and the B5.1-rate-window
sensitivity above.

## 4. Decision rule and success criteria (unchanged rule)

For each of the five gated tests, on September 2026:

> **Confirmed** if and only if the 95% day-block bootstrap interval of `Brier(base + source) - Brier(base)` is entirely
> below 0 **and** `ECE(base + source) <= ECE(base) + 0.005`.

Each test gets one of three outcomes:

| Outcome | Condition | Consequence |
|---|---|---|
| **Confirmed** | the rule holds | The source may be served at that horizon (roadmap P2-P4). |
| **Not confirmed** | the rule fails, but the point estimate of the Brier difference is below 0 | Dropped at that horizon for serving. It is not re-tested on September, ever. |
| **Contradicted** | the point estimate is at or above 0, or the ECE condition fails | Dropped at that horizon and flagged. Diagnostics are allowed for learning, but any fix is a new protocol tested on a new holdout (section 7). |

**Overall success** of H1 is defined per source: a source is confirmed at the horizons where its test is confirmed, and
it is used only there. There is no "overall pass" that rescues a failed horizon.

**What we expect, written before looking** (rough power check). If the true effects equal the validation ones and the
day-to-day noise is similar, a 30-day bootstrap interval is about √(243/30) ≈ 2.8 times wider than B5.1's 243-day one:
roughly ±0.00052 (T-1d FAA), ±0.00057 and ±0.00070 (T-3h and T-1h FAA), and ±0.00023 (METAR at both). So FAA at T-3h and
T-1h should be confirmed clearly. FAA at T-1d and both METAR tests would only just pass at the validation effect size.
September usually has fewer weather programs than summer, so the true effect may be smaller and these three can fail.
**That is an acceptable, informative result, not a reason to change anything.** This estimate is not part of the rule.

### If the holdout disagrees with validation
1. The holdout result decides. Validation is not used to overrule it, and the two are not pooled.
2. No re-tuning, no new features, no new thresholds, no second run on September with changed code. The single exception
   is an **integrity failure** (section 5, stage 2a), which is found **before** any score is computed: the data is fixed
   and the run is repeated, and the report says so.
3. If a number turns out to be wrong because of a bug found **after** scoring, the bug fix and the re-run are reported
   next to the original result, both labelled. The decision then needs `reviewer` sign-off and a note in the report. A
   bug fix that changes a feature definition is a new protocol, not an H1 re-run.
4. The report says plainly what this means for the product. For example: "FAA programs are confirmed at T-3h and T-1h;
   METAR did not clear the bar on September, so we will not depend on live METAR yet."
5. A "contradicted" result for FAA at T-3h or T-1h (the large effects) would mean something is wrong: leakage in
   validation, a September data problem, or a real change. The report must then include a short diagnosis. FAA
   features are not served until a new protocol explains the result.

## 5. Procedure (in this order; nothing is skipped)

### Stage 0. Before September is published (no holdout data, no unlock)
1. [implementer] Make the code changes in section 8 on a branch `exp/b5-1-holdout`. They must not touch feature or model
   logic (checked by `reviewer` with `git diff c153739 -- flyrely_features backtest/runup.py backtest/backtest.py`).
2. [implementer] Tests pass on CI (Ubuntu/dash) as well as on macOS, including the new guard tests.
3. `reviewer` approves and the branch is merged **before** the unlock. The merge commit hash is the **H1 code freeze**,
   and it is written into the report.

### Stage 1. Freeze the models (no unlock; data through 2026-08 only)
4. `python backtest/experiments_b5.py --holdout 2026-09 --stage freeze` (no unlock needed, and refused if any 2026-09
   data is present):
   - rebuilds the fit and calibration features from scratch and checks they equal the B5.1 cache in `data/cache/b5_1/`
     (`X_{h}_{fit,val}.parquet`, exactly equal, NaNs equal);
   - fits every arm (seed 42) and **re-predicts the validation months 2026-01..08**. Every AUC, Brier, ECE and CI must
     equal `results/b5_1_metar_faa.json`. If anything differs, stop and report: the models are not the B5.1 models;
   - saves the models to `data/holdout/models/` (joblib) and writes `backtest/results/h1_models_manifest.json` (SHA-256 of
     each model file, the code-freeze commit, the feature column lists in order, and the B5.1 reproduction check).
5. Commit the manifest to `exp/b5-1-holdout-run` and push it **before downloading September**. This is the proof that
   the models were fixed before the holdout was opened.

### Stage 2. Download September only (unlock per command, never exported)
Run only after BTS lists September 2026 (expected around Nov 17-20; August appeared Oct 19). Check the release by
opening the BTS download page in a browser, not by script.

`FLYRELY_HOLDOUT_UNLOCK` is set as a prefix on each single command. It is never `export`ed, never put in `.env`, a shell
profile or CI, and it is checked afterwards. After section 8's change it takes the **month** as its value, so it opens
only that month:

```
# BTS September 2026 (one zip, about 30 MB), into the holdout folder only
FLYRELY_HOLDOUT_UNLOCK=2026-09 sh backtest/fetch_bts.sh --holdout 2026-09
FLYRELY_HOLDOUT_UNLOCK=2026-09 venv/bin/python backtest/prepare.py --source marketing --holdout 2026-09

# METAR, 100 airports, 2026-09-01 .. 2026-10-01 (100 requests, 2 s pause, about 4 minutes)
FLYRELY_HOLDOUT_UNLOCK=2026-09 venv/bin/python backtest/fetch_metar.py --holdout 2026-09

# FAA advisory list pages only (31 pages, 1.5 s pause). No detail pages. Then build the programs file.
FLYRELY_HOLDOUT_UNLOCK=2026-09 venv/bin/python backtest/fetch_faa_adv.py lists --holdout 2026-09
FLYRELY_HOLDOUT_UNLOCK=2026-09 venv/bin/python backtest/fetch_faa_adv.py build --holdout 2026-09

# Archived-forecast weather (Open-Meteo Previous Runs), same airports as B5.1. The base model uses it
FLYRELY_HOLDOUT_UNLOCK=2026-09 venv/bin/python backtest/fetch_weather.py --holdout 2026-09

env | grep FLYRELY_HOLDOUT || echo "unlock not set: OK"
```

Rules for the download:
- **Where:** everything goes under `data/holdout/2026-09/` (`bts/`, `metar/`, `faa_adv/list/`, `faa_adv/programs.parquet`,
  `weather/`), never in the main `data/bts/parquet_v2/`, `data/metar/`, `data/faa_adv/` or `data/weather/` folders. This
  matters: `fetch_weather.busiest_airports()` and `prepare.py` read every file in the main folders, and
  `fetch_faa_adv.list_frame()` would merge September into `programs_v2.parquet`.
- **Date range:** BTS is September flight dates only. METAR, FAA lists and weather cover **2026-09-01 to 2026-10-01**
  (UTC), because the cutoffs of late-September evening flights in western time zones fall on October 1 in UTC. These
  October-1 rows are only feature inputs, read strictly as of each cutoff. No October flights or outcomes are loaded. The
  run must refuse any BTS row dated after 2026-09-30.
- **What is not fetched:** FAA detail pages (default periods are the protocol), October BTS, anything else.
- **Cost and politeness:** all sources are free and need no account. Same pauses and back-off as before.
- Nobody opens, summarises or plots the September data before stage 3. Stage 2a below is automatic.

### Stage 2a. Integrity checks (automatic, before any score)
`experiments_b5.py --holdout 2026-09 --stage check` (unlock set for that command only) checks and prints only these, then
stops. None of them uses delay outcomes:
- BTS: 30 distinct flight dates, every one with at least 10,000 rows, total rows within ±15% of September 2025, the
  required columns present, no flight date outside September.
- METAR: all 100 airports with at least one report per day on 30 of 30 days; origin coverage of holdout flights at least
  85% (validation: 89.5%).
- FAA: 31 of 31 list pages present and parseable; at least one GS/GDP advisory on at least 25 days.
- Weather: archived forecasts for every B5.1 weather airport and every hour in range.
- The frozen model files match `h1_models_manifest.json`.

If a check fails, the data is fixed (re-download, or a documented skip) and stage 2a is repeated. This does not count as a
second try, because no score has been computed.

### Stage 3. The one scoring run
6. `FLYRELY_HOLDOUT_UNLOCK=2026-09 venv/bin/python backtest/experiments_b5.py --holdout 2026-09 --stage score`, run once.
   It loads the frozen models (checking their hashes), builds September features with the frozen code, predicts, scores,
   and writes `backtest/results/h1_b5_1_2026-09.json`. It never fits a model.
7. Check that the unlock is not set.
8. [implementer] writes `backtest/REPORT_B5_1_HOLDOUT.md` in the `REPORT_B5_1.md` format: the five outcomes, the full
   table next to the B5.1 numbers, the integrity results, the code-freeze commit, and the plain-language product meaning.
   Then a PR to master and `reviewer`.

## 6. After H1: what September becomes
Once the H1 report is merged, September 2026 is **spent**. It becomes ordinary data that later fits and diagnostics may
use. The September files are moved from `data/holdout/2026-09/` into the main folders only then, by a separate commit
that changes the guard (section 8, item 1).

## 7. Policy for October 2026 and the next clean holdout (PENDING SEAN'S APPROVAL)

This is the roadmap recommendation (section 6 and decision 9). Until Sean approves it, **October 2026 onward stays locked**
exactly as now, and nothing below is run.

**7.1 October 2026: second confirmation.** When BTS October lands (around mid-December), run this same protocol again with
`2026-10` in place of `2026-09`. That means: the same frozen models (the stage-1 manifest; no refit), trailing rates
2025-10..2026-09, inputs 2026-10-01..2026-11-01 UTC, and the same checks, rule and report format (`REPORT_B5_1_HOLDOUT_OCT.md`).
The October result can only take something away:
- confirmed in September **and** in October: **confirmed twice**, served as planned;
- confirmed in September, but October's point estimate is at or above 0, or its ECE fails: **demoted**. Taken out of
  serving (or of the P2 plan) until a new protocol explains it;
- confirmed in September, October rule fails but the point estimate is below 0: kept, noted as "weaker in October";
- **not confirmed in September:** October cannot bring it back. Only a new pre-registered protocol, tested on the 2027
  holdout (7.3), can.
- One note: October 31 has no clock change in 2026 (US DST ends November 1), so the B5.1 noon-offset quirk does not apply.

**7.2 November and December 2026: SWIM-parity months.** These are the first months with our own SWIM archive. They are used
for V1 (SWIM vs BTS parity) and for development. They are **not** a model holdout.

**7.3 A new clean holdout from 2027-01.** From the moment Sean approves, months from **2027-01** onward are the protected
holdout for any new model or feature protocol (including a B6 with real program periods, and the P2 refit). The guard
moves from "2026-09 and later" to "2027-01 and later" in one reviewed commit, made **only after** the October report is
merged. Until then, September (until H1 is reported), October, and everything after stay locked. Each later
confirmation uses one new month once, following the same pattern as this protocol.

## 8. Code changes needed (all [implementer]; none made in this PR)

1. **[implementer] One holdout policy module, `backtest/holdout.py`.** It replaces the separate constants
   (`FORBIDDEN_FROM` in `experiments_b5.py` and `experiments_v3.py`, `HOLDOUT_LAST_ALLOWED`/`HOLDOUT_FIRST_MONTH` in
   `fetch_metar.py`, `fetch_faa_adv.py` and `fetch_weather.py`, and `HOLDOUT_START` in `fetch_bts.sh`). It defines
   the locked range (now: 2026-09 and later) and `unlocked_month()`. That function returns a month only when
   `FLYRELY_HOLDOUT_UNLOCK` equals `YYYY-MM` **and** the command was given the matching `--holdout YYYY-MM`. The old
   value `1` is refused with a message explaining the new form. Everything else stays locked. `fetch_bts.sh` mirrors
   this in POSIX sh (it runs under dash on CI).
2. **[implementer] A `--holdout YYYY-MM` flag on each fetcher**, writing only to `data/holdout/YYYY-MM/`:
   - `fetch_bts.sh --holdout 2026-09`: one month's zip into `data/holdout/2026-09/bts/`;
   - `prepare.py --holdout 2026-09`: converts only that zip, into `data/holdout/2026-09/bts/2026-09.parquet`;
   - `fetch_metar.py --holdout 2026-09`: one request per airport for the month plus one day, into
     `data/holdout/2026-09/metar/<IATA>_2026-09.csv`. The airport list is the frozen B5.1 list (the 100 current
     `data/metar/*_2026.csv` airports, saved to `backtest/results/h1_airports.json` in stage 1), not recomputed;
   - `fetch_faa_adv.py lists|build --holdout 2026-09`: list pages and a `programs.parquet` under the holdout folder,
     built with the same `classify()`/`parse_list()`. `details` refuses `--holdout`;
   - `fetch_weather.py --holdout 2026-09`: the same airports as `data/weather/`, `start_date`/`end_date` limited to the
     month plus one day, into `data/holdout/2026-09/weather/`.
3. **[implementer] Loaders that read main plus holdout folders.** `bt.load_months`, `runup.load(end=...)`,
   `metar.Metar(..., metar_dirs=[...])`, `weather.Weather(..., weather_dirs=[...])` and the program loader in
   `experiments_b5.py` take an optional extra folder or range. The defaults stay exactly as now, so B5.1 reproduces.
4. **[implementer] `experiments_b5.py --holdout YYYY-MM --stage freeze|check|score`** (section 5):
   - `freeze`: rebuild the fit/val features and compare them with the B5.1 cache; fit; re-predict validation and compare
     with `b5_1_metar_faa.json`; save the models and the manifest. Refused if any holdout data is present.
   - `check`: the integrity checks of stage 2a only; prints no outcome-based number.
   - `score`: load the models and check their hashes; holdout features (eval rows = that month only, trailing rates =
     the previous 12 months, plus the B5.1-window sensitivity); predict, score and decide. Never fits. Refuses to run twice
     if the result file exists (`--force-rerun` exists only for the bug-fix case in section 4, and the run logs it).
   - Cache in `data/cache/b5_1_holdout_YYYY-MM/`, results in `results/h1_b5_1_YYYY-MM.json`. Holdout mode outcomes:
     confirmed / not confirmed / contradicted (section 4). Per-week Brier difference instead of months improved.
   - Without `--holdout`, behaviour is byte-for-byte B5.1.
5. **[implementer] Tests (`tests/test_holdout_guard.py`), offline only:** the unlock opens only the named month; `=1` is
   refused; without the flag every fetcher refuses 2026-09; `--holdout 2026-09` refuses October BTS rows and 2026-11
   dates; holdout writes never reach the main data folders; `score` refuses without a manifest, on a hash mismatch, and
   on a second run; the three-outcome function; the sh guard under dash.
6. **[executor] After H1 is merged:** move the September files into the main folders and set the locked range to start at
   2026-10 (one commit; section 6).

## 9. Outputs

> **Reporting note (added after A7, PR #20; not part of the frozen code):** the T-3h and T-1h models use hindsight BTS tails for rotation features, so the holdout Brier/AUC are perfect-linkage levels (about 0.123 / 0.106 on validation; the tail-free level is 0.152 / 0.149). Report them with this caveat; see `REPORT_A7_L0.md`.

- `backtest/results/h1_models_manifest.json`, `backtest/results/h1_airports.json` (stage 1, committed before the
  download).
- `backtest/results/h1_b5_1_2026-09.json`, `backtest/REPORT_B5_1_HOLDOUT.md` (stage 3).
- Holdout data under `data/holdout/2026-09/` (gitignored, like all `data/`).

## 10. Not verified
- The BTS September release date. "About Nov 17-20" is inferred from August appearing on Oct 19.
- Whether IEM, the FAA advisory database and Open-Meteo Previous Runs have complete September 2026 data. Stage 2a checks
  this before scoring.
- The power estimate in section 4 assumes the September day-to-day variance matches validation.
