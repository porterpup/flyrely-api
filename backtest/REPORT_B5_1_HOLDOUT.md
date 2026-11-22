# B5.1 holdout (H1): September 2026 results

> **Caveat (section 9 of the protocol):** the T-3h and T-1h models use rotation features built from BTS tail numbers, which are recorded after the fact (hindsight tails). Their Brier and AUC are therefore perfect-linkage levels. On validation that meant T-3h 0.123 and T-1h 0.106; with no tail linkage the same models score 0.152 and 0.149 (`REPORT_A7_L0.md`). The gated differences below compare arms that share the same tails, so the *differences* are the result, not the levels.

**Status:** this was run as pre-registered in `PROTOCOL_B5_1_HOLDOUT.md`, with one dated amendment (Amendment 1, a documented METAR skip, explained below). Scoring was attempted **exactly once**, on code byte-identical to the freeze commit `89be2a4`. Raw numbers are in `backtest/results/h1_b5_1_2026-09.json`, and the single attempt is recorded in `h1_b5_1_2026-09.attempts.json`.

**Holdout:**
- Flight dates 2026-09-01..30 (30 dates): 630,177 evaluation flights, of which 624,486 were flown.
- Trailing-rate window: 2025-09..2026-08.
- The frozen models were fitted on data up to 2026-08 and were not refitted.

## Verdict

| Test | Source | Horizon | Validation Brier diff | Holdout Brier diff [95% CI] | ECE with / base | Outcome |
|---|---|---|---|---|---|---|
| H1-B-1d | B. FAA GS/GDP | T-1d | -0.000986 | -0.000481 [-0.000979, +0.000022] | 0.0218 / 0.0205 | not confirmed |
| H1-A-3h | A. METAR | T-3h | -0.000424 | -0.000268 [-0.000659, +0.000174] | 0.0182 / 0.0209 | not confirmed |
| H1-B-3h | B. FAA GS/GDP | T-3h | -0.002180 | **-0.003539 [-0.004236, -0.002878]** | 0.0191 / 0.0209 | **confirmed** |
| H1-A-1h | A. METAR | T-1h | -0.000345 | -0.000201 [-0.000558, +0.000198] | 0.0127 / 0.0109 | not confirmed |
| H1-B-1h | B. FAA GS/GDP | T-1h | -0.002457 | **-0.003698 [-0.004472, -0.002883]** | 0.0136 / 0.0109 | **confirmed** (ECE +0.0027, within the 0.005 tolerance) |

All CIs are 95% day-block bootstraps. `confirmed_horizons` = A: none; B: 3h, 1h.

- **No test is contradicted:** every point estimate is below 0, and no ECE condition fails for a test whose CI clears 0.
- **This matches the protocol's own forecast:** FAA at T-3h and T-1h clearly confirmed, and the other three tests "only just" at validation effect size.
- **FAA got stronger on the holdout:** its improvement at T-3h and T-1h is about 1.6× the validation effect. The weekly FAA Brier differences are below 0 in all 5 weeks at both horizons. At T-1d, week 5 (09-29..30) is +0.000273.

## What this means for the product

These rules come from protocol section 4; no new policy is added here.

- **FAA ground-stop / GDP features are served at T-3h and T-1h.**
- **FAA at T-1d is not served.** The base model is served at T-1d, and it still beats the booking model on this month: Brier diff -0.003752 [-0.006004, -0.001499].
- **METAR is not served at T-3h or T-1h.** As the protocol puts it, "we will not depend on live METAR yet". METAR was never tested at T-1d.
- **Not-confirmed results are dropped for serving and never re-tested on September.** The October second check (approved 2026-11-10) can only demote the FAA T-3h/T-1h results. It cannot revive METAR or FAA at T-1d; that would need a new protocol on the 2027 holdout.

## Every arm

AUC / Brier / ECE for each arm:

| Horizon | base | +A (METAR) | +B (FAA) | +A+B |
|---|---|---|---|---|
| T-1d | 0.7106 / 0.15695 / 0.0205 | — | 0.7141 / 0.15646 / 0.0218 | — |
| T-3h | 0.8210 / 0.11815 / 0.0209 | 0.8226 / 0.11788 / 0.0182 | 0.8300 / 0.11461 / 0.0191 | 0.8305 / 0.11476 / 0.0226 |
| T-1h | 0.8555 / 0.10179 / 0.0109 | 0.8570 / 0.10159 / 0.0127 | 0.8659 / 0.09809 / 0.0136 | 0.8669 / 0.09779 / 0.0141 |

Booking model: AUC 0.6874, Brier 0.1607, ECE 0.0104.

## Sensitivity (non-gating)

This uses the B5.1 rate window (2025-01..12). It is reported for completeness and **does not change any outcome**.

| Test | Brier diff [95% CI] | ECE with / base | Would be |
|---|---|---|---|
| FAA T-1d | -0.000657 [-0.001095, -0.000209] | 0.0106 / 0.0107 | confirmed |
| METAR T-3h | -0.000165 [-0.000554, +0.000274] | 0.0033 / 0.0051 | not confirmed |
| FAA T-3h | -0.003695 [-0.004536, -0.002928] | 0.0075 / 0.0051 | confirmed |
| METAR T-1h | -0.000207 [-0.000592, +0.000186] | 0.0040 / 0.0029 | not confirmed |
| FAA T-1h | -0.004073 [-0.004927, -0.003177] | 0.0070 / 0.0029 | confirmed |

FAA T-1d would pass under this window. Picking an analysis after seeing the result is exactly what pre-registration rules out, so **FAA T-1d stays not confirmed**.

## Stage 2a integrity checks

The final run passed all 13 checks:
- **BTS:**
  - 30/30 dates;
  - at least 10,000 rows per day (minimum 18,597);
  - total 630,177 rows vs 621,601 in Sept 2025 (within ±15%);
  - required columns present;
  - no date outside September.
- **METAR:**
  - every airport has a report on every day, with the 2 documented skips;
  - the documented skips are exactly missing days;
  - origin coverage 89.2% at 3h and 1h (threshold 85%, validation 89.5%).
- **FAA:**
  - 31/31 list pages present and parseable;
  - GS/GDP advisories on 30 days (at least 25 needed).
- **Weather:** every airport and hour is present.
- **Models:** the frozen models match the manifest.

Coverage on scored flights: METAR origin 89.21%, FAA origin 89.26%, both ends about 79%. FAA flag shares are close to validation (T-1d origin: GDP in force 4.9%, ground stop 1.7%).

Two first-try failures were fixed before any score, as section 5 allows:

1. **FAA GS/GDP on 8 days (25 needed).** The `build` step had run before the `lists` download finished: `programs.parquet` was written at 16:32:39, and the last list page at 16:33:15. Re-running `build` on the complete pages gave 30 days, and the check passed. No data was skipped.
2. **METAR: LGB 29/30 and ORF 29/30 days.** This was handled by Amendment 1.

## Amendment 1: documented METAR skip (2026-11-22)

- **Trigger:** `FAIL metar: every airport has a report on every day: 2 short: {'LGB': '29/30 days', 'ORF': '29/30 days'}`.
- **Evidence the gaps are station outages, not download faults:**
  - LGB has no report on 2026-09-16. The last report before is 09-15 18:53 UTC, and the next is about 36 h later.
  - ORF has no report on 2026-09-22. The last report before is 09-21 22:06 UTC; the end of the gap was not recorded, so the September data was not read again.
  - Both files were re-downloaded, and the re-downloads are byte-identical to the first tries (sha256 values are in `h1_metar_skips_2026-09.json`).
- **Approval:** Sean approved the documented skip (the protocol's own wording, section 5) on 2026-11-22, before any score existed.
- **Scope:**
  - It excuses exactly those two airport-days.
  - Every other airport still needed 30/30.
  - A listed day that had data would have failed the check.
- **Effect:** it changed only the check's pass/fail line. The affected LGB and ORF flights were scored with the frozen as-of METAR logic. Nothing was filled or imputed.
- **Unchanged:** origin coverage of at least 85%, every other check, the models, the features, the CI rule, the 0.005 ECE tolerance, the five gated tests, and score-once.

## Provenance

| Commit | What |
|---|---|
| `89be2a4` | Code freeze (master). |
| `e8a4139` | Stage 1: models manifest, frozen airports, interpretations. Committed before any September download. |
| `8b9b73c` | `--no-ff` merge of e8a4139 onto master code that is identical to 89be2a4 in TREE_PATHS. |
| `db0429c` | Amendment 1 and `h1_metar_skips_2026-09.json`. |
| `6a0fb06` | Temporary change to `stage_check` honouring the two skips. |
| `e70410c` | Pass marker, written by the check itself at 6a0fb06 (`code_commit` 6a0fb06, `tree_clean` true). |
| `accf2cf` | `experiments_b5.py` restored blob-identical to 89be2a4. The TREE_PATHS diff against 89be2a4 is empty. |
| `3dfafeb` | Score results. |

- **One attempt:** started 2026-11-22T22:05:08Z, with `code_commit` accf2cf and no force-rerun.
- **Manifest sha256:** `d028e7e6de8c738029b9215e0dd351c3b231461f153b7e4f295eedab122b70d6`.
- **`code.dirty: true` in the result is expected.** `git_state()` runs `git status --porcelain -- flyrely_features backtest` and counts untracked files. The attempts file is written before data loads, so it is untracked at that moment. The preflight uses `--untracked-files=no`, and it passed. The evidence for clean code is the empty TREE_PATHS diff and `code.commit` = accf2cf.

## Disclosures

- **Gap timestamps were looked at before stage 3.** Amendment 1 needed the METAR timestamps around the two gaps (METAR only, with no delay outcomes). Section 5 says nobody opens the September data before stage 3; this is the one exception.
- **The BTS release was confirmed with a scripted HEAD request**, not in a browser: `transtats.bts.gov/PREZIP/…_2026_9.zip`, `Last-Modified` 2026-11-19. Nothing was downloaded that way; the download went through `fetch_bts.sh --holdout`.
- **Sean ran the METAR and FAA fetches himself.** A privacy hook blocks automated runs of the scripts that embed his contact address in the User-Agent. Follow-up: read that contact from an environment variable.
- **October-1 UTC rows** (METAR, FAA lists and weather) are feature inputs only, read as of each cutoff.

## Reproduce

Not repeatable by design: `--stage score` refuses a second attempt on 2026-09. Stage 2a can be repeated only at 6a0fb06; after the restore it fails on the two documented gaps. To verify instead of re-running:
- check `git diff 89be2a4 accf2cf -- <TREE_PATHS>` (it is empty);
- compare the model hashes with the manifest;
- compare the data hashes with `h1_check_2026-09.pass.json`.
