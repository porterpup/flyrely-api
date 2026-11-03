# B1 validation: trailing recent-performance features and rolling recalibration

Status: PRE-REGISTERED RULE (written before any result). Results are appended below by the run.

Validation only: 2026-01..08 is validation per `PROTOCOL_ADDENDUM_2026-11.md`. Months >= 2026-09 are never loaded
(the script hard-fails on them).

## Selection rule (fixed before running)

1. Candidates are servable variants only: data gap G >= 90 days (G=90, 120), each with recalibration K in {none, 1, 2, 3}.
   G=1/31/60 are diagnostics and cannot be chosen.
2. Chosen variant = lowest walk-forward Brier on flown flights, 2026-01..08.
3. Production adoption also requires the Brier difference (chosen minus committed model, scored as in `b4_diagnostic`
   with haversine distance) on 2026-01..08 to have a 95% day-block bootstrap interval (1,000 resamples, seed 42)
   entirely below 0. Otherwise: do not adopt.

Gap convention: the trailing window for departure date D covers days (D-G-W, D-G]; G=1 means yesterday is the newest
day used. The leakage test asserts no feature uses a day later than D-G.
