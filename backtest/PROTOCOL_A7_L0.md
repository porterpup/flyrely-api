# Protocol A7 + L0: how much is aircraft (tail) linkage worth, and how stable are rotations?

Pre-registered. This file is committed BEFORE any A7/L0 evaluation is run (roadmap step 5, `PLAN_ROADMAP_2026-11.md`
sections 3.3, 3.5, 3.6). Offline, BTS only, $0, no network calls. Decides whether any effort (and any paid data such as
AeroDataBox) on knowing the inbound aircraft is worth it, and which free route to take.

## Data and splits (unchanged from B5.1)

Fit 2025-01..06 (trailing rates 2024), calibration 2025-07..12, evaluation **2026-01..08** (validation months, per
`PROTOCOL_ADDENDUM_2026-11.md`; not the protected holdout). Nothing dated 2026-09 or later is loaded and
`FLYRELY_HOLDOUT_UNLOCK` is never set. Model and calibration: `backtest.fit_calibrated` (seed 42). Clock: the B5.1
`zoneinfo` per-day offsets (`runup.airport_offsets`). Base features: `runup.horizon_features` exactly as in B5.1 (incl.
archived-forecast weather), cached separately in `data/cache/a7_l0/`.

## Important fact about the existing base

The B5.1 run-up base at T-3h and T-1h ALREADY contains perfect-linkage rotation features (`rot_*`, from BTS tail
numbers). So the questions are asked by removing/degrading them, not by adding them to a tail-free base.

## Questions

- **A7-1** What is perfect linkage worth? Brier/AUC/ECE of the tail-free model (arm N) versus the full model with
  perfect linkage (arms P = existing `rot_*`; P+ = `rot_*` plus the new inbound-state features below), at T-3h and T-1h.
- **A7-2** How much of that survives at realistic linkage rates p in {0.29 (SFDPS registration fill), 0.50, 0.75}
  (also 0.25, 0.70, 0.90 as curve points)? A flight is "linked" with probability p, independently, by a fixed seeded
  hash of its row (seed 42); unlinked flights have every rotation and inbound feature missing (NaN).
  - **Serve-trained** (primary): the model is trained and calibrated with the same missingness as it is served with.
  - **Mismatch** (secondary): the perfect-linkage model (P+), calibrated as usual, served on degraded eval data.
- **A7-3** Non-random pattern: "regional" flights (Reporting_Airline differs from Marketing_Airline_Network, i.e.
  contract-operated) are never linked; mainline flights are linked at p = 0.29 and at p = 1.0 (serve-trained).
- **A7-4** (bridge to L0) Prior linkage: no tail is used. The inbound of flight B is *predicted* as the flight number
  that preceded B at B's origin in the previous week's rotation (same flight number and weekday, day d-7; fall back to
  day d-1), restricted to rotations from days strictly before the cutoff day. The predicted inbound's real state at the
  cutoff (departed / delay / pending / projected arrival) feeds the same inbound features (arm PR, serve-trained, no
  tail anywhere). Reports the pairing hit rate (predicted inbound's tail equals B's actual tail) and the Brier gain over N.
- **L0-1** Pairing stability. For each flight B with a tail and an immediate predecessor leg A of the same tail that
  ends at B's origin within 18 h, the pairing is (flight number of A -> flight number of B, at B's origin). Report
  P(pairing on day d equals pairing on the same flight number and origin on day d-7) and on d-1, over flights B for
  which a pairing exists on the earlier day; also the share of B with a pairing on the earlier day (coverage), and the
  unconditional hit rate (hit / all B with a pairing). By mainline vs regional, by carrier, and for the 12 A5b hubs
  (DEN LGA SEA DFW ORD CLT DTW BNA ATL MSP MDW RDU). Also: weekday vs weekend, and the share of pairings that are
  through-flights (same flight number in and out, method M2). Additional predictor: the most common pairing over the
  last 4 same weekdays (plurality of existing ones). Days: evaluation months 2026-01..08.

## New inbound-state features (arm P+, PR), all known at the cutoff c

For the immediate predecessor leg A of B's tail (A ends at B's origin, within 18 h before B's scheduled departure):
`in_departed` (A actual departure <= c), `in_cancelled_known` (A cancelled and A's scheduled departure <= c, else 0),
`in_dep_delay` (actual DepDelay if departed by c, else lower bound max(c - sched, 0)), `in_proj_arr` (departed:
actual departure + scheduled block time; not departed: max(c, scheduled departure) + block time; minus B's scheduled
departure), `in_slack` (B scheduled departure - projected arrival - 35 min turn), `in_proj_late` (max(-in_slack, 0)),
`in_eta_passed` (projected arrival <= c, a proxy for "has landed"), `in_through` (same flight number as B),
`in_minutes_to_dep_sched` (B sched dep - c; constant per horizon, omitted). **Limitation:** the BTS parquet has no
ArrDelay/ArrTime, so true actual arrival is unavailable; "arrived" is an estimate (actual departure + scheduled block),
as a live feed would give an ETA. A diverted inbound is not identifiable from `Diverted` timing at c and is ignored.

## Leakage discipline

1. Every inbound feature uses only times at or before c: A's actual departure is compared with c (both in the B5.1
   Eastern-reference clock); a not-yet-departed A contributes only its schedule and c. Cancellations are visible only once
   A's scheduled departure <= c. B's own outcomes are never features.
2. **The tail is an assumption.** BTS records the tail that actually flew, after the fact. In operations, tails are swapped
   (aircraft changes) and the plan known at T-3h or T-1h may differ. So "perfect linkage" here is an UPPER BOUND on what
   a perfectly informed linkage service could deliver, and every gain below is an upper bound. Arm PR (A7-4) avoids this
   (no tail used), but its predicted pairings are evaluated against BTS tails for the hit rate only.
3. Prior linkage (PR, L0 predictors) use only flights on days strictly before the cutoff day of B (local reference
   clock), never the same day.

## Metrics

Brier, AUC (delayed = DepDelay >= 15, cancellations excluded), ECE (10 equal-count bins, as in B5.1), per
`runup.score`; paired differences Brier(arm) - Brier(reference) with the 95% day-block bootstrap (1,000 resamples,
seed 42), `runup.brier_diff`. AUC and ECE changes are point estimates. Report "gain" = Brier(N) - Brier(arm) (positive
is better) with its CI.

## Decision rules (fixed now)

Let G(h, arm) = Brier(N) - Brier(arm) at horizon h.

1. **Linkage priority (roadmap kill condition).** If G(T-3h, P+) < 0.0005, perfect linkage is not worth paying for at
   any price: linkage is NOT a priority and the AeroDataBox pilot is skipped. If G >= 0.0005 but the lower CI < 0.0005 it
   is "inconclusive, lean skip". If the lower CI >= 0.0005, linkage is worth pursuing.
2. **Free SFDPS level is useful on its own.** If G(T-3h, p=0.29, serve-trained) has CI entirely above 0 and the point
   estimate >= 0.0005, then even the ~29% SFDPS fill is worth collecting.
3. **Paid tail data could be justified** only if, at T-3h, Brier(p=0.50, serve-trained) - Brier(p=1.0) >= 0.0005 AND its
   95% CI lower bound > 0 (closing from a mid free level to perfect still buys >= 0.0005). If it falls below 0.0005 at
   p=0.50 (point estimate), paid tail data is not justified at any price once free linkage reaches 50%. The same is
   also reported against p=0.29 and p=0.75. (Necessary, not sufficient: L1 still has to show that the paid source is
   precise and earlier than free sources; this rule is about accuracy at stake only.)
4. **Rotation priors (free route M3) are worth building** if L0 P(same pairing as d-7 | exists) >= 0.60 overall AND
   arm PR has G(T-3h) with CI lower bound > 0 and point estimate >= 0.0005. "Strong" if G(PR) >= 50% of G(P+).
5. A7 reports the mismatch results and the regional-missing pattern without a decision rule (informational), but
   states whether training on matching missingness helped (Brier mismatch - Brier serve-trained).

Everything is reported whatever it shows. No re-tuning after seeing results; any change needs a new protocol.

## Outputs

`backtest/a7_l0.py`, `flyrely_features/inbound.py`, `tests/test_a7_l0.py`, `backtest/results/a7_l0.json`,
`backtest/REPORT_A7_L0.md`. Cache: `data/cache/a7_l0/` only.
