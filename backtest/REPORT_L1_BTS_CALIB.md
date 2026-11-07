# Report L1 / T4: offline calibration of the linkage rules on BTS

Plan: `PLAN_L1_LIVE_TAIL_LINKAGE.md` (section 2.3 rules, task T4). Code: `backtest/l1_bts_calib.py`, `linkage/rot_live.py`.
Results: `backtest/results/l1_bts_calib.json`. Evaluation rows: BTS validation months 2026-01..08 (5,227,724 flights; 2025-12 loaded only as
history for the d-7 / d-1 lookups). Nothing from 2026-09 on was loaded; no holdout unlock. Offline, $0.

## Summary (plain English)

- **"Same flight number continuing" (M2) is trustworthy for most carriers but not all.** Where the same flight number arrives at a
  flight's origin earlier the same day, it is the very aircraft about 91% of the time for WN, DL, AA, OO, MQ, 9E, OH, PT, YX, AS, QX
  (about 18% of those carriers' flights have such a candidate). For **United it is 3%** (UA reuses one flight number across several legs
  flown by different aircraft), and F9, NK, B6, G4 have almost no candidates. So M2 must be switched on per carrier, not globally;
  the plan's 0.85 default is right for WN/DL/AA/OO/YX/AS and too low for the regionals, but wrong for UA.
- **The weekday prior (M3) is 57% right, same as L0's 0.574, with the exclusion rule it rises as more tails are known.** Removing
  candidates whose aircraft is already filed on another departure lifts precision to 59% (about 10% of flights have a filed tail)
  and 75% (about 65% filed, the T-1h situation), at the cost of some coverage. It stays below the 0.60 bar until about T-1h.
- **Chain length.** Even with perfect knowledge the usual chain is short: at T-3h 61% of flights have the aircraft's latest departed-by-then
  leg one link back, 34% two links, 1.2% three or more, 3.7% none. At T-1h 95% are one link, 0.9% two. So chains of 3+ links are rare.
- **Not measurable here: the aircraft-type filter.** BTS has no aircraft type, so every precision below is without it. It is measured on
  the live archive (T8).
- **Code parity holds.** `rot_live(strict_asof=False, true chain)` equals the frozen `rotation_features` on 2,000 synthetic rows and on
  2,000 real BTS rows (June 2025) at atol 1e-9, at T-3h and T-1h.

## M2 precision by carrier (no exclusion, no type filter)

Candidate = latest earlier leg of the same carrier and flight number arriving at B's origin, sched OUT within 18 h before B, sched IN not after B's sched OUT.
Precision = the candidate is B's true immediate inbound leg (BTS tail). Coverage = share of the carrier's flights that have a candidate.

| Carrier | Flights | Coverage | Precision | Same tail (not nec. immediate) |
|---|---|---|---|---|
| WN | 927,832 | 15.8% | 0.888 | 0.894 |
| DL | 694,931 | 23.3% | 0.936 | 0.951 |
| AA | 667,418 | 19.9% | 0.890 | 0.912 |
| OO | 576,892 | 16.4% | 0.893 | 0.914 |
| **UA** | 559,304 | 24.6% | **0.027** | 0.027 |
| YX | 246,436 | 17.9% | 0.847 | 0.865 |
| AS | 227,606 | 5.8% | 0.904 | 0.916 |
| MQ | 217,727 | 27.9% | 0.960 | 0.976 |
| 9E | 171,357 | 25.1% | 0.957 | 0.983 |
| B6 | 165,621 | 0.1% | 0.963 | 0.976 |
| OH | 163,028 | 27.6% | 0.916 | 0.951 |
| F9 | 147,420 | 0.0% | n/a (1 candidate) | |
| PT | 102,547 | 32.0% | 0.953 | 0.973 |
| G4 | 87,442 | 0.2% | 0.042 | 0.104 |
| QX | 63,406 | 5.5% | 0.988 | 0.995 |
| G7 | 58,748 | 4.0% | 0.097 | 0.102 |
| C5 | 55,353 | 0.4% | 0.070 | 0.095 |
| YV | 47,454 | 0.5% | 0.013 | 0.013 |
| NK | 47,202 | 0.0% | n/a (1 candidate) | |

All carriers pooled: coverage 17.6%, precision 0.776 (0.911 over the twelve carriers with precision above 0.8). Mainline 0.702, regional 0.911.
With the exclusion rule (simulated M1-known share p): precision 0.776 (p=0), 0.790 (p=0.10), 0.877 (p=0.65), coverage 17.6% / 17.3% / 15.6%.

## Chain length with the true chain (links from B back to the latest leg scheduled to have left by the as-of time)

Share of all eval flights (`rot_live` on the BTS true chain; "no leg" = nothing within 18 h; "no tail" = BTS has no tail).

| Horizon | no tail | no leg | 1 link | 2 | 3 | 4 | 5+ |
|---|---|---|---|---|---|---|---|
| T-3h | 0.37% | 3.67% | 61.1% | 33.7% | 1.05% | 0.09% | 0.01% |
| T-1h | 0.37% | 3.48% | 95.3% | 0.88% | 0.01% | 0 | 0 |

Mean links when linked: 1.38 (T-3h), 1.01 (T-1h). Mainline vs regional at T-3h: 1 link 70.0% vs 42.7%, 2 links 24.5% vs 52.8% (regionals fly more short legs).
The plan's cap of 4 links loses 0.01% of chains at T-3h.

## M3 (weekday prior: d-7, else d-1) precision vs the L0 baseline of 0.57

L0 reproduction check: flight-number match given a d-7 pairing = **0.574** (identical to REPORT_A7_L0). The numbers below use row identity (the
candidate leg is B's true inbound), which is slightly stricter. "Given pairing" restricts to flights that truly have an inbound (comparable to L0).
Without the type filter. p = share of departures with a filed tail (M1-known); flights with M1 do not use M3.

| p (M1-known) | Rule | Coverage | Precision | Precision given pairing |
|---|---|---|---|---|
| 0 (no exclusion) | d-7 | 77.9% | 0.570 | 0.603 |
| 0 | d-7 else d-1 | 91.0% | 0.555 | 0.590 |
| 0.10 (about T-3h) | d-7 else d-1 | 89.2% | 0.572 | 0.607 |
| 0.65 (about T-1h) | d-7 else d-1 | 74.7% | 0.716 | 0.752 |

(With d-7 only at p=0.10 / 0.65: 0.592 / 0.754.) The exclusion rule removes 3.8% (p=0.10) and 24% (p=0.65) of d-7 candidates, all of them wrong ones.
Mainline 0.547 vs regional 0.571 at p=0 (d-7 else d-1). The gain from exclusion at T-3h is small (about 2 points) because few tails are filed then.

## Implications for the plan

1. Make M2 a per-carrier allow-list from this table (precision >= 0.85 on BTS), UA excluded; re-check on live data (UA live may differ if the
   TFMS flight number is per leg).
2. Keep M3's default confidence at the measured 0.57 at T-3h and about 0.72 at T-1h (exclusion on), instead of one number.
3. The simulator (L1-sim) needs chains up to 4 links but they are rare; the dominant case is a single link.
4. The aircraft-type filter's gain is unmeasured; treat these precisions as a lower bound.

## Parity test

`tests/test_linkage_rot_parity.py`: synthetic fixture (160 tails, ties, cancellations, missing tails/times, fractional cutoffs, cutoffs after departure) and
a 2,000-row BTS sample (2025-06, skipped if data/bts is missing), `rot_live(strict_asof=False, true chain)` vs `rotation_features` at atol 1e-9, all nine features, T-3h and T-1h.
`strict_asof=True` with `cancel_recv` after the scheduled departure changes only `rot_known_delay`, `rot_leg_pending`, `rot_leg_cancelled`, `rot_projected_delay`, and only
on rows whose latest leg is cancelled; the other five features never move.
