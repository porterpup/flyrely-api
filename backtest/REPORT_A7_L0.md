# Report A7 + L0: value of knowing the inbound aircraft, and stability of rotations

Protocol: `PROTOCOL_A7_L0.md` (committed before any evaluation). Code: `backtest/a7_l0.py`, `flyrely_features/inbound.py`.
Results: `backtest/results/a7_l0.json`. Evaluation: 2026-01..08 (validation months, 4.87M flights), fit 2025-01..06,
calibration 2025-07..12. Nothing from 2026-09 on was loaded. Offline, $0.

## Verdict (plain)

1. **Knowing the inbound aircraft is worth a lot.** Perfect linkage cuts Brier by 0.031 at T-3h and 0.044 at T-1h
   (about 20% and 29% of the tail-free Brier), AUC +0.078 and +0.100. That is 60-90 times the 0.0005 "worth paying for"
   threshold. The roadmap kill condition does NOT trigger, so the AeroDataBox pilot is not ruled out by accuracy.
2. **Even the ~29% SFDPS fill helps**: gain 0.0086 (T-3h) and 0.0124 (T-1h), CIs far above 0. Roughly linear in the linked
   share: each extra 10 points of linkage is worth about 0.003 Brier at T-3h and 0.004 at T-1h.
3. **Paid tail data is not excluded by accuracy.** Going from 50% free linkage to perfect is still worth 0.0157 Brier
   at T-3h (CI 0.0153 to 0.0160), and 0.0079 from 75% to perfect. These are upper bounds (see caveat 1). Whether to pay
   depends on L1 (precision and earliness of the paid tails) and price, not on accuracy at stake.
4. **Rotation priors from last week's pairings do not meet the pre-registered bar** (L0: 57.4% vs 60% required), so rule 4
   says "not worth building" as a stand-alone route. They still recover 25% (T-3h) and 46% (T-1h) of the perfect gain with
   no tail at all, so they are a useful free fallback and complement, not a replacement.
5. **Recommended free routes, in order:** (a) SFDPS registration/icao24 (M1) and through-flight matching (M2) wherever
   they exist, because every linked point pays; (b) ADS-B icao24 to confirm and extend linkage once the inbound is
   airborne (matters most at T-1h); (c) rotation priors only as the fallback for flights with no other link. Train the
   production model with the missingness it will see (see mismatch results).

## A7: Brier, AUC, ECE (evaluation 2026-01..08)

Gain = Brier(N) - Brier(arm), positive is better, 95% day-block bootstrap CI. N = base without any rotation or inbound
features. P = B5.1 base with its existing `rot_*` features (perfect linkage). P+ = P plus the new inbound-state features.
Lp = flights linked with probability p, trained and calibrated with the same missingness. Mp = the P+ model served on
the same degraded data (trained on perfect linkage).

### T-3h

| Arm | Linked share | AUC | Brier | ECE | Gain (Brier) | 95% CI |
|---|---|---|---|---|---|---|
| N (no linkage) | 0 | 0.7482 | 0.15189 | 0.0017 | - | - |
| L0.25 | 25% | 0.7692 | 0.14457 | 0.0026 | 0.0073 | 0.0072 to 0.0075 |
| **L0.29 (SFDPS)** | 29% | 0.7727 | 0.14330 | 0.0026 | **0.0086** | 0.0084 to 0.0088 |
| **L0.5** | 50% | 0.7897 | 0.13685 | 0.0048 | **0.0150** | 0.0147 to 0.0154 |
| L0.7 | 70% | 0.8048 | 0.13069 | 0.0052 | 0.0212 | 0.0207 to 0.0217 |
| **L0.75** | 75% | 0.8087 | 0.12911 | 0.0056 | **0.0228** | 0.0223 to 0.0233 |
| L0.9 | 90% | 0.8191 | 0.12443 | 0.0055 | 0.0275 | 0.0269 to 0.0281 |
| P (existing rot features, perfect) | 100% | 0.8222 | 0.12281 | 0.0066 | 0.0291 | 0.0284 to 0.0297 |
| **P+ (perfect, plus inbound state)** | 100% | 0.8261 | 0.12120 | 0.0062 | **0.0307** | 0.0301 to 0.0314 |
| RG: mainline 29%, regional 0 (19.5% linked) | 19.5% | 0.7647 | 0.14568 | 0.0018 | 0.0062 | 0.0060 to 0.0064 |
| RG: mainline 100%, regional 0 (67.4% linked) | 67.4% | 0.8026 | 0.12994 | 0.0020 | 0.0220 | 0.0214 to 0.0225 |
| PR (prior pairing, no tail) | - | 0.7701 | 0.14431 | 0.0024 | 0.0076 | 0.0073 to 0.0079 |

### T-1h

| Arm | Linked share | AUC | Brier | ECE | Gain (Brier) | 95% CI |
|---|---|---|---|---|---|---|
| N (no linkage) | 0 | 0.7586 | 0.14861 | 0.0058 | - | - |
| L0.25 | 25% | 0.7874 | 0.13795 | 0.0066 | 0.0107 | 0.0104 to 0.0109 |
| **L0.29 (SFDPS)** | 29% | 0.7919 | 0.13616 | 0.0061 | **0.0124** | 0.0122 to 0.0127 |
| **L0.5** | 50% | 0.8140 | 0.12703 | 0.0067 | **0.0216** | 0.0211 to 0.0221 |
| L0.7 | 70% | 0.8333 | 0.11824 | 0.0082 | 0.0304 | 0.0297 to 0.0310 |
| **L0.75** | 75% | 0.8379 | 0.11602 | 0.0077 | **0.0326** | 0.0319 to 0.0333 |
| L0.9 | 90% | 0.8508 | 0.10933 | 0.0071 | 0.0393 | 0.0385 to 0.0401 |
| P (existing rot features, perfect) | 100% | 0.8556 | 0.10642 | 0.0079 | 0.0422 | 0.0413 to 0.0431 |
| **P+ (perfect, plus inbound state)** | 100% | 0.8587 | 0.10497 | 0.0076 | **0.0436** | 0.0427 to 0.0446 |
| RG: mainline 29%, regional 0 | 19.5% | 0.7794 | 0.14045 | 0.0058 | 0.0082 | 0.0079 to 0.0084 |
| RG: mainline 100%, regional 0 | 67.4% | 0.8258 | 0.11967 | 0.0029 | 0.0289 | 0.0282 to 0.0297 |
| PR (prior pairing, no tail) | - | 0.8103 | 0.12866 | 0.0050 | 0.0199 | 0.0193 to 0.0206 |

ECE stays below 0.01 in all serve-trained arms (N is 0.0017 / 0.0058), so calibration is not a concern.

Regional (contract-operated) flights are 32.6% of the eval flights; leaving them unlinked costs a lot (mainline-only at
100% = 67.4% linked: gain 0.0220 vs 0.0291 at full linkage; mainline at 29% yields only 0.0062).

### Gap to perfect (rule 3), Brier(level) - Brier(P+), serve-trained

| Free linkage level | T-3h | 95% CI | T-1h | 95% CI |
|---|---|---|---|---|
| 29% | 0.0221 | 0.0216 to 0.0226 | 0.0312 | 0.0305 to 0.0319 |
| 50% | 0.0157 | 0.0153 to 0.0160 | 0.0221 | 0.0216 to 0.0225 |
| 75% | 0.0079 | 0.0077 to 0.0081 | 0.0111 | 0.0108 to 0.0113 |

### Mismatch: train on perfect linkage, serve degraded (Brier worse than serve-trained)

| p | T-3h penalty | T-1h penalty | ECE (T-3h, mismatch) |
|---|---|---|---|
| 0.29 | +0.0032 | +0.0040 | 0.0245 |
| 0.50 | +0.0022 | +0.0027 | 0.0191 |
| 0.75 | +0.0010 | +0.0014 | 0.0126 |
| 0.90 | +0.0003 | +0.0006 | 0.0087 |

A model trained on perfect linkage and then fed partial linkage is miscalibrated (ECE 0.02+) and loses a third of the
gain at 29%. Production must train with the production missingness (and re-train as linkage improves).

### Prior-pairing arm (PR) hit rate (eval)

The predicted inbound is available for 88.4% of flights and its tail equals the flight's real tail for 56.9% of those
(50.3% of all flights). It is not a tail match that matters most: PR's inbound state carries information about the
inbound flight even when the aircraft differs (same hub bank, same delays), so PR gain is partly "network state at the
inbound leg", not aircraft identity.

## Decision rules (as pre-registered)

| Rule | Result |
|---|---|
| 1. Perfect linkage worth >= 0.0005 at T-3h? | **Yes**: 0.0307, CI lower bound 0.0301. Linkage is worth pursuing; AeroDataBox pilot not killed. |
| 2. SFDPS-level (29%) useful on its own? | **Yes**: 0.0086, CI 0.0084 to 0.0088. |
| 3. Paid tail data justifiable on accuracy at stake (gap from 50% to perfect >= 0.0005, CI > 0)? | **Yes**: 0.0157 (necessary, not sufficient; L1 decides). |
| 4. Rotation priors worth building (L0 d-7 >= 0.60 AND PR gain >= 0.0005 with CI > 0)? | **No**: L0 is 0.574 (fails the 0.60 bar), though PR gain is 0.0076 with a CI above 0. PR gain is 25% of perfect at T-3h (not "strong", which needs 50%); at T-1h it is 46%. |

## L0: how stable are rotations (2026-01..08, 4.87M flights with a pairing; 14.6% are through-flights)

Pairing = (inbound flight number -> this flight number at this origin), from BTS tails. "Match | exists" = same inbound
flight number as the earlier day, among flights with a pairing on that earlier day. Coverage = share of flights with a
pairing on the earlier day. "Hit over all" = match / all flights with a pairing today.

| Predictor | Coverage | P(match given exists) | Hit over all |
|---|---|---|---|
| Same flight, d-1 | 81.5% | 52.1% | 42.4% |
| **Same flight, same weekday d-7** | 83.0% | **57.4%** | 47.6% |
| Plurality of last 4 same weekdays | 89.0% | 55.7% | 49.6% |

Day-to-day (d-1) is worse than week-to-week (d-7): rotations are weekly-pattern driven (weekday vs weekend 57.6% vs 56.9% at d-7;
mainline 57.0%, regional 58.2%). Month-by-month d-7 stability ranged 53.7% (Feb) to 59.0% (Jul): stable over the year.

By origin hub (d-7 match given exists; d-1 in the JSON): RDU 76.1%, BNA 71.1%, LGA 49.5%, MDW 50.2%, DTW 48.2%,
MSP 43.3%, SEA 35.2%, DEN 34.0%, ATL 31.1%, CLT 26.4%, ORD 25.6%, DFW 22.7%. **Large hubs are the least stable**: banked
rotation systems pair arriving and departing flights differently every day. By carrier (d-7): WN 62.3%, F9 61.8%, 9E 61.8%,
OO 61.0%, DL 59.0%, YX 57.8%, AS 56.5%, B6 55.2%, MQ 55.1%, OH 54.1%, UA 52.3%, AA 47.8%.

Reading: priors work for point-to-point carriers and small airports, and fail exactly where pairing matters most (big
hubs). They can serve as a fallback prior (and the PR arm shows they still carry signal), not as the main linkage.

## Caveats (read before using the numbers)

1. **BTS tails are recorded after the fact.** Tail swaps and late aircraft changes mean the tail known at T-3h or
   T-1h can differ from the one that flew. All "perfect linkage" results are therefore **upper bounds**; the true gain of a
   linkage service with some swap error is smaller, by an amount this offline study cannot measure (L1 will). The
   PR arm, which uses no tail, is the one result free of this assumption.
2. The existing B5.1 base already contained these tail features; the N arm is new, so "gain" here is the value of
   linkage, not an addition to the B5.1 verdicts.
3. BTS parquet has no actual arrival times; "inbound has arrived" is an ETA estimate (actual departure + scheduled block).
   True arrival data (SWIM) should add a little more, and diversions are not modelled.
4. Random linkage (independent per flight) is optimistic about how well SFDPS fill spreads; real fill is correlated with
   carrier and airport (the regional-unlinked pattern shows the cost: 0.0062 vs 0.0086 at the same mainline rate).
5. Validation months only; 2026-09 holdout untouched. The levels are single-seed (seed 42); the CIs cover day-to-day
   sampling, not model-fit variance.

## Could paid tail data ever be justified?

On accuracy at stake, yes: closing the gap from a plausible free level (50%) to perfect would be worth up to 0.016 Brier
at T-3h and 0.022 at T-1h (upper bounds), 30 times the 0.0005 bar. So the roadmap's single kill condition does not
trigger and the $8 AeroDataBox pilot on L1 days stays on the table. It should still be judged on L1 (its precision,
whether its tails are known by T-3h, and the free union's effective linkage first: if SFDPS + M2 + ADS-B reach
about 75%, the remaining gap is 0.008 at T-3h). Run L1 on the free methods first.
