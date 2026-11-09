# Arrival delay and missed connections: evaluation report (draft tables)

Protocol: `backtest/PROTOCOL_ARRIVAL_CONNECTIONS.md`. Run on 2026-11-09, artifact sha256 `eec7f6b98e25f7da9450258fbb9f524b4e3cc91693491869cec9edec7fb8f53c`. Evaluation months ['2026-01', '2026-08']. This file is generated from `results/arrival_connections.json`; prose is added at review.

## Plain summary

The registered evaluation ran once on 2026-11-09 (946 s, 4.6 GB peak memory) on Jan to Aug 2026. **Nothing from it ships: no arrival number and no missed-connection number is shown to users.**

- **Arrival (A1 to A4).** The arrival model fails calibration (A1) and is not better than today's departure number used as an arrival forecast (A2). It passes tier ordering (A3) and distribution skill (A4). By the pre-registered rule, arrival is not shipped and direct flights keep the departure headline (D5 does not trigger).
- **Why A1 fails: the level shift.** The model predicts that 19.5% of flights arrive 15+ minutes late; 23.6% actually did. Predictions run about 4 points too low across 2026, the same shift seen for departures in earlier work. ECE at 15 minutes is 0.041 (limit 0.02), the largest decile gap 0.059 (limit 0.04), and only 4 of 8 months meet the monthly limit (March is worst at 0.073). The walk-forward recalibration arm (L-R) cuts ECE to 0.021, still just over the limit, and it did not win the pre-registered arm choice, so arm L was evaluated as the served arm. The model still beats a constant climatology forecast (A2 against climatology passes) and ranks flights as expected (AUC about 0.647), but today's departure probability scores a Brier 0.0031 lower, mostly because it is better centred.
- **Connections (C1 to C6).** The connection model itself passed every rule: ECE 0.008 (limit 0.015), monthly ECE at most 0.017, every layover bucket within 0.02, Brier below the layover-table baseline (difference -0.00107, CI entirely below 0), trip-level ECE 0.017, and the display bands ordered in every month. The rule outcome was `ship_miss_probability_for_validated_connections`.
- **Final outcome.** Clarification 16 (approved before the run) withholds connection numbers unless the arrival ladder is allowed to ship, because the connection model is built on the arrival ladder. Arrival failed A1 and A2, so the final outcome is `no_connection_numbers_arrival_ladder_not_shipped`. This was accepted as scored; the rule is not relaxed after seeing the result.
- **How much the copula adds.** The gain over simple baselines is small: Brier 0.0755 against 0.0765 for the layover table and 0.0755 for the independence variant. The proxy pairs are not booked passengers (no gates, terminals or airline-protected connections).

Any follow-up (for example a tighter recalibration arm, or a calibration month closer to the evaluation period) needs a new dated protocol amendment with its reason before it is run, and a decision from Sean. No re-run before 2026-11-21.

## Arrival
Served arm: **L**. Outcome: **ship_nothing_arrival**.

- A1: FAIL
- A2: FAIL
- A3: PASS
- A4: PASS

| arm | t | AUC | Brier | ECE | max decile gap |
|---|---|---|---|---|---|
| L | 0 | 0.627 | 0.2300 | 0.0442 | 0.0508 |
| L | 15 | 0.647 | 0.1735 | 0.0414 | 0.0589 |
| L | 45 | 0.657 | 0.1013 | 0.0264 | 0.0417 |
| L | 120 | 0.655 | 0.0378 | 0.0091 | 0.0149 |
| L-R | 0 | 0.616 | 0.2304 | 0.0233 | 0.0755 |
| L-R | 15 | 0.635 | 0.1731 | 0.0210 | 0.0466 |
| L-R | 45 | 0.642 | 0.1011 | 0.0140 | 0.0361 |
| L-R | 120 | 0.640 | 0.0378 | 0.0052 | 0.0132 |

## Connections
Variant: **R2+iso**. Outcome (C rules): **ship_miss_probability_for_validated_connections**; after the arrival gate: **no_connection_numbers_arrival_ladder_not_shipped**.

- C1: PASS
- C2: PASS
- C3: PASS
- C4: PASS
- C5: PASS
- C6: PASS

Pairs 8,168,828; observed miss rate 0.0895 (reweighted 0.0759). Brier 0.0755, ECE 0.0080, AUC 0.747. B-L Brier 0.0765.

