# Arrival delay and missed connections: evaluation report (draft tables)

Protocol: `backtest/PROTOCOL_ARRIVAL_CONNECTIONS.md`. Run on 2026-11-09, artifact sha256 `eec7f6b98e25f7da9450258fbb9f524b4e3cc91693491869cec9edec7fb8f53c`. Evaluation months ['2026-01', '2026-08']. This file is generated from `results/arrival_connections.json`; prose is added at review.

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

