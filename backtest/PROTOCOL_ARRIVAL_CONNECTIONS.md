# Pre-registered protocol: arrival delay and missed connections

Status: pre-registration, committed before any evaluation is run. Source: `backtest/PLAN_ARRIVAL_AND_CONNECTIONS.md` section 4 (verbatim below), with Sean's decisions D1-D5 and the frozen lists filled in. Registered 2026-11-09.

## Decisions recorded (Sean, 2026-11-09)

- **D1 approved as recommended.** Default minimum connect times (MCT): 30 min at airports under 100 daily departures, 35 min at 100-400, 40 min at the top 30, 50 min at the large-campus list below. Users can override per connection. Source label `flyrely_default_estimate`.
- **D2: show the percent only.** No "about 1 in N". The display bands (< 5%, 5-15%, >= 15%) are kept as labels and for rule C6.
- **D3 decided (Sean, 2026-11-09).** For cross-airline (separate-ticket) connections, hide the percentage and show a plain warning instead, e.g. "Separate airlines: if your first flight is late, the second won't wait. Allow at least 2-3 hours." This is **display-only**: it changes no model, metric, split or decision rule in this protocol. Proxy pairs are same-network only, so cross-airline connections are not evaluated in any case.
- **D4 approved.** The ladder artifact is gitignored, with a sha256 manifest, and kept on the Mac mini.
- **D5 approved.** Arrival becomes the headline for direct flights once A1-A3 pass.

## Frozen lists (2025 BTS departures only, `parquet_v2` 2025-01..12)

Top-30 hub list, by 2025 departures (ORD 357,302 down to PDX 75,159):
ORD ATL DEN DFW CLT PHX LAX LAS SEA IAH LGA MCO SFO DCA BOS DTW EWR MSP JFK PHL SLC MIA BNA SAN BWI IAD AUS FLL TPA PDX

Large-campus MCT list (50 min): ATL ORD DFW DEN LAX JFK EWR IAH SFO CLT

Airport size classes for MCT use 2025 departures / 365: under 100 per day = 30 min; 100 to under 400 = 35 min; top 30 = 40 min; large-campus list = 50 min (list wins).

Frozen and unchanged files are checked against commit 89be2a4.

---

**Scope and data.**
- Input: BTS Marketing Carrier data, read through `data/bts/parquet_arr` (built by `prepare_arrival.py` from the same zips as `parquet_v2`; shared columns must equal `parquet_v2` exactly).
- Months: 2025-01..2026-08 only. Nothing from 2026-09 on is read, downloaded or converted. `FLYRELY_HOLDOUT_UNLOCK` is never set, and every new script exits if it is set.
- Frozen and unchanged: `flyrely_features/`, `backtest/runup.py`, `backtest/backtest.py`, `backtest/experiments_b5.py`, `backtest/holdout.py`, `backtest/fetch_faa_adv.py`. They may be imported, never edited.

**Splits.**
- Model weights: 2025-01..06.
- Isotonic calibration: 2025-07..12.
- Dependence ρ and any P(miss) recalibration are fitted on proxy pairs from 2025-07..09. Variants are selected on 2025-10..12. Then the chosen variant is refitted on 2025-07..12.
- Evaluation: 2026-01..08 (validation). Split for the one arrival-arm choice: V-select = 2026-01..04, V-confirm = 2026-05..08.

**Model.** `backtest.fit_calibrated` (HGB, seed 42, unchanged hyperparameters) per threshold. Features: the 18 `backtest.committed_features` (operating carrier). Thresholds and labels as in section 2.1. Rearrangement by sorting. Piecewise-linear interpolation. Fit-window tail tables.

**Arrival arms.** L (fixed calibration) and L-R (walk-forward logistic recalibration per threshold, K = 3 months ≤ m−3). L-R is served only if both hold:
- its Brier score at t = 15 on V-select is lower than L's;
- on V-confirm, the 95% day-block bootstrap CI (1,000 resamples, seed 42) of Brier(L-R) − Brier(L) at t = 15 lies entirely below 0.

Otherwise L is served.

**Arrival metrics** (flown flights, chosen arm, 2026-01..08):
- AUC, Brier, ECE (equal-count deciles, `bt.ece_and_deciles`) and maximum decile gap at t ∈ {0, 15, 45, 120};
- monthly ECE and AUC at t = 15;
- tier ordering (0.20 / 0.30) every month;
- ranked probability score (RPS, the sum of Brier scores over the ladder) against a climatology ladder from the fit window;
- central 80% interval coverage [q10, q90] and q50 coverage;
- share of rows needing rearrangement;
- consistency with the production model: mean |S_dep(15) − p_production|.

A "without NK" line is reported. AUC is reported, not a gate: about 0.67 is expected, the same as for departure.

**Arrival decision rules** (all at t = 15 unless stated):
- **A1 Calibration:** ECE ≤ 0.02 and maximum decile gap ≤ 0.04; monthly ECE ≤ 0.04 in every month; ECE ≤ 0.03 at t = 45 and t = 120.
- **A2 Skill:** Brier lower than baseline B-D (the production departure probability used as the arrival forecast), with a 95% day-block CI entirely below 0; and lower than climatology.
- **A3 Tiers:** observed rate high ≥ 1.8 × low, and low < medium < high in every month.
- **A4 Distribution:** RPS skill against climatology > 0 with the CI excluding 0; q10–q90 coverage within 0.80 ± 0.04.
- **Outcome:**
  - Arrival goes into `/predict` and the app headline only if A1–A3 pass.
  - If A1 and A4 pass but A2 fails, the ladder is used **only inside the connection engine**, and `/predict` gets no arrival block.
  - If A1 fails, nothing is shipped and the report recommends next steps.

**Connection variants.**
- R0, R1, R2 (section 2.2), each with and without a final isotonic recalibration of P(miss).
- Selection: lowest Brier on 2025-10..12 proxies.
- Diagnostic only (never shipped from this protocol): direct HGB classifier D, trained on 2025-07..09 pairs with features [L, M, hub, network, hour, month, S₁(s), S₂(0)].

**Connection metrics** (2026-01..08 proxies, chosen variant):
- Brier, log loss, AUC, ECE (equal-count deciles), maximum decile gap;
- reliability by layover bucket, by hub (top 10), by month, and for same versus different operating carrier;
- the same metrics for P(disrupted).

Baselines:
- B-L: empirical miss rate by hub × 5-minute layover bucket from 2025-07..12;
- B-I: R0 with the same marginals.

Sensitivity (report only): M ± 10 min.

**Connection decision rules:**
- **C1:** ECE ≤ 0.015; maximum decile gap ≤ 0.03 (top decile ≤ 0.05).
- **C2:** monthly ECE ≤ 0.03 in every month.
- **C3:** every layover bucket with n ≥ 10,000 has |mean predicted − observed| ≤ 0.02.
- **C4:** Brier lower than B-L, with the 95% day-block CI entirely below 0.
- **C5:** P(disrupted) ECE ≤ 0.02. If C5 fails, the trip-level number is not shown; per-connection numbers can still be shown.
- **C6:** the observed miss rate is ordered across the display bands (< 0.05, 0.05–0.15, ≥ 0.15) in every month.
- **Outcome:**
  - Ship `miss_probability` for validated connections only if C1–C4 and C6 pass.
  - If C4 fails while C1–C3 pass, the report says the model adds nothing over layover length. The served number is then B-L, labelled `model_version: "itin-layover-table-v1"`, with copy "based on how often connections this short at DEN were missed". This is a real BTS statistic chosen by a written rule, not an invented number.
  - If C1–C3 fail, no connection numbers are shown.

**Outputs.** `backtest/results/arrival_connections.json`, `backtest/REPORT_ARRIVAL_CONNECTIONS.md`, and the artifacts `models/arrival_ladder_v1.joblib`, `models/connection_dependence_v1.json`, `models/mct_defaults.csv`. The run is done once. Any re-run after a bug fix is reported as an amendment, with the reason.

---

## Clarifications and disclosures (2026-11-09, added before any evaluation code ran on 2026 months)

These add detail the protocol left open. They change no threshold, split or decision rule above.

1. **Pair buckets (section 3 of the plan).** Written as `[M, M+30), [M+30, M+60), [M+60, 120), [120, 240]`, implemented literally in `backtest/pairs.py` (for M = 40: 40-70, 70-100, 100-120, 120-240). Checked against intent: the plan wants tight connections oversampled, and the second bucket cannot be `[M+30, 60)` because that is empty for every M >= 30. The literal reading is the only coherent one, so no change is needed. Because every metric can be reweighted by the bucket weight `w`, the choice does not bias the reweighted numbers.
2. **Disclosure.** While building the proxy pairs (T2, 2026-11-09), `pairs.py` printed the unweighted proxy miss rate per month for all 20 months, including the 2026 evaluation months (about 7% to 10% in every month). No design choice, threshold or rule depended on those numbers, and they were not examined beyond the log line.
3. **Primary population.** Gating metrics (C1 to C6) use the stratified proxy sample as drawn (unweighted). Metrics reweighted to the all-pairs layover mix (by `w`) are reported next to them, not gated. Dependence rho, the P(miss) isotonic recalibration and baseline B-L are fitted unweighted.
4. **Layover buckets for C3** are the three used for rho (< 60, 60-120, 120-240 min). The four sampling buckets are reported, not gated.
5. **Bootstrap.** All "95% day-block bootstrap" intervals resample whole scheduled-arrival days (arrival date of leg 1 for pairs, flight date for legs), 1,000 resamples, seed 42, percentile interval.
6. **PIT for rho:** delays are integers; the PIT is randomised uniformly within [F(x), F(x+1)) with seed 42; cancelled legs and diverted leg 1 are excluded from the rho fit.
7. **B-L:** empirical miss rate by hub x floor(layover / 5) from the 2025-07..12 proxy pairs; an unseen cell falls back to that layover bucket's overall rate across hubs.
8. **C6** requires the observed rate to be non-decreasing across the non-empty display bands in every month, and strictly lower in the lowest than the highest band.
9. **M +/- 10 sensitivity** is recomputed (and the label rebuilt with the shifted M) on a seeded 10% sample of the 2026 pairs.
10. **--quick mode** reads 2025 months only (evaluation on 2025-11..12, which the models have already seen, so the numbers are a smoke test and never a result), writes `arrival_connections_quick.json`, and uses a 5% pair sample.
11. **P(miss) is never evaluated or served for through flights**, which are not in the proxy pairs.
12. **Arrival label** for metrics: flown flights with a recorded ArrDelay, diverted counted as late at every threshold. Flights flown but missing ArrDelay (not cancelled, not diverted) are dropped.
