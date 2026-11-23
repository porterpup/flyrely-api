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
13. **(added 2026-11-09 with the evaluation code, before any run)** The connection engine always uses the fixed-calibration arm L for both legs' marginals, whichever arrival arm is served for `/predict`. Reason: the L-R walk-forward recalibration needs K = 3 earlier months of published predictions, which do not exist for the 2025-07..09 months where rho is fitted. If arm L fails C1 to C3, the outcome is "no connection numbers" as written; a recalibrated-marginals variant would need a new protocol.
14. **(same date)** "Missing ArrDelay" arrival rows and pairs whose leg 1 lacks ArrDelay are not misses (the missing delay is treated as 0); this affects well under 1.5% of rows and is the same for every variant and baseline.
15. **(added 2026-11-09 at the pre-run review, before any run on 2026 months)** P(miss) is P(miss | both legs operate), so every metric, fit and selection scored against `miss` (variant selection, the P(miss) isotonic recalibration, baseline B-L, diagnostic D, C1 to C4, C6, the reliability tables and the M +/- 10 sensitivity) uses the proxy pairs where both legs operated. P(disrupted) and rule C5 use every pair. Reason: a pair with a cancelled leg has `miss = 0` by definition, and scoring a conditional probability on it biases calibration downwards by roughly the pair cancellation rate (3% to 5% of pairs in late 2025). Disclosure: the reviewer had seen the 2025-only `--quick` smoke-test output (models already trained on those months) when making this change; no 2026 number has been seen.
16. **(same date)** Combined outcome. The copula P(miss) is built on the arrival ladder, so it is served only when the arrival outcome allows the ladder to be used: A1 to A3 pass, or A1 and A4 pass with A2 failing. In every other arrival outcome ("If A1 fails, nothing is shipped"), a connection outcome of "ship miss_probability" becomes "no connection numbers". The layover-table fallback B-L uses no ladder and is left as the C rules decide. The dependence file's `decision_outcome` is this final outcome.

---

## Amendment 1 (2026-11-22): recalibrated re-test on September 2026

Written and committed before the freeze stage, before any September arrival file exists, and before any September arrival outcome has been computed or shown. It adds to the sections above and rewords none of them, except where it says so (clarification 16, below).

### Trigger and approvals

- **Trigger.** The registered evaluation (run once on 2026-11-09, Jan to Aug 2026) failed A1 and A2, so nothing shipped. The cause was a level shift: arrival ECE at 15 minutes 0.041 against 0.02, 19.5% predicted late against 23.6% observed, and only 4 of 8 months within the monthly limit. See `backtest/REPORT_ARRIVAL_CONNECTIONS.md` and `backtest/results/arrival_connections.json`. The report said any follow-up needs a dated amendment with its reason and a decision from Sean, with no re-run before 2026-11-21.
- **Approvals.** Sean approved an amended recalibration re-run after 2026-11-20 on 2026-11-10 (together with a time-only tight-connection warning, which is display-only and changes no rule here), and approved the decisions D1 to D6 below on 2026-11-22.

### Decisions (Sean, 2026-11-22)

- **D1, arm L-S: a level-only recalibration.** For each arrival threshold t in the ladder (-30, -15, 0, 15, 30, 45, 60, 90, 120, 180, 300): `logit(p') = logit(p_L) + a_t`, slope fixed at 1, then the existing rearrangement step (sorting across thresholds).
  - `a_t` is the unweighted maximum-likelihood offset on the clarification-12 labels (flown flights with a recorded ArrDelay, diverted counted late), p clipped to [1e-6, 1-1e-6]. It is the root of sum(y - sigmoid(logit(p) + a)) = 0.
  - Nothing else is fitted. The ladder (`models/arrival_ladder_v1.joblib`), the 2025 isotonic shape, the production model and the connection model stay frozen.
- **D2: fit window.** The trailing six months before the evaluation month. For 2026-09 that is **2026-03..2026-08**. The rule is "the six calendar months immediately before the evaluation month", not a list of dates.
- **D3: evaluation month 2026-09, single run.** September 2026 was a protected holdout until it was spent at H1 (`PROTOCOL_B5_1_HOLDOUT.md`, the departure-model test). PR #70 moved the lock to start at 2026-10, so September is ordinary data and no unlock is needed; `FLYRELY_HOLDOUT_UNLOCK` is never set. September was chosen because it is the newest month that is not protected, it lies after every month used to fit or select anything (2025-01..2026-08), and it is a month on which no arrival result has been computed. October 2026 stays protected for D5.
- **D4: C confirmation.** If A passes, a take-away-only September confirmation of the connection model runs in the same run. The connection model stays as validated (R2 with isotonic recalibration, from `models/connection_dependence_v1.json`; arm L marginals per clarification 13).
  - C1, C2, C3, C5 and C6 at unchanged thresholds on September proxy pairs (same pair construction, `pairs.py --retest`).
  - C4 is read as a **point estimate**: the Brier difference against B-L must be below 0. The 95% day-block CI is reported, not gated.
  - If C1 to C3 or C6 fails: no copula numbers (`no_connection_numbers`, or `no_connection_numbers_c6_failed`), except for the B-L case in the next bullet.
  - If the C4 point estimate is 0 or above: the original C4 rule applies (the served number is the layover-table B-L, `itin-layover-table-v1`). C6 failing withholds the copula only; if the C4 point estimate is 0 or above, B-L is served as in the original rule.
  - The ordering is that of `connection_metrics`: C1-C4 and C6 pass: copula; C1-C3 pass and C4 fails: B-L; C1-C3 fail: no numbers; otherwise (C6 fails with C4 passing): `no_connection_numbers_c6_failed`.
  - B-L is the 2025-07..12 table of the original evaluation, stored in the freeze file. "A passes" means the re-test arrival outcome lets the ladder be used (see clarification 16 below). If A fails, the C confirmation is not run.
- **D5: October confirmation (rule recorded now, not implemented).** A take-away-only October 2026 confirmation in December, after the B5.1 October report merges and October is moved to the main data folders (the lock moves again). It needs its own freeze and check, a single scoring run, and the same rules as D4. This amendment adds no code for it.
- **D6: no more attempts on 2026 data.** If A fails, there are no further attempts on 2026 data. The next attempt needs a new protocol on 2027-01 or later data.
- **Both API flags stay off whatever the result.** Turning anything on is a separate decision by Sean after the report.

### Gates

A1 to A4 keep their thresholds and their code (`decide_arrival` in `backtest/eval_arrival_retest.py` copies the decision block of `arrival_evaluation` statement for statement; all measurement functions are imported from `eval_arrival_connections.py`, which is not edited). With one evaluation month the "monthly ECE" is that month's ECE; "ordered in every month" is that month's tier ordering. The climatology ladder is the 2025-01..06 ladder of the original evaluation, stored in the freeze file. The arm is L-S only: there is no arm choice, no L-R, and no second try with other offsets. Arm L is reported for reference and gates nothing.

### Clarification 16, amended

"Arrival outcome" in clarification 16 means the **re-test outcome** of this amendment. The copula P(miss) is served only if the re-test outcome is `ship_arrival_in_predict_and_headline` or `ladder_inside_connection_engine_only`. In every other re-test outcome a copula result becomes `no_connection_numbers_arrival_ladder_not_shipped`. The Jan-Aug arrival outcome no longer governs. Fallback B-L and "no numbers" are left as the C rules decide.

### Score-once rules

- The stages are `freeze`, `check` and `score` of `backtest/eval_arrival_retest.py`, in that order.
- **Freeze** fits a_t on 2026-03..08 and writes `backtest/results/arrival_retest_freeze.json`: the offsets, the window, the code-freeze commit (HEAD, clean tree), the SHA-256 of the four frozen files (below), the climatology ladder and the B-L table. It refuses if any September arrival file exists (`data/bts/parquet_arr/*2026-09*`, `data/cache/arrival/*2026-09*`), on a dirty tree, or if a freeze file already exists. The freeze file is committed before the September files are built.
- **September files** are built only after the freeze commit, with `prepare_arrival.py --retest`, `arrival_data.py --retest` and `pairs.py --retest`, each of which refuses unless the freeze file exists and is committed. The freeze stage also refuses unless the four model hashes equal the values listed below. They print row counts and no outcome rate.
- **Check** verifies 30 dates, shared columns equal to `parquet_v2/2026-09`, 630,177 rows, no date outside September, pairs built, and prints no outcome-based number. It writes a pass marker with file hashes.
- **Score** refuses: without the freeze file or the check marker; on any hash mismatch (frozen model files, freeze file, September data files); on a dirty tree; on code that differs from the freeze commit (git blob comparison of the frozen code list in the freeze file, which includes `flyrely_features/`, `flyrely_dist/`, `main.py`, the eval and data scripts, `backtest/holdout.py`); and when `FLYRELY_HOLDOUT_UNLOCK` is set. An attempt record (`arrival_retest_2026-09.attempts.json`) is written **before** any September outcome is read. September predictions are always recomputed in memory and never read from or written to a `pred_2026-09.npz`. A second run is refused. `--force-rerun` needs `--reason`, is for a bug fix only, writes `arrival_retest_2026-09_rerun<N>.json` and never overwrites the original; both results are then reported. The score stage never fits anything. Output: `backtest/results/arrival_retest_2026-09.json`.
- **Nothing reads September arrival outcomes or fits on September from now until the arrival score.** That covers every script and every person: no look at `ArrDelay`, no arrival late rate, no proxy miss rate for 2026-09.
- **Frozen and unchanged:** the files of the original protocol (`flyrely_features/`, `backtest/runup.py`, `backtest/backtest.py`, `backtest/experiments_b5.py`, `backtest/holdout.py`, `backtest/eval_arrival_connections.py`) are now checked against commit **b717936** (a diff against it must be empty). `backtest/holdout.py` changed in PR #70 as the policy-only lock move (the first protected month 2026-09 became 2026-10; 89be2a4 to b717936); it has not changed since. `arrival_data.py`, `pairs.py` and `prepare_arrival.py` change in this amendment only to admit 2026-09 behind an explicit `--retest` flag, with an unchanged default window (2025-01..2026-08) and no outcome printing; the original evaluation is unaffected.
- **Frozen model files** (hashes are written to the freeze file at the freeze stage and must equal these):
  - `models/arrival_ladder_v1.joblib`: `eec7f6b98e25f7da9450258fbb9f524b4e3cc91693491869cec9edec7fb8f53c` (gitignored; kept on the Mac mini)
  - `models/flight_delay_model.joblib`: `fa854d7561663b0202f43b05741fe57f92699f2e99baf150e9dcd327097a5bf7`
  - `models/connection_dependence_v1.json`: `0c513c96b8b73e034c0a663bf8f8c08e6b27c0b83e3f5bf6a66d09a42baad761` (the PR #57 artefact; identical to the `_final_dm` block of `results/arrival_connections.json`; committed with this amendment so that it can be pinned. Its `decision_outcome` is `no_connection_numbers_arrival_ladder_not_shipped`, so the API loader refuses it and the flags stay off)
  - `models/mct_defaults.csv`: `ea889882d2f3c54995cf00abb3c58da31be9fdae5ac06ec8f006822873a054dd`

### Disclosures

- **(a) H1 scored departure models on September.** The published H1 report (`REPORT_B5_1_HOLDOUT.md`) shows September row counts (630,177 evaluation flights) and departure ECE of about 0.011 to 0.022. Sean was also shown the departure booking-model decile calibration for September: 22.7% predicted against 22.4% observed late departures. That is departure only. No arrival late rate, arrival delay or proxy miss rate for September has been computed or shown.
- **(b) The amendment exists because of the Jan-Aug failure.** The offsets are fitted on 2026-03..08, months that include those the failed evaluation scored. September is the first month after them. The re-test is therefore a held-out test of a level correction chosen after seeing a failure, not a fresh pre-registration; this is why D6 allows no further attempt on 2026 data.
- **(c) The weekly schedule refresh** rebuilds `route_schedule` including September (schedules only; no metric, no delay outcome).
- **(d) Weather.** `data/weather` has no September 2 to 30 rows for AVL, GSO, HSV and ROC. The arrival ladder does not use weather: its inputs are the 18 `backtest.committed_features` (calendar fields, static airport and airline delay rates, hub flags, distance), and the monthly caches built by `arrival_data.py` contain no weather column. The four airports therefore do not affect the arrival ladder, the connection model or any gate here, and no weather is fetched for this amendment.
