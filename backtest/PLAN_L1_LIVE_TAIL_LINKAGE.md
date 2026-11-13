> Written by the `planner` agent on 2026-11-07. Saved by the main session, because the planner ran read-only.

# Plan L1: live tail linkage (which plane flies each flight, and how its previous flight is doing)

Status: plan only. No code changed, no network calls made. Written 2026-11-07 by `planner`.
Inputs read: HANDOFF.md, REPORT_A7_L0.md, PROTOCOL_A7_L0.md, PLAN_ROADMAP_2026-11.md (sections 2, 3, 4.6–4.7, 5), REPORT_A5B_OPENSKY_HUBS.md, swim/ (consumer, filters, archive, retention), codeshare.py, opensky_fetcher.py, backtest/opensky_client.py, backtest/opensky_coverage.py, models/route_schedule.csv.gz, copilot/status.py, copilot/schedule.py, flyrely_features/rotation.py, backtest/inbound.py, backtest/runup.py, deploy/macmini/.

**Data measured:** about 32 minutes of 2026-11-07 (13:27–13:59 UTC) for all three feeds, plus 10-minute TFMS and SFDPS samples from the 2026-11-05 smoke test. Measurement was read-only.

**Three key points:**
1. **Tail fill is not the problem; timing is.** Every SFDPS flight-plan message for the target carriers carried the registration and icao24. That was 6,800 of 6,800 messages on 11-07 and 859 of 859 on 11-05. The earlier "28.7%" figure was an artefact of a 10-minute sample window. The real limit is that a flight's own plan, and so its tail, first appears only about 55–190 minutes before takeoff (median about 90). So the tail is known for most flights at T-1h, but for only about 5–12% at T-3h.
2. **Spirit (NKS) and Hawaiian (HAL) do not appear in any of the three feeds,** although the filter would keep them. This needs checking (task T1).
3. **The archive is deleted after 7 days.** The first full day (11-08) is lost around 11-15 unless the retention change in task T0 lands first. It also needs Sean to raise the disk budget from 20 GB to about 30 GB.

## Summary for Sean (plain English)

- **What we're trying to do.** About 3 hours and 1 hour before a flight, our best predictor is "where is the plane that will fly it, and is it running late?" To answer that we need to know which plane (its tail number) flies each flight. In testing, knowing this cut prediction error by about 20–30%. Today the live app has no source for it.
- **What the free FAA feed shows.** I looked at the real messages we've been recording since this morning. The plane's tail number is in the FAA flight plan for almost every airline flight (about 99.9% of flight-plan messages). The catch is timing. Airlines file the flight plan only about 1 to 2½ hours before takeoff. So:
  - **1 hour before departure:** we'll usually know the plane directly (our estimate is roughly 6 or 7 flights in 10). For the rest we can often work it out.
  - **3 hours before departure:** we'll rarely know it directly (about 1 flight in 10). Mostly we'll have to infer it from patterns: the same flight number arriving earlier, and which plane usually does this flight on this weekday.
- **The other sources.**
  - The FAA "big picture" feed (TFMS) never names the plane.
  - The airport-surface feed (STDDS) names it only about 30 minutes before departure. That is too late to predict with, but useful for checking our answers.
  - OpenSky (free radio tracking) only sees a plane once it is flying. It is a cross-check, not a main source.
- **The plan.**
  1. Build a small free service on the Mac mini that reads the FAA feeds every minute and keeps a running list of every flight and its plane.
  2. Have it work out each flight's incoming plane, with a confidence label.
  3. Record what it would have said 3 hours and 1 hour before every departure.
  4. When the official government records for November arrive (mid-January), grade those answers and measure how much they improve predictions.
- **Spending.** $0. No scraping. The model-freeze rule is respected. The privacy list (LADD) is applied everywhere, and tail numbers never leave the Mac.
- **What I need from you.**
  - OK to use about 30 GB of disk instead of 20 GB. The Mac has 59 GB free.
  - Confirm whether Spirit and Hawaiian still fly under their own call signs. Neither appears in the FAA data.
  - The FAA privacy list. This is already on your to-do list.

---

## 1. Measured coverage

### 1.1 What was measured (and what wasn't)

| Source | Data inspected | Notes |
|---|---|---|
| SFDPS | 11-07 13:27–13:59 UTC (filtered archive, 71,650 messages received, 12,633 kept) and the 11-05 smoke file 19:20–19:30 UTC | Morning US bank only; one weekday plus a Saturday. |
| TFMS | 11-07 13:27–13:59 UTC, 93,451 archived messages | |
| STDDS | 11-07 13:27–about 13:59 UTC, first 60,000 messages | |
| OpenSky | **Not measured in this plan.** No polling was done. | Figures quoted are from A5b (one weekday, 12 hubs). |

Full-day figures, other weekdays, and evening banks are not measured yet; task T1 measures them.

### 1.2 Per-field facts (measured)

- **SFDPS:**
  - `registration` and `aircraftAddress` (icao24) appear on flight-plan messages (`FH`, `AH`, `HU`), in both PROPOSED and ACTIVE status. Among airline callsigns, 100.0% of these messages carried registration in both samples (6,800 of 6,800 and 859 of 859).
  - Departure messages (`DH`) carry no registration.
  - icao24 is present whenever registration is.
  - In 3,362 registrations, no registration ever appeared with two different icao24s.
  - About 2% of registrations are non-N (foreign-registered).
- **Swaps:** within the 32-minute window, 1 of 3,901 flights showed two different registrations. Swaps that happen before filing cannot be seen here.
- **Plan creation time:** this is embedded in `FDPS_GUFI` (`us.fdps.<creation time>…`), so lead time can be measured without censoring.
- **TFMS:**
  - 0 of 93,451 messages contain `registration`, `aircraftAddress`, `icao24`, `REG/` or `tailNumber`.
  - It does carry the schedule about 24 h ahead (`FlightScheduleActivate` and `FlightCreate`, with `igtd`, airline OUT/OFF/ON/IN times, and `aircraftModel`).
  - It carries CDM airline times (`cdmPart="true"`) for AAL, UAL, DAL, SWA, ASA and JBU.
  - It carries an ERAM `gufi`.
- **STDDS:**
  - SMES `SurfaceMovementEventMessage` (runway on/off, spot in/out) carries `callsign`, `acAddress` (icao24) and `sfdpsGufi`. That was 100% of 1,167 callsigns at 41 airports.
  - TDES `TDLSCSPMessage` (pre-departure clearance text) contains `/AN <registration>` in 51% of PDCs (68 airports). The lead time is about 30 minutes before the proposed time (deciles 22–30 min).
  - TDES `TowerDepartureEventMessage` gives the gate and clearance time.
  - The SMES and TDES `enhancedData` blocks carry both `eramGufi` and `sfdpsGufi`. This is a cross-walk between feeds.
- **Privacy flags:** all sampled SFDPS messages had `FDPS_BlockAtFAA=false` and `FDPS_Sensitive=false`. 167 messages had `FDPS_Restricted=D`, whose meaning is unknown and should be asked of the FAA. STDDS carries `blockAtFAA`.

### 1.3 Per carrier (measured, 11-07 window unless noted)

Column definitions:
- **"Deps with reg by OFF":** airline flights whose actual wheels-off fell in 13:28–13:59 UTC and that had a registration in any SFDPS message. This is inflated by the plan messages SFDPS re-sends around departure. The 11-05 10-minute window gave 47% on the same measure, because plan messages fell before the window. **The full-day fill is not measured** (T1).
- **"Plan lead":** the median minutes from plan creation to actual OFF, for those departures. n is in brackets.

| Carrier | ICAO | SFDPS deps with reg by OFF | Plan lead median (n) | STDDS PDC with reg (n) | SMES icao24 on surface events | TFMS tail |
|---|---|---|---|---|---|---|
| AA | AAL | 0.99 (93) | 91 min (59); 11-05: 90 (18) | 0.39 (93) | 100% (all callsigns) | 0% |
| DL | DAL | 0.95 (102) | 61 (34) | 0.75 (104) | 100% | 0% |
| UA | UAL | 0.94 (99) | 70 (98); 11-05: 68 (15) | 0.72 (72) | 100% | 0% |
| WN | SWA | 0.89 (101) | 149 (85); 11-05: 165 (25) | 0.79 (111) | 100% | 0% |
| B6 | JBU | 0.95 (19) | 170 (13) | 0.75 (20) | 100% | 0% |
| AS | ASA | 0.80 (5); P(reg in plan msg) 0.98 | too few | 0.83 (18) | 100% | 0% |
| NK | NKS | **not observed in any feed** | – | – | – | – |
| F9 | FFT | 0.92 (12) | 77 (9) | 0.00 (9) | 100% | 0% |
| HA | HAL | **not observed in any feed** (HAL is in the filter's carrier list) | – | – | – | – |
| G4 | AAY | 0.91 (11) | 57 (9) | – | 100% | 0% |
| Regionals: SKW, RPA, ENY, EDV, PDT, JIA, GJS, ASH, UCA, MXY | | 0.88–1.00 | SKW 78–85, ENY 93, RPA 73, JIA 83, EDV 61 | 0.00 (all regionals) | 100% | 0% |

Plan-creation lead before OFF, all airline departures:

| Sample | Deciles (min) | ≥60 min | ≥120 min | ≥180 min |
|---|---|---|---|---|
| 11-07, n=440 | 57 / 63 / 73 / 81 / 89 / 98 / 111 / 140 / 173 | 0.85 | 0.28 | 0.08 |
| 11-05, n=129 | 56 / 68 / 78 / 84 / 93 / 103 / 125 / 160 / 192 | 0.86 | 0.31 | 0.12 |

**Reading.** The registration is in nearly every flight plan, but a plan exists only about 1–3 h before takeoff. WN and B6 file early; DL, UA and G4 file late. The as-of times are relative to scheduled gate departure, and OFF is typically about 15–20 min after that. So the lead relative to the T-1h and T-3h as-of times is about 15–20 min shorter than the table shows.

### 1.4 What combining sources could reach (ESTIMATED, not measured)

Effective linkage = coverage × precision against the BTS tail. Assumptions:
- M1 availability comes from the lead distribution above, shifted by 15–20 min.
- M1 precision is 0.95–0.98 (swaps after filing are unmeasured).
- M2 (through-flight) covers about 14.6% of pairings (from L0), with precision about 0.85 (to measure on BTS, T4).
- M3 (weekday prior) covers 0.83–0.89 with precision 0.57 (L0, BTS), possibly 0.60–0.70 with an aircraft-type filter and exclusions (to measure).
- ADS-B adds identity only after the plane is airborne. SFDPS already has the inbound leg's tail once that leg is filed, so ADS-B adds at most about 0–3 points (filling non-ERAM gaps such as some Hawaii or international inbounds).
- STDDS gives tails only about 30 min ahead, so it adds about 0 at T-1h. It is used as truth and confirmation.

| Horizon | M1 (own plan reg) | +M2 (through-flight) | +M3 (prior) | Free union, effective | High-confidence share (M1+M2) |
|---|---|---|---|---|---|
| T-1h | 0.55–0.70 | +0.03–0.05 | +0.12–0.20 | **≈0.70–0.85** | ≈0.58–0.74 |
| T-3h | 0.05–0.10 | +0.11–0.13 | +0.35–0.45 | **≈0.50–0.65** | ≈0.16–0.23 |

By carrier, M1 at T-1h is roughly:
- WN about 0.9;
- B6 about 0.9;
- AA about 0.65–0.75;
- SKW and ENY about 0.55;
- UA about 0.4;
- F9 about 0.5;
- DL about 0.3;
- G4 about 0.2.

Small n, rough.

**Mapping to accuracy (estimate from the A7 curves).**
- **T-1h:** a 0.6–0.75 high-confidence share gives roughly 0.026–0.033 of the 0.044 perfect-linkage gain.
- **T-3h:** most links are prior-based. The A7 PR arm (prior pairing, no tail) alone gave 0.0076, so expect roughly 0.008–0.013 of the 0.031 perfect gain.

These are upper-bound-style estimates. Prior links are often wrong, and A7's L_p arms modelled only missing links, not wrong ones. The A7 roadmap guess of "29% SFDPS fill" understates full-day fill and overstates T-3h availability.

### 1.5 Measured, full day (2026-11-09, 11-11, 11-12)

Values are read from `backtest/results/l1_fill_2026-11-09.json`, `l1_fill_2026-11-11.json` and `l1_fill_2026-11-12.json`, rounded to 2 decimals. Legs are scheduled legs per day. Column names follow the JSON keys.

| Carrier | 11-09 legs | 11-09 fill | 11-09 kn60 | 11-09 kn180 | 11-09 pdc+reg | 11-09 smes | 11-11 legs | 11-11 fill | 11-11 kn60 | 11-11 kn180 | 11-11 pdc+reg | 11-11 smes | 11-12 legs | 11-12 fill | 11-12 kn60 | 11-12 kn180 | 11-12 pdc+reg | 11-12 smes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AAL | 1889 | 0.94 | 0.49 | 0.00 | 0.33 | 0.97 | 1441 | 0.96 | 0.58 | 0.00 | 0.32 | 0.97 | 1681 | 0.91 | 0.47 | 0.00 | 0.30 | 0.95 |
| DAL | 2537 | 0.94 | 0.01 | 0.00 | 0.54 | 0.96 | 2391 | 0.95 | 0.01 | 0.00 | 0.53 | 0.96 | 2460 | 0.90 | 0.01 | 0.00 | 0.51 | 0.93 |
| UAL | 2118 | 0.91 | 0.01 | 0.00 | 0.50 | 0.94 | 2048 | 0.92 | 0.01 | 0.00 | 0.50 | 0.94 | 2105 | 0.88 | 0.01 | 0.01 | 0.46 | 0.92 |
| SWA | 4186 | 0.94 | 0.46 | 0.02 | 0.65 | 0.78 | 3613 | 0.94 | 0.47 | 0.02 | 0.62 | 0.77 | 3804 | 0.92 | 0.46 | 0.03 | 0.58 | 0.75 |
| JBU | 549 | 0.96 | 0.57 | 0.06 | 0.65 | 0.95 | 457 | 0.97 | 0.58 | 0.09 | 0.65 | 0.93 | 466 | 0.86 | 0.52 | 0.05 | 0.61 | 0.86 |
| ASA | 676 | 0.61 | 0.21 | 0.00 | 0.44 | 0.92 | 570 | 0.58 | 0.19 | 0.00 | 0.37 | 0.90 | 650 | 0.58 | 0.16 | 0.00 | 0.39 | 0.89 |
| FFT | 423 | 0.97 | 0.00 | 0.00 | 0.00 | 0.99 | 316 | 0.96 | 0.00 | 0.00 | 0.00 | 0.98 | 339 | 0.92 | 0.00 | 0.00 | 0.00 | 0.94 |
| AAY | 261 | 0.87 | 0.02 | 0.00 | 0.00 | 0.36 | 66 | 0.83 | 0.02 | 0.00 | 0.00 | 0.30 | 223 | 0.88 | 0.00 | 0.00 | 0.00 | 0.33 |
| regionals_all | 5951 | 0.93 | 0.28 | 0.00 | 0.00 | 0.93 | 5802 | 0.92 | 0.27 | 0.00 | 0.00 | 0.92 | 5867 | 0.87 | 0.24 | 0.00 | 0.00 | 0.87 |

- **Alaska fill.** Alaska (ASA, including Hawaiian) fill is only about 0.58–0.61 on all three days.
- **Registration lead times cluster at fixed values.** DL and UA plans appear at most about 45–49 min before departure, and AA at about 74 min. This is a filing-system ceiling, so M1 (own plan) cannot get earlier. Forward chaining (S3, `linkage/PROTOCOL_CHAIN.md`) is the route to T-3h.
- **The 11-12 dip.** Overall domestic fill fell from 0.92 (11-09 and 11-11) to 0.88 (11-12). The loss is concentrated in UTC hours 16–23 and in the Northeast: JFK 0.96→0.73, EWR 0.97→0.75, LGA 0.96→0.77, PHL 0.95→0.79. Carriers RPA, EDV, JBU and PDT were most affected. The median change across airports was only −3.5 points. The 2026-11-13 SWIM outage (09:49–21:50 UTC) does not explain it, because 11-12 legs need data only until about 11-13 01:00 UTC. The likely cause is an operational or upstream effect in the NYC/PHL area that evening (unconfirmed). Measured with `scripts/l1_measure_fill.py --breakdown` (PR #63).
- **Exclude 2026-11-13** from linkage evaluation because of the outage.

---

## 2. Design

### 2.1 Approaches weighed

| | A. Separate incremental extractor into a local SQLite live store (chosen) | B. Parse inside the consumer, or hourly batch only |
|---|---|---|
| How | A new launchd service tails the growing `.jsonl.zst` hour files every 60 s, parses them, and upserts SQLite (WAL). A nightly freeze writes parquet. | Add parsing to `swim/consumer.py` (A), or a cron that parses closed hour files (B). |
| Pros | The consumer's ack/fsync path is untouched. Restartable: it re-reads the hour files from the start. Latency is about 1–2 min. The same parser runs live and in replay. | Fewer moving parts. |
| Cons | One more service. | Parsing inside the consumer risks the FAA-mandated ack path. Hourly batch gives up to 60 min staleness, which is useless at T-1h. |

Pick **A**.

Training approach: **simulate L1 on BTS** ("L1-sim": replay method availability and wrong links using measured live lead and precision), rather than A7-style random missingness. Live linkage is structured by carrier, hour and lead, and includes wrong links. A7 showed a mismatch costs a third of the gain.

### 2.2 Pipeline

```
SWIM consumers (unchanged) ──► data/swim/<src>/<date>/*.jsonl.zst  (raw, retention per source)
                                   │ every 60 s, byte-offset tail, as-of = archived `recv`
                                   ▼
linkage.extract  (launchd com.flyrely.linkage.extract)
  SFDPS: per GUFI → callsign, origin/dest, status, estimated/actual runway times, reg, icao24, first_recv per fact
  TFMS : schedule universe (FlightCreate/ScheduleActivate: igtd, airline OUT/OFF/ON/IN, aircraftModel, ERAM gufi),
         FlightTimes actuals (OUT/OFF/ON/IN with recv), cancellations, ETA/ETD updates
  STDDS: SMES runway on/off + acAddress + sfdpsGufi; TDES PDC /AN reg; DE gate/clearance
  LADD + FAA block flags applied here (drop the leg and all its facts)
                                   ▼
data/linkage/live.sqlite  (WAL; API and snapshotter open read-only)
                                   ▼
linkage.engine.resolve(leg, as_of) → LinkResult(tail_key, method, confidence, chain[], leg_at_c)
                                   ▼
linkage.snapshot (launchd com.flyrely.linkage.snapshot, every 5 min):
  for every population leg with sched OUT in [now+3h, now+3h+5m) and [now+1h, now+1h+5m) → write link_snapshots
                                   ▼
nightly 09:30 UTC: freeze yesterday → data/linkage/daily/<date>/*.parquet (kept indefinitely; about 5–10 MB/day est.)
                   rebuild pairing priors (own archive: d-7, d-1, plurality of last 4 same weekdays when ≥28 days)
```

- **As-of rule (leakage):** every stored fact carries `first_recv`, the archived receive time. The engine only uses facts with `first_recv <= as_of`. Live snapshots are logged in real time as well, so both "replay" and "as it really was" exist, and they must agree (test).
- **Population universe for chains:** legs in the TFMS airline schedule (FlightCreate/ScheduleActivate), both ends US domestic (BTS universe, territories included), with a carrier in the BTS reporting set or its regional partners. Ferry, charter, international and re-filed duplicate plans are excluded, or merged on (callsign, origin, dest, igtd). This is a parity rule: BTS rotations contain only these legs.
- **Flight identity from the user's ticket:**
  1. `codeshare.resolve()` gives the operating carrier and number.
  2. A new IATA→ICAO callsign table (`linkage/carriers.py`: AA→AAL, OO→SKW, MQ→ENY, 9E→EDV, OH→JIA, YX→RPA, QX→QXE, G7→GJS, PT→PDT, YV→ASH, C5→UCA, MX→MXY, and so on) gives the callsign.
  3. Match on (callsign, origin, dest, igtd date) with igtd within ±90 min of the schedule time.
  4. Cross-feed join: SFDPS GUFI ↔ ERAM gufi via STDDS `enhancedData`, else via (callsign, origin, dest, date).

### 2.3 Flight-to-tail matching rules and confidence

For departing leg B at as-of time c:

| Rule | Condition (all facts with first_recv ≤ c) | Tail / inbound | Initial confidence (re-estimated in T4/T8, then frozen in PROTOCOL_L1) |
|---|---|---|---|
| **M1 direct** | B's own SFDPS plan carries reg R (latest amendment ≤ c wins) | tail = R. Chain = legs whose plans carry R, ordered by sched OUT, within 18 h before B | 0.95 (target ≥0.95 vs BTS) |
| **M2 through-flight** | Same callsign has an earlier leg A arriving at B's origin, sched OUT within 18 h before B, same TFMS aircraft type family, and A not claimed by another departure via M1 | inbound = A. Tail = A's reg if A is filed, else unknown-but-linked | per carrier, from BTS (T4); default 0.85 |
| **M3 prior** | Own-archive pairing for (B's flight number, origin, weekday): d-7, else d-1, else plurality of the last 4 same weekdays once ≥28 days exist. Candidate A must be scheduled today, same type family, and not claimed via M1/M2 by another departure | inbound = A | per carrier × airport tier hit rate (L0 BTS, then own archive); default 0.57 |
| **none** | – | rot_* and in_* missing (NaN) | – |

- **Priority:** M1 > M2 > M3. A later M1 overrides a prior link, and the change is logged.
- **Exclusion:** a candidate inbound whose tail is already known (M1) to fly a different departure is removed. This is the free precision gain over plain L0 priors.
- **Chains:** rot_* need "the latest leg of this aircraft scheduled to have left by c" plus the legs between it and B (`rot_legs_between`, block time, slack). Walk backward from B, link by link, using the same rules, until a leg with sched OUT ≤ c (max 4 links, 18 h). Chain confidence is the product of link confidences, stored per snapshot. When the tail is known (M1), take the chain from that reg's filed legs. Fall back to rule-linking for any unfiled intermediate leg.
- **ADS-B and STDDS roles:**
  - OpenSky icao24 by callsign confirms an airborne inbound and fills plan gaps. The N-number↔icao24 mapping uses `backtest/opensky_coverage.n_number_to_icao24`.
  - SMES on/off events give "inbound landed" (actual ON) at about 41 airports.
  - The STDDS PDC `/AN` reg is used as late confirmation and interim truth only.
  - Neither is needed for the main path.

### 2.4 Storage (SQLite `data/linkage/live.sqlite`, gitignored, local only)

| Table | Key columns |
|---|---|
| `legs` | leg_id (callsign\|origin\|dest\|igtd_date), carrier_icao, flight_no, origin, dest, sched_out_utc, sched_in_utc, aircraft_type, sfdps_gufi, eram_gufi, tfms_flight_ref, out/off/on/in actual + each *_recv, cancelled + cancelled_recv, first_seen_recv |
| `leg_aircraft` | leg_id, source (sfdps\|pdc\|smes\|opensky), reg, icao24, first_recv, last_recv. Several rows if the value changes (swap audit) |
| `surface_events` | airport, callsign, icao24, event, event_time, recv, sfdps_gufi |
| `link_snapshots` | leg_id, horizon, as_of, method, confidence, tail_key, inbound_leg_id, leg_at_c_id, chain_json, engine_version, store_lag_s (staleness at as_of) |
| `pairing_prior` | carrier, flight_no, origin, weekday, inbound_flight_no, n_days, share, source (own\|bts), built_on |

- Nightly parquet freeze of the same tables is kept indefinitely; BTS truth arrives 7–11 weeks later.
- Raw retention change is needed (T0). Raw SFDPS is kept 35 days and raw TFMS 14 days until the extractor has back-filled, so parser bugs can be fixed by re-extracting.
- **Privacy:**
  - `Blocklist` (swim/filters.py) is re-applied at extraction on callsign, reg and icao24.
  - Any message with `FDPS_BlockAtFAA`/`FDPS_Sensitive`/`blockAtFAA` = true is dropped.
  - When the LADD file's mtime changes, a scrub job deletes matching legs from SQLite and parquet. A one-off scrub of raw and derived data runs when the list first arrives.
  - Registrations and icao24 never leave the Mac. They are never returned by any endpoint, never logged in API logs, and never copied to prediction logs; `tail_key` there is a salted hash.
  - Test fixtures use synthetic registrations only.

### 2.5 How the API consumes it, and parity risks

- **Consumer:** `linkage/query.py` (read-only SQLite) exposes `inbound_state(operating_callsign, origin, dest, sched_out_utc, as_of)`. It returns the LinkResult plus the live leg states needed by rot_*.
- **First user:** the roadmap's run-up shadow scheduler (P3). **`/predict` is not changed by L1.** The booking model stays as it is.
- **Later:** when run-up serving launches (P4), the run-up path computes rot_* via `linkage/rot_live.py` and feeds a model trained under L1-sim missingness. The response contract is unchanged. A risk factor such as "the inbound flight is running about 40 min late" may be added later via the existing `risk_factors` list.
- **No-fallback rule:** a flight with no link gets missing rot_* by design. This is the trained model's own input path, not a fallback. If the live store is stale (`store_lag_s` above 600 s, or swim.health reports SFDPS or TFMS down), the run-up path must not predict as if "unlinked": it returns 503 `prediction_failed` for that horizon. This needs Sean's and P0's confirmation; the booking `/predict` is unaffected.
- **Code parity:**
  - `linkage/rot_live.py` computes the nine rot_* from an explicit chain. With `strict_asof=False` and the BTS true chain, it must equal `flyrely_features.rotation.rotation_features` exactly (frozen code is imported, never edited).
  - With `strict_asof=True` (cancellation visible only if received ≤ c), it is the version used for L1-sim training and serving.

Feature-parity risks: BTS-derived training vs inferred live.

| # | Risk | Mitigation and check |
|---|---|---|
| 1 | Tail identity: BTS records the tail that flew; live uses the plan-time or inferred tail. Swaps are biased toward delayed flights (A7 caveat 1). | Measure M1 precision vs BTS November (E-L1). L1-sim injects wrong links at measured rates. |
| 2 | Leg universe: SWIM includes international, ferry and re-filed legs; BTS doesn't. This changes `rot_legs_between` and slack. | Universe rule in 2.2. Test: on BTS-truth days, the share of chains with an extra non-BTS leg is ≤2%. |
| 3 | Scheduled times: BTS CRS vs TFMS igtd/airlineOutTime; elapsed vs (in − out). | V1-style check when BTS November lands: median \|igtd − CRS\| = 0 and ≥95% within 5 min. |
| 4 | Actual departure: BTS gate OUT vs SWIM OUT (CDM carriers only) or OFF − taxi median. | Same adapter as roadmap P1 network; its parity test applies. |
| 5 | Cancellations: frozen `rotation.py` uses the hindsight cancel flag. | `strict_asof` in training (new module). Report the difference. |
| 6 | Clock: B5.1 uses the Eastern-reference local clock with per-day offsets. | Convert UTC with the same zoneinfo offsets. Unit test on DST days. |
| 7 | Missingness pattern: random in A7, structured live. | L1-sim calibrated on live calibration days. Mix check below. |
| 8 | Wrong links: absent in A7 L_p arms. | L1-sim includes them. E-L1 measures the realized gain. |

**Train/serve check (pre-registered, gate before E-L1 scoring):**
- (a) Exact code parity as above.
- (b) Simulator mix: share of method ∈ {M1, M2, M3, none} by horizon × carrier group × local hour band. L1-sim on BTS same-weekday days vs live snapshots on calibration days. Every cell with n ≥ 200 must be within ±0.05, and the simulated M3 hit rate within ±0.05 of the live interim hit rate.
- (c) Feature distribution: per rot_* feature, population stability index (PSI) < 0.10 between simulated-on-BTS and live-snapshot features on eval days (informational, reported).

---

## 3. Pre-registered offline evaluation (draft of `backtest/PROTOCOL_L1.md`, committed before any eval day is scored)

Two parts:
- **L1-A:** how good the linkage is.
- **E-L1:** how much it is worth with real live linkage.

Both are offline, $0, and use no 2026-09+ BTS beyond what H1 already unlocks. Nothing from frozen paths is edited.

### 3.1 Days

- **Calibration days:** 2026-11-08 … 11-14. Used for lead distributions, simulator calibration and confidence values. Never scored.
- **Eval days:** 2026-11-16 … 11-24 and 11-30 … 12-13, 23 days. Thanksgiving (11-25 … 11-29) is excluded and reported separately as information.
- **Day replacement:** a day is excluded if the SFDPS or TFMS archive has a gap over 15 min between 11:00 and 03:00 UTC, or the snapshotter misses more than 5% of its ticks. If more than 5 days are lost, extend with 12-14 … 12-20 in order.
- **Minimum to score:** 14 valid days.

### 3.2 Population

Departures from the top-100 airports (frozen BTS-August list) by the 10 named carriers and their regional partners, scheduled 13:00–03:00 UTC so that T-3h falls in the collection window, matched to a BTS row when BTS lands. Strata:
- carrier;
- mainline vs regional;
- airport tier (12 A5b hubs, ranks 1–30, 31–100).

### 3.3 L1-A metrics (per method, free union, horizon, stratum)

- **Coverage:** share of the population with a link.
- **Precision:** the linked inbound equals the true inbound, on flights with truth.
- **Effective linkage** = coverage × precision.
- **Chain-exact:** leg_at_c and legs_between equal to truth.
- **Truth:**
  - **Interim:** icao24 at B's OFF from SMES (41 airports) or OpenSky (callsign within 30 km and ±20 min of OFF). This is physical, independent of flight plans.
  - **Final:** the BTS November and December tail.
  - M1 precision is decided on the final truth only; interim truth shares no source with M2/M3 but is near-tautological for M1.

### 3.4 E-L1 arms (T-3h and T-1h, same flights)

All features come from BTS as in B5.1 (`runup.horizon_features`, frozen code, read-only), except the rot_* (and in_* for P+) features.

| Arm | Linkage | Model |
|---|---|---|
| N | none | trained without rot/in features (as A7 N) |
| **L1** | live snapshot chain identity; leg states from BTS at c, strict as-of | trained and calibrated with L1-sim (fit 2025-01..06, cal 2025-07..12, seed 42) |
| PR | live M3 prior only | trained with the same prior simulation |
| P | BTS true tail (upper bound) | the A7 P model |
| L1-mismatch | live chain | frozen perfect-linkage model (information) |

Models are frozen with SHA-256 hashes in `data/linkage/models/manifest.json` before the first eval day's BTS data is downloaded.

### 3.5 Metrics and decision rules (fixed now)

Metrics: Brier, AUC and ECE per `runup.score`. G(arm) = Brier(N) − Brier(arm), with a 95% day-block bootstrap (1,000 resamples, seed 42), `runup.brier_diff`.

1. **Free linkage viable (roadmap 3.5):** L1-A free-union effective linkage at T-1h ≥ 0.60 over the top-100 on final truth.
2. **M1 trusted:** M1 precision at T-1h ≥ 0.95 → keep M1 at confidence 0.95. If 0.90–0.95, use the measured value. If < 0.90, investigate swap timing before any serving.
3. **Adopt live linkage for run-up serving (P2/P3):** G(T-3h, L1) ≥ 0.0005 with CI lower bound > 0. T-1h is reported and decided the same way, independently per horizon.
4. **Swap-bias check:** compare G(L1) with the A7 curve interpolated at the measured effective linkage. If realized < 0.5 × predicted, "swap bias material": report it and do not use A7 curves for future value estimates.
5. **Serve-trained beats mismatch:** report Brier(L1-mismatch) − Brier(L1). If > 0, the L1-sim-trained model is the one shipped.
6. **Paid tails:** report Brier(L1) − Brier(P) as the remaining accuracy at stake. No spend decision is triggered; spending is Sean's call.
7. **Gate:** if the section 2.5 train/serve checks (a) or (b) fail, E-L1 is not scored. Fix, then re-run the calibration days and append a protocol amendment before scoring.

Everything is reported whatever it shows. No re-tuning after seeing eval-day outcomes.

### 3.6 Outputs

- `backtest/l1_eval.py`
- `backtest/results/l1_a_interim.json`, `l1_a_final.json`, `e_l1.json`
- `backtest/REPORT_L1.md`

---

## 4. Tasks

**Holdout freeze (applies to every task):** no task edits `flyrely_features/`, `backtest/runup.py`, `backtest/backtest.py`, `backtest/experiments_b5.py`, `backtest/holdout.py` or `backtest/fetch_faa_adv.py`. Importing them read-only is fine. Each PR's acceptance includes `git diff --stat 89be2a4 -- <those paths>` being empty.

Test command for every task: `cd flyrely-api && venv/bin/python -m pytest -q` (the full suite must pass), plus the task's own tests.

### T0 [executor] Retention for L1 (do by 11-09; urgent)

- In `swim/retention.py` set `DEFAULT_SOURCE_DAYS = {"stdds": 1, "sfdps": 35, "tfms": 14}`.
- Update the comment in `deploy/macmini/install_macmini.py` with the new disk estimate (about 27 GB peak).
- Add tests in `tests/test_swim_ops.py`: per-source days are honoured, and the newest day is never deleted.
- Needs Sean's OK on about 30 GB.
- **Accept:** `python -m swim.retention --dry-run` lists no SFDPS day under 35 days old; tests pass.

### T1 [executor] Full-day fill measurement (run on 11-09 for UTC day 11-08; repeat for 11-10 and 11-11)

- New script `scripts/l1_measure_fill.py` (read-only over archives; no reg is printed).
- For each of the 10 carriers and the regionals, output:
  - share of TFMS-scheduled legs with any SFDPS reg;
  - first-reg-receive lead vs igtd: deciles, and shares ≥60/≥180 min;
  - SMES and PDC coverage;
  - counts of NKS and HAL callsigns, and of HA/NK flight numbers under other callsigns (e.g. ASA).
- Writes `backtest/results/l1_fill_<date>.json`.
- **Accept:** the JSON has all carriers. A unit test on a synthetic 3-message fixture checks the lead computation. Numbers are pasted into section 1 as "measured, full day".

### T2 [implementer] STDDS filter

- In `swim/filters.py`, the stdds path keeps SMES `SurfaceMovementEventMessage`, TDES `TowerDepartureEventMessage`/`TDLSCSPMessage`, and status messages.
- It drops `asdexMsg` position reports and `SafetyLogicHoldBar`. Together these were about 92% of messages in the sample.
- LADD and block flags still apply.
- Then set stdds retention to 14 days.
- **Accept:** tests on recorded synthetic fixtures; a replay on the 11-07 file shows ≥85% byte reduction (via `swim/replay_filter.py`); disk estimate updated.

### T3 [implementer] Extractor and store

- New files: `linkage/__init__.py`, `linkage/carriers.py`, `linkage/extract.py`, `linkage/store.py`.
- New launchd template `deploy/templates/launchd-linkage.plist.tmpl`; add `com.flyrely.linkage.extract` to `install_macmini.py`.
- Back-fill command: `python -m linkage.extract --backfill 2026-11-07`.
- Parsing, tables and rules as in sections 2.2–2.4. Byte-offset tailing; on restart, re-read the current hour's files.
- **Accept:**
  - Tests in `tests/test_linkage_extract.py` (synthetic XML fixtures for SFDPS FH/AH/DH/CL, TFMS Create/Times, SMES SE, TDES PD/DE) check:
    - fields are parsed;
    - first_recv is monotonic;
    - LADD drops a leg by reg, icao24 or callsign;
    - the `BlockAtFAA` drop works;
    - a re-filed plan is merged;
    - the non-schedule ferry leg is excluded from the chain universe.
  - A live smoke: store lag < 120 s for 1 h.
  - The back-fill for 11-08 reproduces T1's counts within 1%.

### T4 [implementer] Offline calibration on BTS and code parity (BTS validation months 2026-01..08 only)

- `linkage/rot_live.py` (rot_* from an explicit chain, `strict_asof` flag).
- `backtest/l1_bts_calib.py` (new): M2 precision by carrier; chain-length distribution at T-3h/T-1h; M3 precision with the type filter and exclusion rule.
- Output: `backtest/results/l1_bts_calib.json`.
- **Accept:** `tests/test_linkage_rot_parity.py`:
  - On a synthetic fixture and on a data-backed sample when data/bts exists, `rot_live(strict_asof=False, true chain)` equals `rotation_features` on 2,000 rows at atol 1e-9.
  - `strict_asof=True` differs only on cancelled legs.

### T5 [implementer] Engine, priors, snapshotter

- `linkage/engine.py`, `linkage/priors.py`, `linkage/snapshot.py`; service `com.flyrely.linkage.snapshot` (5-min ticks); nightly freeze and prior rebuild.
- **Accept:** `tests/test_linkage_engine.py` with a synthetic day covers:
  - M1 > M2 > M3 priority;
  - exclusion of a claimed inbound;
  - chain walk across 2 links;
  - the as-of rule (a fact received 1 s after c is ignored);
  - replay-vs-live equality on a recorded hour;
  - a LADD-listed tail never appears in snapshots;
  - `tail_key` is hashed.
- Running for 24 h logs snapshots for ≥95% of population legs at both horizons.

### T6 [executor] Commit `backtest/PROTOCOL_L1.md`

- Section 3 verbatim, plus the confidence values frozen from T4 and the calibration days.
- Commit before 11-16 00:00 UTC.
- **Accept:** the commit timestamp predates the first eval day; the reviewer checks it.

### T7 [implementer] OpenSky interim-truth arm (evaluation only)

- `linkage/opensky_arm.py` reuses `backtest/opensky_client.py`: CONUS `/states/all` every 180 s, 11:00–03:00 UTC.
- That is about 1,280 credits/day, keeping the 1,500-credit floor and stopping on a 429.
- It stores only (callsign, icao24, time, position, on_ground) for airline callsigns into `leg_aircraft` and `surface_events`, applying LADD.
- Service `com.flyrely.linkage.opensky`, active 11-15 … 12-13 only.
- **Accept:** budget tests (floor, 429 stop) with a mocked client; the credits ledger shows ≤1,400/day.
- Note: this is evaluation-only, under the OpenSky licence question (not for production).

### T8 [implementer] L1-sim and simulator calibration

- `linkage/simulate.py` replays M1/M2/M3 availability and errors on BTS rows. It uses:
  - lead distributions per carrier × hour from the calibration days;
  - M2/M3 precision from T4;
  - swap rate (initially 0.02; updated from final truth for future refits only).
- `backtest/l1_mixcheck.py` runs check 2.5(b).
- **Accept:** tests confirm the simulator is deterministic (seed 42) and uses only as-of facts; the mix check passes, or an amendment is written before 11-16.

### T9 [implementer] Train and freeze E-L1 models (after H1 scoring, about 11-21)

- `backtest/l1_train.py` (new) trains N, L1, PR and P for T-3h/T-1h. Splits are as A7 (fit 2025-01..06, cal 2025-07..12, evaluated on validation months for a sanity check). Writes the manifest with hashes.
- **Accept:**
  - The N and P validation Brier reproduce A7 (0.15189/0.12281 at T-3h, 0.14861/0.10642 at T-1h) within 0.0005.
  - The manifest is committed before the November BTS download.

### T10 [implementer → reviewer] L1-A interim, then final; E-L1 scoring

- `backtest/l1_eval.py interim` (after 7 eval days, about 11-25, SMES/OpenSky truth); `final` (BTS November about mid-Jan 2027, December about mid-Feb 2027); `score` (E-L1, once ≥14 valid days have BTS).
- Writes `REPORT_L1.md` with the decision rules applied as written.
- **Accept:**
  - Tests on a synthetic population cover the metrics and bootstrap call.
  - Refuses to score if the manifest hashes differ or the gate checks failed.

### T11 [implementer] Read-only query module (for P0/P3; no `/predict` change)

- `linkage/query.py` and the staleness signal (`store_lag_s`). Extend `swim.health --status` with linkage-store freshness.
- **Accept:** tests show a stale store raises `LinkageStale`, and that the query never returns a registration or icao24.

### T12 [executor] Docs

- HANDOFF section "L1", the new launchd labels, and an update to the LADD scrub procedure in `deploy/LADD_BLOCKLIST.md`.
- **Accept:** doc-only diff.

Typical flow: each implementer task goes to a PR and then to `reviewer`. Executor tasks go straight to a PR.

---

## 5. Timeline (today is Sat 2026-11-07)

| Dates | Step | Data needed |
|---|---|---|
| 11-07 … 11-09 | T0 (urgent: first full day 11-08 is otherwise deleted about 11-15), T1 on 11-08 | 1 full UTC day |
| 11-09 … 11-13 | T2, T3 (back-fill from 11-07), T4 (offline, BTS only) | archive since 11-07 |
| 11-08 … 11-14 | Calibration days (lead distributions) | 7 days |
| 11-12 … 11-15 | T5 engine and snapshotter running by 11-15. d-7 priors first available 11-15 (7 prior days) | 7 days before first eval day |
| 11-15 | T6 protocol commit, T8 mix check, T7 OpenSky arm on | calibration days |
| 11-16 … 12-13 | Eval days (23 planned; Thanksgiving 11-25 … 11-29 excluded). Plurality-of-4 priors usable from 12-06 (28 days) | continuous |
| 11-17 … 11-20 | B5.1 holdout (H1). Nothing in L1 touches frozen paths | – |
| about 11-21 | T9 train and freeze E-L1 models (needs H1 done so the September unlock is closed) | BTS ≤ 2026-08 |
| about 11-25 | T10 interim L1-A (first 7 eval days, SMES/OpenSky truth) | 7 eval days |
| about 2027-01-15…20 | BTS November lands: L1-A final for 11 November eval days; E-L1 preliminary (below the 14-day minimum, reported as interim) | BTS Nov |
| about 2027-02-15…20 | BTS December lands: E-L1 final (≥14 valid days), REPORT_L1 | BTS Dec |
| then | Roadmap P2/P3 refit with L1-sim and shadow mode, if rule 3 adopts | – |

Effort: about 9–12 implementer-days and about 1.5 executor-days. Cost $0.

---

## 6. Decisions for Sean

1. Disk budget from 20 GB to about 30 GB (59 GB free) for longer raw SFDPS/TFMS retention.
2. Spirit (NK) and Hawaiian (HA): neither callsign appears in any FAA feed. Has Spirit stopped flying, and does Hawaiian now fly as Alaska (ASA)? Hawaii inter-island coverage by SFDPS is also unknown.
3. The LADD list (existing to-do 6). Also ask the FAA what `FDPS_Restricted="D"` means.
4. A stale live store should make the run-up horizon return 503 rather than predict as "no link". This is to confirm at P0.

## 7. Not verified

- Full-day fill. Evening and weekday lead times.
- Whether SFDPS publishes a plan at creation for every carrier. The PROPOSED messages measured carried registration 100%, but publication timing was only seen for flights first created inside the window.
- Swap rate between filing and departure.
- Whether the SFDPS `flightPlan identifier` equals the ERAM `gufi` used by TFMS and STDDS.
- Whether international inbound legs get SFDPS plans before entering US airspace.
- M2 precision (T4).
- OpenSky identity rates outside the A5b day.
- The commercial OpenSky licence.

### Critical Files for Implementation

- `swim/filters.py` (regexes, Blocklist/LADD, STDDS path to filter)
- `swim/archive.py` (read_lines, record format with `recv`)
- `flyrely_features/rotation.py` (frozen rot_* definition to match exactly; read-only)
- `backtest/inbound.py` (in_* features, Pairings/prior logic to reuse)
- `swim/retention.py` and `deploy/macmini/install_macmini.py` (retention and new launchd services)

Other relevant paths:
- `codeshare.py`
- `backtest/opensky_client.py`
- `backtest/opensky_coverage.py` (`n_number_to_icao24`)
- `copilot/status.py`
- `backtest/PLAN_ROADMAP_2026-11.md`
- `backtest/REPORT_A7_L0.md`
