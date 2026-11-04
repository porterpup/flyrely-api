# A5: Is the free OpenSky API good enough for T-3h / T-1h features?

Run: 2026-11-03, BTS month 2026-08 only (holdout untouched). Code: `backtest/opensky_coverage.py`; numbers: `backtest/results/a5_opensky_coverage.json`.

## Verdict

**The free `/flights/*` archive is not usable as a feed (about 25% of flights). The live `/states/all` feed looks much more promising, with caveats, and no paid source is justified yet.**

- **`/flights/*` (historical archive):** OpenSky has a record for 26% of our flights on the departure side, 25% on the arrival side, 22% for both. The cause is not established (it could be receiver coverage, how the archive assigns airports, or something else). When a record exists it is very accurate: times track wheels-off/wheels-on to about 0.5 minutes (median). Arrivals also only appear after a nightly batch, so this endpoint cannot feed T-3h/T-1h features live regardless.
- **`/states/all` (live state vectors):** a different path with much higher coverage. Over 40 CONUS snapshots (6-minute spacing, 3 hours 54 minutes of US evening), the number of airline-callsign aircraft airborne was about **95% of what BTS-derived schedules imply** (mainline 96%, regional 93%; per-carrier values 0.7 to 1.2). Treat this as "near-complete, roughly 85-100%", because the denominator (August schedules re-timed to a November weekday) is only good to roughly +/-10-15%. Data age is about 1 second at the median.
- **Open weak spot:** whether a flight is seen near its origin around departure (what a "departed yet / pushed back" feature needs). By my flight-level measure only 46% of flights that were seen at all were seen within 25 km of their origin in the departure window. That mixes real ground-reception gaps with the 6-minute sampling gap, so it is a lower bound that this experiment cannot separate.
- **Credits:** a CONUS states feed fits the 4,000/day limit at a 2 to 5 minute cadence (2,880 or 1,152 credits/day). It lives in its own bucket, so it does not compete with `/flights`.
- **Paid source:** not established as needed. The gap to test next is departure/pushback detection at the hubs with finer sampling (1-credit airport boxes at 1-2 minutes). If that stays below roughly 90% at hubs, that is the gap a paid source (such as AeroAPI) would need to fill: near-complete departure and arrival actuals by flight, within minutes, plus inbound aircraft status. I did not research pricing.

## Sample and credits

- 6 UTC days (Aug 4, 8, 12, 16, 20, 25), 72 two-hour `/flights/all` windows, 375,908 unique OpenSky flights. 97,889 completed BTS flights in scope (cancelled/diverted excluded, flights well inside the day edges).
- **Credits spent: 2,415 total.** `/flights` bucket: 2,250 (ended at 1,750 remaining; floor was 1,500). `/states` bucket: 165 including the 40-snapshot poll (ended at 3,835; the cap for this follow-up was 600).
- The latency probes (KATL departure/arrival, one recent `/flights/all` window, the 2x2 degree and first CONUS states calls) were one-off ad-hoc calls through the same client, not scripted in `opensky_coverage.py`; they are time-relative, so a rerun would not reproduce them. Their raw responses are in the local cache and numbers are in the JSON `live_probes` and ledger. Only the `/flights/all` windows and the state-vector poll are reproducible commands.
- Costs observed: every `/flights/*` call (all, departure, arrival) = **30 credits**, regardless of how recent; `/states/all` = 1 credit for a 2x2 degree box, **4 credits for all of CONUS**. Matches the docs.

## 1. Match rate (OpenSky flight found for a BTS flight)

| Definition | Rate |
|---|---|
| Callsign + time only | 29.3% |
| Departure side (callsign + origin airport + within 60 min of wheels-off) | **25.8%** |
| Arrival side (callsign + dest airport + within 60 min of wheels-on) | **24.8%** |
| Both on one record | 21.5% |
| Route + time only, no callsign | 23.4% |

Callsign candidates were operating carrier + number and marketing carrier + number (e.g. EDV5123 and DAL5123).

- **By carrier (departure side):** regionals match better than mainline. PT 52%, G7 47%, 9E 44%, C5 41%, YX 39%, QX 38%, OH 36%, MQ 35%, YV 33%, OO 33%. Mainline: WN 25%, DL 22%, AA 17%, F9 16%, UA 16%, AS 16%, G4 15%, B6 12%.
- **By origin airport tier:** top-30 25%, ranks 31-100 28%, other 25% (departure side). By destination tier, arrival-side matches are top-30 25%, ranks 31-100 25%, other 21%. No tier is good.
- **By hour:** ranges 12% to 44% with no clean pattern; no hour is above 50%.
- Hub examples (departure side): DTW 45%, CLT 41%, ATL 34%, BOS 18%, SEA 17%, IAH 17%.

## 2. Time accuracy (matched flights)

| Comparison | n | Median bias | Median abs | P90 abs |
|---|---|---|---|---|
| firstSeen vs BTS wheels-off | 25,242 | +0.5 min | 0.5 min | 1.1 min |
| lastSeen vs BTS wheels-on | 24,258 | +0.2 min | 0.4 min | 1.0 min |
| firstSeen vs BTS off-block (gate out) | 25,242 | +17 min | 17 min | 32 min |
| lastSeen vs BTS gate-in | 24,258 | -6.7 min | 6.7 min | 17 min |

OpenSky's times are runway events, not gate events. They are nearly exact against wheels-off/on, and offset by taxi time against gate times.

## 3. Delay-flag fidelity (>=15 min late)

OpenSky-derived delay = firstSeen (or lastSeen) minus the scheduled time. Bias offset was fitted on the first 3 sampled days and scored on the last 3.

| Flag | Matched flights only | All flights (unmatched counted as "not late") |
|---|---|---|
| Departed >=15 late, raw | precision 0.39, recall 1.00 (flags too much: firstSeen is about 17 min after off-block) | n/a |
| Departed >=15 late, taxi-corrected | precision 0.69, recall 0.93 | precision 0.76, recall **0.23** |
| Arrived >=15 late, raw | precision 0.999, recall 0.79 | |
| Arrived >=15 late, corrected | precision 0.96, recall 0.90 | precision 0.97, recall **0.20** |

Delay correlation with BTS is 0.98 (departure) and 0.99 (arrival). When OpenSky has the flight, the signal is good (arrival especially). Departure precision is weaker because taxi time varies by airport, so a per-airport taxi-out estimate would help. Over all flights, recall is about 20-23% because of the coverage gap.

## 4. Inbound-aircraft linkage

- N-number to icao24 is practical and needs no registry download. I implemented the standard algorithm (`n_number_to_icao24`): 98.7% of BTS tails are computable, and it agrees with OpenSky's icao24 on **99.2%** of matched flights (the rest are probably tail swaps or reassigned codes).
- 64.6% of scoped flights have a same-day inbound leg in BTS (63,289 rotations).
- But OpenSky observed the inbound arrival (by icao24, correct arrival airport, within 60 min) for only **28.3%** of rotations, and both the inbound arrival and this departure for **11.8%**.
- When observed, inbound time is accurate (median 0.4 min) and the inbound >=15-late flag has precision 1.00, recall 0.76.
- Timing: 74% of inbound legs land at least 1 h before scheduled departure, but only 21% at least 3 h before. So at T-3h the inbound aircraft is usually still airborne or not yet departed; the useful signal there is the inbound's departure and position, not its arrival.
- Same-UTC-day only; overnight aircraft are not linked in this sample.

## 5. Latency

- `/flights/arrival` and `/flights/aircraft`: nightly batch (docs). Probe confirmed: KATL arrivals over the prior 10 h returned 2 flights, both over 5 h old. Not usable live.
- `/flights/departure` is near-live: returned in-progress flights with lastSeen within minutes of now (81% had no arrival airport yet). But it was sparse: 85 KATL departures in about 10 h (real ATL volume is several hundred).
- `/flights/all` for a recent window returned 1,018 flights versus an average of 5,220 per historical window (single sample, different hours, so only suggestive).
- `/states/all` (live state vectors): data age is about 1 s median (snapshot time minus last position time; p90 1 s, p99 about 220 s for stale transponders). Request time minus snapshot time: median 3 s. This is the realistic live source (see section 7).

## 6. Credit cost and production projection

| Approach | Credits/day | vs 4,000 limit |
|---|---|---|
| `/flights/*` calls | 30 each | only 133 calls/day total; polling one airport every 15 min for 16 h costs 1,920. Not scalable |
| `/states/all` CONUS every 5 min, 24 h | 1,152 | fits |
| `/states/all` CONUS every 2 min, 24 h | 2,880 | fits (72%) |
| `/states/all` CONUS every 1 min, 24 h | 5,760 | over |
| `/states/all` one 5x5 degree box per airport (1 credit) every 5 min, 16 h | 192 per airport | about 20 airports |

The `/states` bucket is separate from `/flights`, so a CONUS states feed (2 to 5 min) fits within one account's daily limit. A states-based feed would need our own logic to turn state vectors into events (on_ground transitions, callsign-to-flight mapping, tracking by icao24). Its coverage is measured in section 7 and is far higher than the `/flights` archive.

## 7. Live state-vector coverage (follow-up, 165 `/states` credits)

Method: polled CONUS `/states/all` (4 credits per call) 40 times at 6-minute spacing, 2026-11-03 22:44 to 2026-11-04 02:38 UTC (6:44 pm to 10:38 pm US Eastern, a Tuesday evening). All snapshots are cached. There is no BTS truth for today, so the denominator is the **August 2026 BTS domestic schedule on the same weekday, re-timed to the November local schedule** (local scheduled time with the November UTC offset, August actual taxi and air times).

**A. Concurrent airborne aircraft (the more robust measure).** Per snapshot: airline-callsign aircraft airborne in the snapshot, versus BTS-implied flights airborne at that moment (averaged over the same-weekday Augusts).

| Group | Expected airborne (mean) | Observed ratio, callsigns matching a BTS flight number | Observed ratio, all airline callsigns |
|---|---|---|---|
| Mainline (AA DL UA WN AS B6 F9 G4) | 1,527 | **0.96** | 1.11 |
| Regional (9E OO YX MQ OH YV PT QX G7 C5) | 487 | **0.93** | 1.03 |
| All | 2,014 | **0.95** | 1.09 |

By carrier (matching callsigns): AA 1.08, WN 1.07, UA 1.02, OH 1.07, YX 1.02, PT 0.97, C5 0.97, MQ 0.91, OO 0.84, 9E 0.82, AS 0.83, DL 0.81, QX 0.77, B6 0.70, F9 0.67 (G4, YV, G7 are small and noisy). By UTC hour the all-carrier ratio ranged 0.93 to 0.98. A ratio below 1 can mean missing aircraft or an August schedule that is bigger than November's (B6, F9, AS and DL cut or change schedules seasonally); a ratio above 1 means November has flights August did not. The all-callsign ratio is inflated for mainline by international and non-BTS flights inside the CONUS box. So the honest reading is "near-complete, roughly 85-100%", not a precise number.

**B. Per flight: seen near the origin around departure (what T-3h/T-1h needs).** 1,489 flights with scheduled departure between 30 minutes after the first snapshot and 2 hours before the last (23:14 to 00:38 UTC, which is 15:00 to 19:59 local across time zones), taken from August flights that recur on at least 75% of the same-weekday Augusts. "Near origin" = a state vector within 25 km of the origin airport, from 30 minutes before to 60 minutes after scheduled departure (on the ground or airborne).

| Group | Flights | Seen at all (-30 min to +2 h of sched. dep.) | Near origin, given seen at all | Near origin before scheduled departure, given seen |
|---|---|---|---|---|
| All | 1,489 | 35% | 46% | 17% |
| Mainline | 968 | 36% | 48% | 17% |
| Regional | 521 | 32% | 41% | 17% |

Per carrier (seen at all / near origin given seen): DL 47% / 69%, UA 55% / 67%, YX 33% / 65%, 9E 51% / 50%, OO 32% / 46%, B6 43% / 35%, AA 32% / 27%, MQ 21% / 19%, OH 40% / 10%, WN 18% / 2%. Hubs (seen at all / near origin given seen): LGA 58% / 82%, ATL 51% / 79%, DEN 44% / 78%, MSP 58% / 67%, SEA 33% / 62%, ORD 41% / 44%, DFW 29% / 38%, CLT 44% / 20%, MDW 16% / 0%. For flights seen near origin, the first sighting was a median 4 minutes after scheduled departure (p10 -22, p90 +40).

**Limits of B, stated plainly:**
- "Seen at all" at 35% is **not** a coverage rate. Its denominator includes August flights that do not run on this November date (Southwest at 18% is the clearest sign of a changed schedule, since WN renumbers seasonally). It contradicts nothing in A: 95% of airborne aircraft are present while only a third of the August flight numbers are found, which means the flight-number denominator is the weak part.
- "Near origin, given seen" removes the existence problem but not the sampling problem. At 6-minute spacing a climbing aircraft leaves a 25 km radius in a few minutes, so a flight is only caught near its origin if a snapshot lands while it is on the ground with its transponder on (or in the first minutes of climb). Large per-airport differences (LGA 82% vs CLT 20%, WN 2%) look more like ground reception, transponder habits and the delay window than a single coverage rate.
- Only one evening (about 4 hours), one weekday, November. No morning or midday data.
- The experiment cannot tell how much of the 46% gap is receiver coverage and how much is the 6-minute cadence. A denser test (1-credit boxes around hubs at 1-2 minute spacing) would.

**C. Latency.** Snapshot time minus aircraft position time: median 1 s, p90 1 s, p99 219 s. Request time minus snapshot time: median 3 s. So live state vectors are effectively real time (seconds).

**What it could feed.** Airborne/inbound-aircraft presence and position (is the inbound plane airborne, where is it), and "has this flight left" via airborne detection after takeoff, at roughly 95% airborne coverage. Not yet shown: reliable pushback/gate-out times, and arrival at the destination gate.

## What OpenSky could feed

- Via `/flights/*`: arrival/departure actuals and inbound lateness **when present** (about 25% of flights, 11% for a fully linked rotation). Accurate, but only useful as backfill for training-style analysis, not live (arrivals are nightly batch).
- Via `/states/all` live polling: inbound aircraft airborne status and position (about 95% concurrent coverage), departure detected after takeoff. These could feed the inbound-aircraft and "already departed" parts of the T-1h/T-3h features. Backtesting such features against BTS needs a recorded state-vector history, which we do not have; this experiment measured presence, not feature value.

## Gap a paid source might need to fill

Not established. If a denser hub-level test shows pushback/takeoff detection below about 90%, or if we need gate-out/gate-in times (which OpenSky does not provide: its times are runway events), the gap would be near-complete departure and arrival actuals by flight and tail, available within minutes, for all US scheduled carriers including the ten regionals. Whether AeroAPI meets that and at what price is not assessed here, and this report does not recommend buying it on current evidence.

## Caveats

- Sample is 6 days of one month (August, summer convective season); coverage on other months is unmeasured.
- `/flights/*` results describe the historical archive, not the live path; the live path is measured separately in section 7.
- The state-vector denominator is August schedules re-timed to November; seasonal schedule changes cap its accuracy at roughly +/-10-15%.
- Matching windows (60 min) are generous; stricter windows would lower rates slightly. The 0.5-minute median error suggests matches that exist are real.
- Raw responses are cached in `data/opensky_cache/` (gitignored); `python backtest/opensky_coverage.py analyze` reruns for free.
