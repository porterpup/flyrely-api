# A5: Is the free OpenSky API good enough for T-3h / T-1h features?

Run: 2026-11-03, BTS month 2026-08 only (holdout untouched). Code: `backtest/opensky_coverage.py`; numbers: `backtest/results/a5_opensky_coverage.json`.

## Verdict

**No, not as the primary feed. Yes as a thin, high-precision supplement.**

- OpenSky sees only about a quarter of our flights (26% depart-side, 25% arrive-side, 22% both). That is a coverage ceiling in OpenSky's receiver network, not a matching bug: only about 3,000 US-departing AAL-callsign flights appear in 6 days, against roughly 16,000 AA flights in BTS.
- What it does see is very accurate. Times track wheels-off/wheels-on to about 0.5 minutes (median). The hard problem is that a feature missing for 75% of flights is mostly missing.
- A T-3h / T-1h model needs the "recent actuals" and "inbound aircraft" features for nearly every flight. A feed at about 25% coverage means the model would be running without them for three flights in four. A paid or full-coverage source is likely needed to fill that gap (fill rate near 100% for departure/arrival actuals and inbound tail position). I did not research pricing.
- Latency is also a problem for arrivals (see below).

## Sample and credits

- 6 UTC days (Aug 4, 8, 12, 16, 20, 25), 72 two-hour `/flights/all` windows, 375,908 unique OpenSky flights. 97,889 completed BTS flights in scope (cancelled/diverted excluded, flights well inside the day edges).
- **Credits spent: 2,255 of the 2,500 cap.** `/flights` bucket ended at 1,750 (floor was 1,500); `/states` bucket at 3,995 (separate bucket).
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
- **By origin airport tier:** top-30 25%, ranks 31-100 28%, other 25% (departure side). Arrival side is better at small airports (41%) than hubs (20%). No tier is good.
- **By hour:** ranges 12% to 44% with no clean pattern; no hour is above 50%.
- Hub examples (departure side): DTW 45%, CLT 41%, ATL 34%, BOS 18%, SEA 17%, IAH 17%.

## 2. Time accuracy (matched flights)

| Comparison | n | Bias | Median abs | P90 abs |
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
- `/states/all` (live state vectors, authenticated, up to 1 h history): a CONUS call returned 6,687 aircraft, 647 on the ground, about 2,685 with a major-airline callsign. This is the realistic live source.

## 6. Credit cost and production projection

| Approach | Credits/day | vs 4,000 limit |
|---|---|---|
| `/flights/*` calls | 30 each | only 133 calls/day total; polling one airport every 15 min for 16 h costs 1,920. Not scalable |
| `/states/all` CONUS every 5 min, 24 h | 1,152 | fits |
| `/states/all` CONUS every 2 min, 24 h | 2,880 | fits (72%) |
| `/states/all` CONUS every 1 min, 24 h | 5,760 | over |
| `/states/all` one 5x5 degree box per airport (1 credit) every 5 min, 16 h | 192 per airport | about 20 airports |

The `/states` bucket is separate from `/flights`, so a CONUS states feed (2 to 5 min) fits within one account's daily limit. A states-based feed would need our own logic to turn state vectors into events (on_ground transitions, callsign-to-flight mapping, tracking by icao24). Coverage of state vectors in the air is likely similar to the historical coverage, so this does not fix the 25% ceiling; I did not measure state-vector coverage against BTS.

## What OpenSky could feed

- Arrival/departure actuals and inbound lateness **when present** (about 25% of flights, 11% for a fully linked rotation): accurate, but not enough alone.
- Best use: a supplementary signal for flagged hubs/regionals (PT, G7, 9E, DTW, CLT, ATL have 35-50% coverage), or to validate another feed.
- Not suitable: anything that must be present for most flights, or any arrival-based live feature via `/flights`.

## Gap a paid source would need to fill

Near-complete (>90%) coverage of departure and arrival actuals by flight and tail, available within minutes (not next day), plus inbound aircraft position or status, for all US scheduled carriers including the ten regionals. Whether AeroAPI meets this and at what price is not assessed here.

## Caveats

- Sample is 6 days of one month (August, summer convective season); coverage on other months is unmeasured.
- `/flights/all` is only the historical archive; the live path differs (see Latency).
- Matching windows (60 min) are generous; stricter windows would lower rates slightly. The 0.5-minute median error suggests matches that exist are real.
- Raw responses are cached in `data/opensky_cache/` (gitignored); `python backtest/opensky_coverage.py analyze` reruns for free.
