# FlyRely roadmap, November 2026: SWIM, live accuracy, tail linkage, run-up serving

Status: **plan and draft pre-registrations only.** No OpenSky, SWIM, AeroDataBox or other data calls were made to write this,
and no code was changed. Written 2026-11-05. It replaces `PLAN_A6_NATIONWIDE.md` (renamed). The source research from
that plan (ADS-B network terms, schedule sources, the scraping review, the A6 event definitions) is kept in the
appendices, because it is still correct.

Inputs: `HANDOFF.md` (2026-11-05), `REPORT_A5B_OPENSKY_HUBS.md`, `REPORT_B5_1.md`, `swim/consumer.py`, the PR #13 review
("before always-on" items), `prediction_log.py` and `migrations/001_predictions.sql`, `flyrely_features/{network,rotation,faa,metar}.py`.

What changed since the A6 plan:
- **FAA SWIM is approved** (TFMS Flight + Flow + Status; SFDPS Flight FIXM + Status; STDDS SMES, TDES, ISMC). The smoke test
  (PR #13) measured: TFMS has no registration or icao24; TFMS carries airline OUT/OFF/ON/IN on 26-28% of flights (10-minute
  sample); about half of TFMS messages are flow (TMI) data; SFDPS has the registration on 28.7% of airline flights,
  icao24 on about 23%, and actual wheels-off on 87%; about 5 GB/day compressed for both feeds; STDDS untested.
- **B5/B5.1 adopted** FAA GS/GDP (T-1d, T-3h, T-1h) and METAR (T-3h, T-1h), on validation months only.
- **Sean's decisions:** OpenSky polling is OK during the MVP (revisit the licence before launch, no email); the adsb.lol
  attribution is fine; no scraping; the AeroDataBox $8 pilot is not yet approved.

---

## 0. Summary

### 0.1 Two facts that shape the order

1. **The prediction log is already merged.** `feat/prediction-log` was merged as PR #5 on 2026-11-03; the branch has no
   commits that master lacks and can be deleted. Every `/predict` call already writes a row (horizon `booking`). What is
   missing is everything after that: confirming the rows survive deploys in production, collecting outcomes, and scoring.
   Because the training label is BTS departure delay of 15 minutes or more, and BTS arrives two months late, **the only
   way to score a live prediction within days is SWIM** (or AeroAPI per flight). So live accuracy tracking and the SWIM
   archiver are tied together, and both go first.
2. **Production serves only the booking-time model.** The T-3h and T-1h run-up models (AUC 0.82 / 0.86) are backtest-only.
   Their strongest inputs are same-day network state (`network.py`: departures, delays, cancellations, backlog at the
   origin, destination and airline) and aircraft rotation (`rotation.py`: where this flight's aircraft is and how late its
   earlier leg is). Neither has a live source today. **"B5.1 to production" is therefore really "serve the run-up models",
   and SWIM is what makes that possible.** FAA flow and METAR are increments on top.

### 0.2 Ordered roadmap

| # | Step | Owner | Effort | Cash cost | Depends on |
|---|---|---|---|---|---|
| 1 | **R1 Live accuracy tracking, part 1**: confirm the production log is durable; outcomes table; BTS final-outcome loader; scorer and weekly report | [executor] check, then [implementer] | 0.5 d + 2 d | $0 | none |
| 2 | **R2 SWIM always-on archiver** (TFMS + SFDPS) on Railway: ingest filtering, zstd, R2 upload, LADD filter, heartbeat and alerts, no auth restart loop | [implementer], then `reviewer` | 3-4 d | about $25-40/month Railway + $1-5/month R2 | Sean: R2 account, alert destination |
| 3 | **H1 B5.1 holdout confirmation**: pre-register now, run when BTS September lands (about Nov 17-20) | [planner/reviewer] protocol, [implementer] run | 0.5 d + 1 d | $0 | BTS release |
| 4 | **R3 SWIM live-state extractor** to Postgres (flight state, TMI programs, feed status), then **R1 part 2**: provisional outcomes from SWIM | [implementer] | 3 d + 1 d | $0 (fits current Postgres) | R2 |
| 5 | **A7 + L0** offline value-of-linkage and pairing-stability study on BTS validation months | [implementer], then `reviewer` | 1.5 d | $0 | none (can run in parallel with 2-4) |
| 6 | **A6S + L1** live collection (SWIM truth; OpenSky and adsb.lol arms; linkage methods; AeroDataBox pilot on two days) | [implementer] build, main session runs, `reviewer` | 4-5 d build, 5 collection days | $8 (AeroDataBox) | R2 running 7+ days, A7 result |
| 7 | **S1 STDDS onboarding** (smoke, volume, filter, third service) | [executor] smoke, [implementer] service | 0.5 d + 1 d | +$7-13/month | R2 stable |
| 8 | **P0-P4 Run-up serving** (live feature adapters, parity tests, retrained models, shadow mode, launch) | [planner/reviewer] design, [implementer] build | 8-12 d | $0 extra; AeroDataBox $75-99/month only if L1 adopts it | R3, H1, A7, L1 |
| 9 | **V1 SWIM-to-BTS parity check** when BTS November lands (about mid-January 2027) | [implementer], `reviewer` | 1 d | $0 | R2 archive for November |

Steps 1, 3 and 5 need no new infrastructure and can start today in parallel. Step 2 is the critical path for 4, 6, 7 and 8.

### 0.3 Headline answers

- **What SWIM replaces:** ADS-B detection as the source of departure/arrival truth and as the live "has the inbound left /
  landed" signal; the BTS-August denominator; the FAA advisory scraping and nasstatus for live programs (nasstatus stays as
  a fallback); possibly the AeroDataBox schedule. **What it does not replace:** aircraft identity for about 71% of flights
  (the tail gap), gate OUT/IN times for about 72% of flights (wheels-off is not the training label), METAR, BTS as the
  training and final-label truth, and redundancy during SWIM outages. Section 1.
- **A6 is re-scoped** to "A6S": SWIM is the per-flight truth (validated against BTS), and the questions become SWIM
  completeness and latency, and what each ADS-B arm (OpenSky O, adsb.lol L) adds for icao24 and redundancy. All three arms
  remain. Section 2.
- **Tail linkage** is measured by a pre-registered experiment L1 that combines SFDPS registration/icao24, through-flight
  numbers, a rotation-pairing prior learned from our own archive, and ADS-B icao24, with AeroDataBox as a separate arm.
  Section 3.
- **AeroDataBox $8 pilot: GO**, spent on L1 days D1-D2 (proposed Nov 18-19). One kill condition, decided before spending:
  skip it if A7 shows that even perfect linkage is worth less than 0.0005 Brier at T-3h. Section 3.5.
- **Hosting: Railway**, one service per source, about **$30-55/month all-in** including R2 storage, with Hetzner (about
  $10/month all-in, more upkeep) as the fallback if the Railway bill passes $60. Section 4.

---

## 1. What SWIM replaces, and what it does not

| Need | Before SWIM | With SWIM (measured fill) | Verdict |
|---|---|---|---|
| Per-flight departure/arrival truth (A6 denominator) | BTS August same-weekday estimate (+/-10-15%, flight-number churn) | TFMS departure/arrival messages + SFDPS wheels-off (87%) for every IFR flight, nationwide, seconds after the event | **Replaced**, after a completeness check against BTS (A6S Q1, V1) |
| Live "inbound has departed / landed / ETA" for a known inbound flight | ADS-B detection (A5b: arrivals 0.97, departures 0.90 at 2-minute polling; 3 of 12 hubs reliable) | TFMS/SFDPS by callsign; TFMS also carries ETAs | **Replaced** for status. ADS-B stays as a cross-check |
| Same-day network features (`network.py`: departed, delayed share, mean late, cancelled, pending) | None live (BTS only, two months late) | TFMS schedule-derived flight creates (scheduled counts, pending), departures, cancellations, nationwide | **Newly possible.** Delay definition needs care (next row) |
| Departure delay as defined in training (gate OUT minus scheduled, BTS) | none | OUT only on 26-28% (TFMS FlightTimes, sample). Wheels-off on 87% (SFDPS). OUT has to be estimated as OFF minus taxi-out | **Partial.** Needs the taxi-out mapping and a parity check (P1, V1). STDDS TDES may add OUT/taxi at tower sites |
| FAA GS/GDP programs at serve time | ATCSCC advisory DB (research fetcher) or nasstatus | TFMS Flow (about 50% of TFMS messages) | **Replaced**, nasstatus as fallback; parity check P1-FAA |
| Today's schedule ("what flights are we looking for") | AeroDataBox (proposed) or BTS August | TFMS flight creates from schedule data, if they arrive early enough | **To measure** (A6S Q4). If yes, AeroDataBox is needed only for tails |
| Aircraft identity (tail / icao24) per flight | none (A5b linkage failed) | SFDPS registration 28.7%, icao24 about 23% | **Not replaced.** The tail gap is the main open problem (section 3) |
| Surface timing at big hubs (pushback, taxi) | ADS-B ground vectors (missing at CLT, DTW, ORD in A5b) | STDDS SMES/TDES at about 35 ASDE-X + 9 ASSC airports | **Likely replaced** at those airports; STDDS untested (S1) |
| Observed weather | METAR (IEM in training) | none | **Not replaced.** Live METAR from aviationweather.gov |
| Training data and final labels | BTS | SWIM history only from Nov 2026 | **Not replaced.** BTS stays the label and training truth |
| Redundancy (FAA requires us to tell users about outages) | none | SCDS Status topics tell us when a feed is down | ADS-B arm stays as the degraded-mode source |

So SWIM moves FlyRely from "detect events with ADS-B" to "read events from the FAA". ADS-B keeps three jobs: icao24 for
airborne flights (the linkage helper), surface timing where STDDS does not reach, and a backup feed.

---

## 2. A6S: SWIM completeness and what ADS-B adds (revised A6 protocol, draft)

To be copied into `backtest/REPORT_A6S_SWIM_COVERAGE.md` and committed before D1 (gate G1).

### 2.1 Questions
- **Q1 Completeness.** Does SWIM (TFMS ∪ SFDPS) contain every airline departure and arrival at the top-100 airports?
- **Q2 Latency and fill.** How soon after the event do OFF/ON messages arrive, and over a full day, what share of flights
  have OUT, OFF, ON, IN?
- **Q3 ADS-B value.** Against SWIM per-flight truth, what recall, precision and timing error do arms O and L reach (detectors
  D0/D1/A0/A1 from Appendix C), and for what share of departures does ADS-B give the icao24 by 20 minutes after wheels-off?
- **Q4 Schedule lead.** What share of flights have a TFMS flight create at least 24 h / 6 h / 3 h before scheduled departure?

**Yes: coverage is now measured against SWIM truth instead of the BTS August estimate.** BTS August is still reported for
continuity with A5b, and SWIM itself is checked against BTS (Q1 now against August same-weekday counts, later per-flight
against BTS November in V1).

### 2.2 Arms
- **S (SWIM):** TFMS Flight + SFDPS from the always-on archiver (R2). Primary truth.
- **O (OpenSky):** CONUS `/states/all` every 120 s, 11:00-03:00 UTC, 1,920 credits/day, the A5b client with its
  1,500-credit floor and 429 stop. Allowed for the MVP; the licence must be revisited before launch.
- **L (adsb.lol):** the same round clock, the frozen 250 nm tile set, at most 1 req/s, ODbL attribution. This is the
  launch-safe ADS-B arm, so its standalone numbers matter most.
- O and L also run on the 7 days before D1, because L1's rotation prior needs observed history (section 3.3).

### 2.3 Days
D1 Wed 2026-11-18, D2 Thu 11-19, D3 Tue 12-01, D4 Wed 12-02, D5 Thu 12-03. Thanksgiving week is excluded. Backups in
order: Tue 12-08, Wed 12-09, Thu 12-10. A day is replaced if any arm loses more than 10% of rounds or the SWIM archiver has
a gap over 15 minutes. (The A6 dates Nov 10/12/17 are dropped: the archiver will not have run for 7 days by then.)

### 2.4 Population and truth
- Population: flights with an airline callsign (3-letter ICAO code of a BTS reporting carrier, or its regional partner, plus
  1-4 digits) departing from or arriving at one of the top-100 airports (frozen list from BTS August), scheduled within the
  11:00-03:00 UTC window, minus the first and last 30 minutes.
- SWIM truth event: the actual OFF (or ON) time from TFMS departure/arrival messages, else SFDPS actual times. Flights with
  a cancel message and no OFF are cancellations.

### 2.5 Metrics
- Q1: SWIM airline departures and arrivals per airport-day ÷ BTS August same-weekday expected; also the share of flights in
  the TFMS schedule with an OFF message (missing-departure rate).
- Q2: median and p90 of (receive time − event time) for OFF and ON; full-day fill of OUT/OFF/ON/IN by carrier group
  (mainline, regional).
- Q3: per-flight recall (SWIM flight with a detected event of the same callsign within ±20 min), precision, median and p90
  |detected − SWIM| time error; marginal gain of O and L over each other (Appendix A.2); share of departures with icao24
  from ADS-B within 20 minutes of OFF.
- Q4: share of population flights with a TFMS flight create at least 24 / 6 / 3 h before scheduled departure.

### 2.6 Decision rules (written now)
- **SWIM is accepted as truth** if, aggregated over the top-100, SWIM departures ÷ BTS-August expected is between 0.90 and
  1.10 and the missing-departure rate is at most 2%. Otherwise A6S reports SWIM gaps by airport and V1 decides.
- **SWIM replaces ADS-B for live status** if OFF and ON median latency is at most 2 minutes and p90 at most 5 minutes.
- **An ADS-B arm is "useful for linkage"** if it gives the icao24 within 20 minutes of OFF for at least 70% of departures
  that SFDPS does not cover. The production ADS-B arm is the cheapest useful one; if only O is useful, that is flagged as a
  launch blocker (licence).
- **TFMS is the schedule source** if at least 95% of population flights have a flight create at least 6 h ahead. Then
  AeroDataBox is needed for tails only.

---

## 3. Tail and inbound linkage

### 3.1 Why it matters
The run-up models' rotation features (`rot_known_delay`, `rot_projected_delay`, `rot_slack` and the rest) need, at the
as-of time, the aircraft's earlier leg today and how late it is. SWIM gives the status of any flight by callsign, so
**status is no longer the bottleneck. Identity is: which earlier flight is this departure's inbound?** A5b showed that the
August tail assignment is useless (modal tail share 25-33%).

### 3.2 Sources and how they combine
| Method | What it gives | Known limits |
|---|---|---|
| M1 SFDPS flight plan | `REG/` (28.7%) and icao24 (about 23%) on the departure's own plan | Fill rate by lead time is unknown: plans may be filed only 1-2 h ahead |
| M2 Through-flight | Same flight number arriving earlier the same day at the origin: usually the same aircraft | Only some flights are through-flights |
| M3 Rotation-pairing prior | Pairing "inbound flight A → outbound flight B", learned from the previous 7 days of our own archive (SFDPS registration + ADS-B icao24 + callsign sequence), applied to today's schedule; confirmed when A is airborne and its icao24 is seen | Only as good as day-to-day pairing stability (L0 measures it on BTS) |
| M4 AeroDataBox | `aircraft.reg` / `modeS` on the departure board | Paid; fill by lead time unknown; may itself come from the same FAA filings |
| ADS-B icao24 | Turns an airborne callsign into an aircraft (`n_number_to_icao24` maps N-numbers), so M2/M3 inbound A can be confirmed and tracked | Only after A is airborne; coverage from A6S Q3 |
| AeroAPI `inbound_fa_flight_id` | The inbound directly, per flight | $0.005-0.02 per call; only for user-tracked flights; Standard tier needed for commercial use |

Fixed priority when methods disagree: M1 > M4 > M2 > M3 (free-only priority: M1 > M2 > M3).

### 3.3 L0 (offline, now): pairing stability on BTS
On validation months 2026-01..08 (no holdout): for each flight B with a tail, define its inbound pairing as the flight
number of the same tail's previous leg into B's origin that day. Report P(pairing on day d equals pairing on d−7) and on
d−1, by carrier and hub. This tells us whether M3 can work before any live collection. [implementer], part of A7.

### 3.4 L1: linkage coverage (pre-registered draft; runs on A6S days)
- **Population:** A6S departures (section 2.4) with scheduled departure 13:00-03:00 UTC, so the T-3h as-of time falls inside
  collection.
- **As-of times:** T-3h and T-1h before scheduled departure. A method may use only messages received before the as-of time.
- **Truth (interim):** the icao24 that flew B: SFDPS icao24/registration on B after OFF, else the ADS-B (O ∪ L) aircraft with
  B's callsign within 30 km of the origin within ±20 min of SWIM OFF. Truth inbound A: the latest arrival at B's origin by
  that aircraft before B's OFF. **Final truth:** the BTS November tail (about mid-January 2027), used to re-score; the
  go/no-go below uses interim truth, and the final re-score is reported as confirmation.
- **Metrics per method and for the free union and full union:** coverage (share of population with a prediction), precision
  (prediction equals truth, on flights with truth), **effective linkage = coverage × precision**, and "usable inbound" (linked
  and A's SWIM status known at the as-of time). Reported by horizon, carrier group and airport tier (12 A5b hubs, ranks
  1-30, 31-100).
- **AeroDataBox arm:** on D1 and D2 only, at 24 airports (the 12 A5b hubs plus ranks 13-24 by BTS August departures not
  already included). One FIDS pull per airport per hour from 10:00 to 01:00 UTC (16 pulls, `direction=Both`, 12 h window,
  2 units each), plus one pull the evening before: about 24 × 17 × 2 × 2 days ≈ **1,630 units of 5,000**. Raw responses are
  deleted after 7 days (Pro plan storage limit); only derived per-flight predictions are kept. ADB metrics are computed on
  the same 24 airports for both unions.

### 3.5 Decision rules (written now)
- **Free linkage is viable** if the free union's effective linkage at T-1h is at least 0.60 across the top-100.
- **AeroDataBox is adopted** (Ultra 2 $75 or Growth $99 per month) if, at T-3h on the 24 airports, effective linkage of
  (free ∪ M4) minus (free) is at least 0.15, M4 precision is at least 0.90, **and** A7 says that closing that gap is worth
  at least 0.0005 Brier at T-3h.
- **AeroDataBox is rejected** if the gain is under 0.05 or its precision is under 0.90. In between: partial. Use AeroAPI
  per flight for user-tracked flights instead.

**Go/no-go on the $8 pilot: GO.** Cost is trivial and it is the only way to learn whether AeroDataBox's tails are earlier
than SFDPS's. Spend it on D1-D2, not before, so that it is measured on the same flights as the free methods. **Single kill
condition, checked before spending:** if A7 shows that perfect linkage (100%) is worth less than 0.0005 Brier at T-3h,
linkage is not worth paying for at any price, and the pilot is skipped.

### 3.6 A7: how much accuracy is at stake (offline, now)
Validation months only, no external calls:
1. Retrain the T-3h and T-1h models with rotation features knocked out for a random (1 − p) of flights, p ∈ {0, 0.25,
   0.29, 0.5, 0.7, 0.9, 1}, both with and without training on the same missingness (the serve model must be trained with the
   missingness it will see).
2. Also a non-random pattern: missing for all regional carriers, present for mainline at SFDPS-like rates.
3. Report Brier/AUC/ECE against p. This gives the value of each extra point of linkage, which feeds the ADB rule and the
   P2 model choice.

---

## 4. Always-on SWIM ingestion architecture (R2)

### 4.1 Shape
```
Solace SCDS queue (one per source)
   └─ Railway service "swim-tfms" / "swim-sfdps" / later "swim-stdds"   (1 replica each, same image)
        consumer: receive → header/regex filter → LADD drop → zstd hourly file on volume → fsync → ack
        extractor thread (R3): key events → Postgres live tables (upsert every 5 s)
        uploader thread: closed hourly files → Cloudflare R2 → local delete after 24 h
        heartbeat thread: healthchecks.io ping every 60 s, one-line JSON metrics per minute
   └─ Railway cron "swim-daily": R2 raw → parsed parquet (flight events, flight plans/tails, TMI, EDCTs) + daily summary
API (existing Railway service) reads Postgres live tables at /predict time; never talks to Solace.
```
The archiver stays simple: ack only after fsync (as now), and parse heavily only in the batch job. The live extractor
reads only the few fields features need.

### 4.2 Where it runs, and costs (prices checked 2026-11-05; confirm at sign-up)
| Option | Monthly cost | Pros | Cons |
|---|---|---|---|
| **Railway (recommended)** | Per consumer at about 0.3 vCPU + 0.5 GB average: about $6 CPU ($20/vCPU-month) + $5 RAM ($10/GB-month) ≈ $8-13. Two consumers now, three with STDDS, plus a small daily cron: **$25-40**. Volume 20 GB at $0.15 = $3. Egress to R2 about 45 GB at $0.05 = $2. | The API already runs there; managed restarts, sealed variables, logs; Claude can deploy | Usage-based, so a CPU-heavy consumer costs more; Hobby includes only $5 of usage (Pro is $20/seat with $20 included) |
| Hetzner CX23 VM | about €4-6 + €0.50 IPv4 ≈ **$6-8** | Cheapest; plenty of CPU for three consumers | We own patching, Docker/systemd, disk, and the VM's own monitoring |
| Fly.io | about $6-12 per small machine | Similar to Railway | Another vendor and account |
| Sean's Mac | $0 | — | Not always on; already ruled out |

Storage: **Cloudflare R2** at $0.015/GB-month (first 10 GB free), no egress fees, operations within the free tier (about
2,200 uploads/month). Backblaze B2 (about $0.006/GB-month) is the cheaper alternative if the archive grows.

**Total: about $30-55/month** on Railway + R2 for TFMS + SFDPS + STDDS. Move to Hetzner if the Railway bill exceeds $60/month.

### 4.3 Cutting the 5 GB/day
Measure first: [executor] reads the existing smoke-test archives offline and writes a byte and message histogram by message
type and source, plus messages per second. No network. Then:
1. **Compression:** one long-lived zstd stream (level 6) per hourly file, `flush(FLUSH_BLOCK)` + fsync before each ack,
   instead of one gzip member per 1 s batch. Expect 2-3x smaller. The reader must tolerate a truncated final frame.
2. **Airline-only flight messages:** keep TFMS/SFDPS flight messages only for airline-pattern callsigns. This drops general
   aviation (and most LADD traffic) at the door.
3. **Message-type allow-list** (names to be confirmed from the histogram): TFMS keep flight create/schedule activate, flight
   plan and amendment, departure, arrival, FlightTimes, cancel, ETA updates; drop boundary crossings and oceanic reports.
   SFDPS keep flight plan, amendment, departure, arrival, cancel, hand-off only where it carries times; tracks as below.
   **Flow and Status: keep everything** (TMI programs, EDCT lists, and outage notices are needed).
4. **Track downsampling:** keep one track position per flight per 5 minutes, plus every position within 40 km of the origin
   or destination airport below 3,000 m (the part useful for takeoff/landing timing).
5. **Filter cheaply:** use Solace message properties (such as `TFMDataClass`) before decoding the body; fall back to a regex
   on the body for callsign and message type. No XML parsing in the hot path.

Target: at most 1 GB/day compressed for TFMS + SFDPS. A **24-hour unfiltered capture** is kept once, for filter calibration,
and deleted after 14 days.

### 4.4 Retention
| Tier | Where | Kept | Note |
|---|---|---|---|
| Spool | Railway volume | until uploaded, then 24 h | alert at 70% full |
| Raw filtered | R2 `raw/{source}/{date}/` | **13 months**, lifecycle rule | one full seasonal cycle for retraining; about 400 GB at the end ≈ $6/month |
| Parsed tables | R2 `parsed/` parquet | indefinitely while the SWIM agreement is active | small (MB/day); LADD-filtered on write and re-filtered monthly |
| Live state | Postgres | 72 h rolling | about 80k rows |
| Unfiltered calibration capture | R2 `calib/` | 14 days | once |

On non-renewal of the SWIM agreement, delete per its terms (check the SAA clause when signing the 2027 renewal).

### 4.5 LADD
- Obtain the FAA LADD industry list (monthly) and the PIA rules. **Sean may need to register as a data recipient**; the
  process must be checked (open item).
- `swim/ladd.py`: blocked registrations, icao24s and callsigns. Applied at ingest (drop the message), again when building
  parsed tables, and again at serve/display time, so a newly added aircraft is suppressed even in older data.
- The same filter is applied to ADS-B data merged with SWIM.
- Alerts if the list is more than 40 days old. Never try to re-identify PIA aircraft.

### 4.6 Monitoring and alerts
- **Heartbeat:** each consumer pings its healthchecks.io check (free tier, 20 checks) every 60 s **only if** it is connected
  and has archived at least one message in the last 5 minutes. Grace 10 minutes. So "silent but connected" (zero messages)
  alerts the same way as "process dead".
- **Reconnect storm:** more than 5 reconnects in 30 minutes → fail ping.
- **Authentication failure:** the consumer does **not** exit into a restart loop (repeated bad logins risk locking the
  account). It sends a fail ping with reason `auth`, then parks: it sleeps and retries at most once every 6 hours. Exit code 3
  stays for CLI runs only. Railway restart policy stays `ON_FAILURE` with 3 retries as a backstop.
- **Uploader:** ping per successful hourly upload, grace 2 h. **Extractor:** ping while Postgres live tables are updated
  within 5 minutes. **LADD:** daily freshness check.
- **Metrics line** per minute (JSON): messages and bytes by type, dropped by filter, dropped by LADD, redelivered, receive lag
  (receive time − sender timestamp). Daily summary row in `swim_feed_status` and a message to Sean's chosen channel.
- **Outage obligation:** the extractor parses SCDS Status messages into `swim_feed_status`; the API exposes it, and the app
  shows a banner ("live FAA data delayed") when a feed is down or stale. Live features fall back to "unavailable" (missing),
  never to stale values.

### 4.7 Process, image and onboarding rules
- **One process per source, one durable queue per source, one replica.** Two consumers on the same non-exclusive queue split
  the messages between them, and the SWIM agreement forbids duplicate subscriptions. When Railway runs, nobody runs the
  consumer on the Mac: the consumer refuses to start without `SWIM_ROLE=prod`, or `--dev` plus a stated reason (logged).
  Development reads from R2.
- **Image:** `Dockerfile.swim` (python:3.12-slim + `ca-certificates`), `SCDS_TRUST_STORE_DIR=/etc/ssl/certs`, start command
  `python -m swim.consumer --source <src>`; `FLYRELY_ENV_FILE` unset (Railway variables instead of .env).
- **Credentials:** Railway sealed variables `SCDS_*`; never logged (the redaction in PR #13 stays).
- **Remaining PR #13 items:** most "must fix" items are done in master; still open: the per-batch gzip members (replaced by
  zstd above), metrics/heartbeat, LADD, and confirming the dedupe getters against the live broker.
- **STDDS onboarding (S1):** a supervised 10-minute smoke per message type (SMES, TDES, ISMC) [executor]; byte histogram;
  then decide filters. SMES surface positions (about 1 Hz per target) are probably the bulk, so keep SMES events and airline
  targets downsampled to 1 per 10 s near runways and gates, and keep TDES in full. Add it as the third service only after
  the volume is known [implementer].

---

## 5. Path to production for run-up models with the B5.1 features (P0-P4)

### 5.1 Live feature adapters, each with a parity test
| Feature group | Training source | Live source | Parity plan |
|---|---|---|---|
| Network (`network.py`) | BTS same-day actuals | Postgres flight state from SWIM: scheduled (TFMS create, IGTD ≈ CRS time), departed, cancelled, delay | Delay = OUT − scheduled when OUT exists; else OFF − taxi-out median (BTS by airport and hour) − scheduled. Check on the 26-28% with both OUT and OFF (interim), and against BTS November (V1) |
| Rotation (`rotation.py`) | BTS tails | L1 linkage + SWIM status of the inbound | Missing linkage served as missing, with a model trained on matching missingness (A7) |
| FAA GS/GDP (`faa.py`) | ATCSCC advisories (send time, chain, default lengths GS 90 / GDP 480) | **TFMS Flow**. If Flow carries the ATCSCC advisory text, parse it with the existing `classify()`, which gives exact parity; otherwise map GS/GDP program messages to the same (element, kind, status, sent_utc) rows. **Default lengths, as the protocol requires.** Fallback: nasstatus (first-seen time as send time) | Over the A6S days, compare SWIM-derived program events against the advisory DB lists for the same days (one-off research fetch, as in B5): ≥95% of events matched and median send-time difference ≤2 min |
| METAR (`metar.py`) | IEM archive, 10-minute lag | aviationweather.gov `/api/data/metar` for the 100 airports in one call every 5 minutes, via the existing `from_aviationweather()`, the same 10-minute lag | 3 days of live captures vs IEM for the same reports: identical derived features in ≥99% of airport-hours |

Real program periods from SWIM Flow (instead of default lengths) would need a new pre-registered experiment (B6); training
history for it could come from the advisory detail pages. It is not on the critical path.

### 5.2 Steps
- **P0** [planner/reviewer]: serving design: endpoint contract per horizon, where live state is read, the missing-data rules.
- **P1** [implementer]: the four adapters above and their parity tests (CI runs them on recorded fixtures; no network).
- **P2** [implementer]: refit the T-1d, T-3h and T-1h models with the adopted B5.1 sets and A7's missingness choice; frozen
  artifacts with versions.
- **P3** [implementer]: shadow mode: the API computes run-up predictions for each logged flight at T-1d/T-3h/T-1h (a
  scheduler, not user calls), logs them with their horizon, and does not show them.
- **P4** go-live rule (written now): show run-up predictions once at least 2 weeks and 2,000 shadow predictions have
  provisional outcomes, and live Brier at T-3h is no worse than the backtest Brier + 0.010 and better than the booking model
  on the same flights. Final confirmation when BTS for those weeks lands.

---

## 6. H1: holdout confirmation when BTS September lands

- **Now** [planner/reviewer]: commit `backtest/PROTOCOL_B5_HOLDOUT.md`: code frozen at master `2825135` (B5.1); same
  models, features and decision rule (95% day-block bootstrap CI of the Brier difference entirely below 0, and ECE within
  +0.005); evaluation month 2026-09; October 2026 as a second, separately reported confirmation when it lands (about
  mid-December). Results are reported whatever they show; no re-tuning.
- **When BTS September is released** (expected around Nov 17-20; August came out Oct 19) [implementer]: with
  `FLYRELY_HOLDOUT_UNLOCK=1` set only for these commands, fetch BTS September, METAR for the 100 airports for September, and
  the advisory list pages for September (lists only, default lengths); run one evaluation; write `REPORT_B5_HOLDOUT.md`;
  `reviewer`.
- Expected result: T-3h/T-1h FAA (about −0.002) should hold. T-1d FAA (−0.0010) and METAR (about −0.0004) are small and may
  not clear the CI on one month. If they fail, they are dropped at that horizon: no second try on the same month.
- **After H1**, September is spent. Recommendation: October is the second confirmation; November-December 2026 become the
  SWIM-parity months (V1); the next clean model holdout starts 2027-01.

---

## 7. R1: live accuracy tracking (the business's top priority)

### 7.1 State
- `prediction_log.py` + `migrations/001_predictions.sql` are on master (PR #5, merged 2026-11-03). `/predict` writes one row
  per call in a background task, with features, model version, horizon (`booking`), minutes to departure and the operating
  flight. SQLite by default, Postgres when `DATABASE_URL` is set. The local `feat/prediction-log` branch is fully merged and
  can be deleted.
- **Risk:** if `DATABASE_URL` is not set on Railway, rows go to SQLite on the container's temporary disk and are lost at
  every deploy. Check first.
- Missing: outcomes, scoring, reporting.

### 7.2 Build
1. [executor] Confirm `DATABASE_URL` is present on the Railway API service (key name only, never the value) and that the
   `predictions` table is receiving rows (count by day). Delete the merged branch.
2. [implementer] `migrations/002_outcomes.sql`: `outcomes` keyed by operating carrier + flight number + origin + local
   scheduled departure date: scheduled departure UTC, OUT, OFF, departure delay, cancelled, diverted, `label_d15`,
   `label_quality` (`exact` = gate OUT; `estimated` = OFF minus taxi-out), `source` (`swim`, `aeroapi`, `bts`), fetched time.
   BTS final rows replace provisional ones but both are kept for agreement stats.
3. [implementer] Outcome loaders: BTS monthly (final; same label as training); SWIM provisional from R3 (within hours);
   optional AeroAPI for app-tracked flights (decision for Sean).
4. [implementer] `scripts/score_live.py`: weekly report (n, Brier, log loss, AUC, ECE and reliability bins) by model version,
   horizon, client, carrier group and lead time; comparison with backtest expectations; provisional-vs-final label agreement;
   counts of predictions without outcomes. Output JSON + markdown; later a dashboard.
5. Alert: weekly summary to Sean's channel; drift alert if the 14-day rolling Brier exceeds the backtest Brier + 0.02 with
   n ≥ 500.

---

## 8. Build-step list (tags, effort, cost)

| ID | Step | Tag | Effort | Cost |
|---|---|---|---|---|
| R1.1 | Confirm production log durability; delete merged branch | [executor] | 0.5 h | $0 |
| R1.2 | Outcomes migration + BTS loader + scorer + weekly report | [implementer] | 2 d | $0 |
| H1.a | Pre-register `PROTOCOL_B5_HOLDOUT.md` | [planner/reviewer] | 0.5 d | $0 |
| A7/L0 | Missingness and pairing-stability study | [implementer] → `reviewer` | 1.5 d | $0 |
| R2.0 | Byte/message histogram of the smoke archives (offline) | [executor] | 1 h | $0 |
| R2.1 | Consumer: zstd stream, filters, downsampling, LADD module, heartbeat, auth park, prod-role guard | [implementer] | 2 d | $0 |
| R2.2 | `Dockerfile.swim`, two Railway services, R2 uploader, lifecycle rules, healthchecks.io | [implementer] | 1-1.5 d | $25-40/month + $1-5/month |
| R2.3 | Review before always-on | [planner/reviewer] (`reviewer`) | 0.5 d | $0 |
| R3 | Live extractor → Postgres (flight state, TMI, feed status); outage endpoint; app banner (app repo, small) | [implementer] | 3 d | $0 |
| R1.3 | SWIM provisional outcome loader | [implementer] | 1 d | $0 |
| H1.b | Run the holdout when BTS September lands; report | [implementer] → `reviewer` | 1 d | $0 |
| A6S.1 | Freeze top-100 airports and adsb.lol tiles (JSON + test) | [executor] | 2 h | $0 |
| A6S.2 | adsb.lol client; A6 poller (O + L) running from Nov 11; shared detector module with A5b regression test | [implementer] | 2 d | $0 |
| A6S.3 | AeroDataBox client + pilot pulls (7-day raw deletion) | [implementer] | 0.5 d | $8 |
| A6S.4 | Analysis: A6S metrics, L1 linkage metrics, report | [implementer] → `reviewer` | 2 d | $0 |
| S1 | STDDS smoke + histogram; third service | [executor] + [implementer] | 1.5 d | +$7-13/month |
| P0 | Run-up serving design | [planner/reviewer] | 0.5 d | $0 |
| P1-P3 | Adapters + parity tests, refit, shadow mode | [implementer] | 6-9 d | $0 (+$75-99/month only if L1 adopts AeroDataBox) |
| P4 | Go-live check | [planner/reviewer] | 0.5 d | $0 |
| V1 | SWIM vs BTS November parity | [implementer] → `reviewer` | 1 d | $0 |

Calendar (if started now): R1.2, H1.a, A7 and R2 this week and next; archiver live by about Nov 11; holdout run about
Nov 17-20; A6S/L1 days Nov 18-19 and Dec 1-3; shadow run-up from mid-December; V1 in mid-January.

Recurring cost after all steps: **about $30-55/month** (Railway + R2), plus $75-99/month only if AeroDataBox is adopted.
OpenSky stays free during the MVP; its licence cost before launch is unknown.

---

## 9. Decisions needed from Sean

1. **AeroDataBox $8 pilot:** approve, to be spent on Nov 18-19 (recommended). Production ($75-99/month) only if L1's rule says adopt.
2. **Hosting:** approve Railway for the SWIM consumers, about $30-55/month all-in (Hetzner about $10/month is the fallback).
   Check whether the Railway account is on Hobby or Pro.
3. **Cloudflare R2:** create an account (free tier, card on file) or allow Backblaze B2.
4. **LADD list:** register as a data recipient if the FAA requires a named person (we will check the process first).
5. **Alerts:** where to send them (email, SMS, Discord).
6. **A6S/L1 dates:** Nov 18, 19, Dec 1, 2, 3 (backups Dec 8-10).
7. **Interim outcomes via AeroAPI** for app-tracked flights until SWIM outcomes are live: pennies per flight, but only allowed
   commercially on the Standard tier. Which tier does the app use?
8. **Outage notice** wording in the app (an FAA obligation), and whether run-up predictions launch as "beta" after shadow mode.
9. **Holdout policy after September** (recommended: October is the second confirmation; the next clean holdout starts 2027-01).

## 10. Not verified
- TFMS full-day OUT/OFF/ON/IN fill, schedule lead, and whether TFMS Flow carries ATCSCC advisory text (A6S, R2.0).
- SFDPS registration fill by lead time (L1).
- STDDS volume and content (S1).
- The LADD list sign-up process for SWIM data consumers.
- Railway, R2 and Hetzner prices are from their pricing pages and third-party summaries as of 2026-11-05; Hetzner changed
  prices twice in 2026, so check before buying.
- AeroDataBox tail fill and its source (the pilot measures it).

Sources for prices: [Railway pricing](https://docs.railway.com/reference/pricing/plans),
[Cloudflare R2 pricing](https://developers.cloudflare.com/r2/pricing), [Hetzner CX23 summary](https://www.bitdoze.com/md/hetzner-cloud-review.md),
[Hetzner 2026 price changes](https://findstack.com/resources/hetzner-price-increase-2026).

---

# Appendices: research from the A6 plan (2026-11-05), still current

Changes since then: OpenSky polling is allowed for the MVP by Sean's decision (the licence wording in A.1 still applies
before launch); no OpenSky email is planned; the adsb.lol attribution line is approved; scraping is ruled out.

Section references inside the appendices (§4, §5, §6, P1) point to the old A6 plan's numbering; its P1 pilot is superseded by L1 (section 3.4) and its A6 protocol by A6S (section 2).

## Appendix A. Volunteer ADS-B networks and commercial terms

### A.1 Summary table

| Network | API (current) | Rate limit | Licence / terms | Commercial use for FlyRely? | Sources |
|---|---|---|---|---|---|
| **OpenSky** | REST `/states/all` (bounding box; CONUS = 4 credits), `/flights/*`, `/tracks/*` | Standard user 4,000 credits/day per endpoint group; active feeder 8,000; **licensed 14,400 (refilled hourly)** | Non-profit research and education only. *"Any use by a for-profit or commercial entity … requires written permission and a license"*. Operational REST use *"requires a written license, regardless of the entity's non-profit status."* No redistribution. Publications must cite and anonymise. | **No, not without a written licence.** There is no public price; contact `contact@opensky-network.org`. The agreement has a "commercial internal testing and evaluation" category that needs written permission, which is the right request for A6. | [Terms](https://opensky-network.org/about/terms-of-use), [REST docs](https://openskynetwork.github.io/opensky-api/rest.html) |
| **adsb.lol** | `api.adsb.lol`: `/v2/point/{lat}/{lon}/{radius}` (≤250 nm), `/v2/hex/{icao}`, `/v2/callsign/{cs}`, `/v2/reg/{reg}`, `/api/0/routeset`. Compatible with the ADSBexchange v2 format. | "Dynamic based on load"; a key will be required in future (obtained by feeding) | **ODbL 1.0** for *"the API as well as all data ADSB.lol makes public"*. The ToS adds: *"If you want to use the API for production purposes, please contact me."* Daily historical dumps are also ODbL (`globe_history_*` on GitHub). | **Yes.** ODbL allows commercial use with (a) attribution and (b) share-alike for any *derived database* we use publicly. Model outputs shown to users are "Produced Works", which need only an attribution notice. **Open point:** if we offer a merged event table publicly or to partners, it must be ODbL. Courtesy email to the maintainer before production. | [OpenAPI spec](https://api.adsb.lol/api/openapi.json), [API page](https://www.adsb.lol/docs/open-data/api/), [historical](https://www.adsb.lol/docs/open-data/historical/), [ODbL](https://opendatacommons.org/licenses/odbl/1-0/) |
| **adsb.fi** | `opendata.adsb.fi/api/v3/lat/{lat}/lon/{lon}/dist/{nm}` (≤250 nm), `/v2/hex`, `/v2/callsign`, … `/v2/snapshot` is for feeders only | 1 req/s public; invalid requests count | *"adsb.fi open data is for personal, non-commercial use only."* No resale; attribution required. | **No.** Contact them for commercial terms. | [adsbfi/opendata](https://github.com/adsbfi/opendata) |
| **airplanes.live** | `api.airplanes.live/v2/point/…` (≤250 nm), hex, callsign, … | 1 req/s | The API guide lists "No SLA, No Uptime Guarantee, **Non-Commercial Use**". A third-party table says internal business use may be allowed, but that is unconfirmed. The site returned 403 to automated fetches, so the wording was not verified first-hand. | **No, not without written permission.** | [API guide](https://airplanes.live/api-guide/) (via search snippet), [comparison 2026-10-28](https://flyitalyadsb.com/en/dati/confronto/) |
| **ADS-B Exchange** (owned by JETNET) | Community API on RapidAPI ($10 per 10k requests/month); Enterprise (whole-world call, history, event stream) | per plan | The Community API is non-commercial. *"A project for a company needs a commercial license regardless of if the end results are being sold or used internally."* | **Paid licence only** (quote on request). Third-party figures range from $100 to $500+ per month. Unverified. | [support article](https://support.adsbexchange.com/hc/en-us/articles/37363886073613), [comparison](https://flyitalyadsb.com/en/dati/confronto/) |
| **ADSB One** | archived 2026; merged into airplanes.live | n/a | n/a | No | [api-archive](https://github.com/airplanes-live/api-archive) |
| **FlyItalyADSB** | REST, radius ≤500 km; Pro gives the whole world in one request | Free 30 req/min, 300k/month | CC BY-SA 4.0. *"Allowed on every plan, Free included."* | **Yes**, but its US receiver density is unknown (about 3,200 receivers worldwide, mostly in Europe). It could be a cheap third arm. €9/month Starter, €49/month Pro. | [comparison](https://flyitalyadsb.com/en/dati/confronto/) |
| **Flightradar24 API** | credit-based, Explorer/Essential/Advanced | per credits | Explorer is $9/month for 60k credits (blog). I did not verify the commercial terms. | Unverified. It is a paid product, not a volunteer network. | [FR24 blog](https://www.flightradar24.com/blog/b2b/flightradar24-api/) |

**Conclusion.** The only free network whose licence clearly allows FlyRely's commercial use is **adsb.lol (ODbL)**. OpenSky needs a written licence. adsb.fi and airplanes.live are non-commercial. ADS-B Exchange is paid.

**Also note:** adsb.lol publishes **daily full-sky historical traces under ODbL** (`globe_history_2026`). They could be used to backfill multi-day validation, which would replace OpenSky's research-only Trino history. I have not verified their size; expect tens of GB per day compressed. Download only the airports and days we need.

### A.2 Merge design (by icao24 and timestamp)

Unified observation row: `source, icao24 (lowercase hex), t_obs (UTC s), lat, lon, alt_baro_m, alt_geom_m, on_ground, vr_ms, gs, callsign, snapshot_t`.

- **OpenSky** state vector: `t_obs = time_position` (falling back to `last_contact`), and `on_ground` comes from field 8.
- **adsb.lol** aircraft: `t_obs = response.now − seen_pos`. `on_ground = (alt_baro == "ground")`. `vr = baro_rate × 0.00508`. Use only `type` in {adsb_icao, adsb_icao_nt, mlat, adsr_icao}, and keep the `type` field.
- **Dedupe:** treat two rows from different sources with the same icao24, `|Δt| ≤ 5 s` and distance ≤ 300 m as one physical observation. Keep the fresher one and record both sources.
- Run event detection on the **merged** time series for each icao24, sorted by `t_obs`. The A5b nearest-airport guard and the 30-minute dedupe stay unchanged.

**Measuring the gain from each source**, per airport and day:
1. Run the detectors on each source alone (O, L) and on the union (O∪L).
2. **Marginal gain** of a source = ratio(union) − ratio(without that source). Report it for departures, arrivals, and bracketed departures.
3. **Unique attribution:** count events whose bracketing observation pair needs at least one observation found only in that source.
4. **Surface coverage:** on-ground airline vectors per source per airport-hour within 5 km. This directly tests the CLT/DTW gap, where A5b saw 8 and 16 ground rows over the whole run.
5. **Bracket width:** distribution per source and for the union.

---

## Appendix B. Schedules: "what are we looking for, and when?"

US scale for reference: about 21,700 BTS-reporting-carrier flights per day (August 2026). The **top 100 origins carry 89.3%** of departures, the top 150 carry 95.1%, and the 12 A5b hubs carry 36.6%. These figures come from `data/bts/parquet_v2/2026-08.parquet`.

| Rank | Source | Completeness | Cost (≈100 airports, one month) | Terms | Tail? | Inbound flight? | Actual times? |
|---|---|---|---|---|---|---|---|
| 1 | **FAA SWIM TFMS flight data** (via SCDS) | All IFR flights nationwide: flight plans, departure and arrival messages, tracks. Schedule data feeds TFMS per a NASA slide; check the TFMS schedule ("flight create") content after onboarding. | **$0**, plus hosting for an always-on JMS client (about $5–20/month) | Free. Commercial non-NAS-impacting use allowed. LADD blocking required. Notify our users of outages. See §6. | Partial. SFDPS carries `REG` (Item 18) when filed, and TFMS terminal / TFDM carry tails at some sites. **Fill rate unverified; measure it.** | Not directly; derive from tail sequence | **Yes** (departure and arrival messages; quality to be measured) |
| 2 | **AeroDataBox** FIDS `/flights/airports/{iata}/{from}/{to}` (Tier 2 = 2 units/call; `direction=Both`; no pagination) | US: "100% schedules, 86% live status" (their table, updated 2026-06-03). Live layer usually covers up to tomorrow. | Pilot: **Pro $8/month** (5k units). Production with hourly refresh: about 108k units/month → **Ultra 2 $75** (API.Market). With 30-minute refresh: about 216k → **Growth $99** (direct). | Paid plans allow commercial use. Basic free does **not**. B2B sublicensing only on Mega and Scale. **Caching: Standard plans keep data only 7 days**; Ultra 2, Mega and Growth allow indefinite storage while subscribed. | `aircraft.reg` and `aircraft.modeS` (the icao24 hex directly), "sometimes" from the live layer, always from ADS-B. **Fill rate by lead time is the P1 question.** | No field. Derive it from the same tail on the arrivals board. | `revisedTime` (gate, or mixed), `runwayTime` (takeoff/landing). The `quality` array says Basic, Live or Approximate. |
| 3 | **FlightAware AeroAPI** | Excellent | Standard tier ($100/month minimum; Personal tier is non-commercial). Per 15-record set: `scheduled_departures` / `scheduled_arrivals` / `departures` / `arrivals` $0.005; `/schedules/{d1}/{d2}` $0.020; `/flights/{ident}` $0.005; history $0.020. One nationwide pass of dep+arr boards is about 2 × 19.4k / 15 ≈ 2,600 sets/day ≈ **$390/month**. With hourly refresh (each flight seen about 6 times) it is about **$2,300 list, about $1,900 after volume discount**. `/schedules` (no tail) ≈ $780/month. | Standard allows internal use and B2C | `registration` on flight objects (airport endpoints); not on `/schedules` | **Yes: `inbound_fa_flight_id`** | Yes (`actual_off`, `actual_on`, `actual_out`, `actual_in`) |
| 4 | **AviationStack** `/v1/timetable?iataCode=&type=`, `/v1/flights`, `/v1/flightsFuture` | Unknown; data sources undisclosed | Free: 100 req/month, **personal use only**, no schedules endpoints. **Basic $49.99/month** (10k req, commercial). Professional $149.99 (50k). Overage $0.012–0.02 per request. | Commercial from Basic | Documented response has `aircraft.registration` and `aircraft.icao24`. **Fill rate unverified.** | No | `actual`, `estimated_runway`, `actual_runway` fields (unverified quality) |
| 5 | **BTS On-Time (Reporting Carrier)** | All reporting carriers | $0 | Public domain | **Yes, the actual tail** | Derivable (same tail, previous leg) | Yes (out, off, on, in) |
| 6 | OpenSky routes / vradarserver standing-data (behind adsb.lol `/api/0/routeset`) | Callsign → route only, crowd-sourced, no times or dates | $0 | OpenSky: licence needed. standing-data: licence not checked. | No | No | No |
| 7 | Airline-published schedules | Airlines publish through OAG and Cirium (paid). No free machine-readable US airline feed was found. | n/a | n/a | No | No | No |
| 8 | ASDI | **Decommissioned 2016**; superseded by TFMS data via SWIM | n/a | n/a | n/a | n/a | n/a |
| 9 | Airport / airline web boards | see §5 | $0 | Prohibited or permission needed | No | Airline pages only | Gate times |

More on rows 5 and 8:
- **BTS lag.** The newest month is **August 2026** (released 2026-10-19). September 2026 should appear around mid to late November. BTS is two months late, so it can never be a same-day denominator. It remains the training truth and the **A7** source. Holdout reminder: data from 2026-09 onward stays locked for the B-series model evaluations.
- **ASDI sources:** [2016 SWIM Users Forum](https://www.faa.gov/sites/faa.gov/files/2022-12/2016-June-SWIM-Users-Forum-Briefing.pdf) ("ASDI Decommission Complete"), [2015 briefing](https://www.faa.gov/sites/faa.gov/files/2022-12/2015-April-SWIM_Users_Forum-Briefing.pdf). There is also a planned TFMS → FMDS transition ([SWIFT focus group](https://www.faa.gov/air_traffic/technology/swim/swift/swift-focus-group-afternoon-session)), which is a schema-change risk.

Sources:
- AeroAPI: [pricing page](https://www.flightaware.com/commercial/aeroapi/); OpenAPI spec v4.17.1, `aeroapi-openapi.yml`. The spec says `scheduled_departures` accepts a start no further than 2 days ahead, and `/schedules` has no registration field.
- AeroDataBox: [pricing](https://aerodatabox.com/pricing), [coverage](https://www.aerodatabox.com/data-coverage), [OpenAPI (direct)](https://doc.aerodatabox.com/docs/openapi-direct-v1.json). Schemas checked: `FlightAircraftContract.reg/modeS`, `FlightAirportMovementContract.revisedTime/runwayTime/quality`, and `FlightContract.flightPlan`, *"as filed with the ATC provider … mainland U.S."*. That suggests AeroDataBox itself consumes FAA data.
- AviationStack: [pricing](https://aviationstack.com/pricing), [timetable blog](https://blog.apilayer.com/aviationstack-new-features-updates/).
- BTS: [August 2026 release](https://www.bts.gov/newsroom/airline-service-quality-performance-data-august-2026).

**Best schedule source:** SWIM TFMS if onboarding works (free). Otherwise, or until then, **AeroDataBox at $75–99/month** for ~100 airports. AeroAPI is the best per-flight inbound source, used only on demand.

### B.1 AeroDataBox and AviationStack as gap-closers (coordinator request)

| | AeroDataBox | AviationStack |
|---|---|---|
| Free tier | Basic: 400 units/month, 1 req/s, **non-commercial**, FIDS window 12 h | 100 req/month, personal only, **no timetable or schedules** |
| Cheapest commercial tier | Pro $7.50–8 (5k units, 7-day caching) | Basic $49.99 (10k req) |
| ~100-airport production | $75 (Ultra 2) to $99 (Growth) | Professional $149.99 or more. If ~100 results per page hold (unverified), 100 airports × 2 directions × refreshes is far over 50k/month, so Business at $499.99. |
| FIDS covers all major US airports? | Claims 100% schedules and 86% live for the US. The free `/health/services/feeds/{service}/airports` endpoint lists the airports per layer, so check our top 100 at 0 units. | Not documented |
| Tail fill and lead time | `reg` from the live layer "sometimes", usually up to tomorrow. **P1 measures it.** | Unverified; not measurable on the free tier |
| Actual times and source | `revisedTime` and `runwayTime`; sources "aggregated from third-party suppliers" plus ADS-B; `flightPlan` from US ATC filings | `actual`, `actual_runway`; sources undisclosed |
| Verdict | **Pilot it (P1).** | Backup only. Pilot it only if P1 fails, at $49.99 for one month. |

### B.2 Combined pipeline sketch (schedule + tail + live position → "inbound landed / late")

```
every 60 min (30 min at hubs), per airport A in top-100:
  ADB FIDS(A, now-1h .. now+6h, Both)  -> flights{number, callsign, sched/revised/runway times, reg, modeS, status}
  upsert into flights table (keyed by operating flight number + date + origin)

for each departure D at A with T-3h or T-1h horizon due:
  tail = D.reg (or modeS)
  inbound I = latest arrival at A on today's boards with same reg and sched_arr < D.sched_dep
             (fallback: AeroAPI /flights/{ident} -> inbound_fa_flight_id, $0.005-0.01, only for user-tracked flights)
  live = latest merged ADS-B observation for icao24 = modeS (OpenSky CONUS snapshot if licensed, else adsb.lol /v2/hex)
  features:
    inbound_landed      = I.runwayTime(arrival) known OR live.on_ground near A
    inbound_eta_delay   = (ETA from live position/ADB revisedTime) - I.sched_arr
    turn_slack          = D.sched_dep - inbound ETA - min_turn(type, airline)
    inbound_unknown     = no tail or no position  (explicit missing-indicator)
```

Monthly cost at about 100 airports:
- AeroDataBox: $75 (hourly) or $99 (30-minute).
- Live positions: OpenSky CONUS, whose licence fee is unknown, or adsb.lol at $0.
- AeroAPI fallback: about $0.01–0.04 per tracked flight.
- Hosting: about $10.

**Total ≈ $85–120/month plus any OpenSky licence.** If SWIM gives tails and departure/arrival messages with good fill, AeroDataBox drops to schedule-only, or out altogether.

---
## Appendix C. Event definitions (A6 §6.5, unchanged)

Airline callsign and nearest-airport attribution are exactly as in A5b (3-letter BTS carrier ICAO code + 1–4 digits; events attributed to the nearest scheduled-service airport; `MAX_GAP` 600 s; one event per aircraft, type and 30 minutes).

- **Departure detector D0 (A5b):** an airborne observation within 15 km and ≤1,200 m AGL, either preceded within 600 s by a ground observation within 5 km (bracketed) or with vr > +2 m/s and no prior sighting (first-seen-climbing). Takeoff time and bracket as in A5b.
- **Departure detector D1 (looser, from the A5b review):** D0 **OR** a consecutive pair for the same icao24 where the earlier observation is **on ground within 5 km** of the airport and the later is **airborne at any distance or height**, with Δt ≤ 600 s and no ground observation at another airport in between.
  - Inferred takeoff = max(t_ground, t_air − min(600, AGL_air / max(vr, 5 m/s))).
  - Bracket = t_air − t_ground.
- **Arrival detector A0 (A5b):** unchanged.
- **Arrival detector A1 (looser, secondary):** A0 **OR** an airborne observation at any distance followed within 600 s by a ground observation within 5 km.
- Every detector is run on: O alone, L alone, O∪L (merged per §1.2). Each at cadences 1 min (block only), 2, 4 and 6 min, obtained by thinning the same rounds.


## Appendix D. Scraping airport and airline boards (coordinator request): not recommended

What I checked. Each site got one plain fetch of its robots.txt, flight page and terms page, with a browser user agent. Boards were not polled and JS endpoints were not reverse-engineered.

| Hub | robots.txt | Board page | Behind it | Terms | Tail / inbound | Actuals |
|---|---|---|---|---|---|---|
| ATL (atl.com) | **403** (WAF blocks non-browser clients even for robots.txt) | 403 | not determinable | 403 | Not shown on airport FIDS generally | Status / "departed" time |
| ORD / MDW (flychicago.com) | allows all except SharePoint internals | old URLs 404; JS app | not determinable | not found | no | gate times |
| DFW (dfwairport.com) | `Allow: /` | `/flights/` JS-rendered (date picker, today/tomorrow) | not in static HTML or bundles | legal / disclaimer pages need click-through acceptance | no | status |
| DEN (flydenver.com) | **403** WAF | 403 | not determinable | 403 | no | status |
| CLT (cltairport.com) | `Allow: /` | `/flights/` JS-rendered | not determinable | "improper use constitutes abuse and may be subject to prosecution"; all rights reserved | no | status |
| LGA (laguardiaairport.com / PANYNJ) | `Disallow:` (empty) | `/flight/search` JS component | PANYNJ in-house component; backend not visible | PANYNJ website disclaimer; use is acceptance of terms | no | status |
| MSP (mspairport.com) | Drupal defaults; blocks some named scraper bots | old URL 404 | not determinable | **"For commercial uses … permission is required."** ([MAC terms](https://metroairports.org/privacy-policy-terms-use)) | no | status |
| DTW (metroairport.com) | Drupal defaults | `/flights` JS | not determinable | terms URL 404 | no | status |
| SEA (portseattle.org) | Drupal defaults | `/sea/flight-status` JS | not determinable | terms URL 404 | no | status |
| BNA (flybna.com) | `Allow: /` | connection failed | — | — | — | — |
| RDU (rdu.com) | **`Disallow: /api/`** | `/flights` JS | its own `/api/` (disallowed) | — | no | status |

Notes on the table:
- Airport FIDS data generally comes from airline host feeds through vendors such as SITA, Collins/ARINC, OAG and FlightView, under licence to the airport. The sites do not offer it for reuse.
- No airport board checked shows the tail or the inbound flight.

**Airline status pages** (AA, DL, UA) do show "where is my plane" or the inbound flight, which is the most valuable field. However:
- AA's terms bar scraping *"without an express written agreement"* ([aa.com legal](https://www.aa.com/i18n/customer-service/support/legal-information.jsp)), and AA has won an injunction against a scraper before ([AA v. FareChase, 2003](https://eff.org/files/filenode/AA_v_Farechase/20030310_prelim_inj.pdf)).
- Delta's professional-site terms ban robots and scrapers ([pro.delta.com](https://pro.delta.com/content/agency/global/en/legal.html)).
- These pages sit behind bot management such as Akamai.

High value, but **high legal and blocking risk for a commercial app**.

**Comparison (≈100 airports):**

| | Scraping | AeroDataBox |
|---|---|---|
| Cash cost | $0 | $75–99/month |
| Build and maintenance | About 60+ distinct front ends, JS-rendered, WAF-protected (ATL, DEN). Breaks whenever a site is redesigned; ongoing [implementer] time every month. | One documented API |
| Coverage | Uneven; airport boards have no tails; airline pages cover only that airline | 100% schedules, 86% live claimed; tails and modeS |
| Legal | MSP needs permission for commercial use; CLT warns of prosecution; RDU disallows `/api/`; airlines prohibit scraping | Commercial licence included |

**Recommendation: AeroDataBox. Scraping does not make the cut, so no scraping pilot is written.**

