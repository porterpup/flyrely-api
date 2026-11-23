# Protocol PAID TRIAL: does FlightAware AeroAPI name a flight's aircraft and inbound flight earlier and more accurately than free SWIM linkage?

Dated 2026-11-22. Pre-registered: this file is committed BEFORE any trial call exists (trial days are 2026-12-01 to 2026-12-03).
Sean approved this trial for planning on 2026-11-22. Data collection waits only for the portal checks in the Gate (S0), which Sean's risk acceptance otherwise satisfies.

## Question

Does FlightAware AeroAPI name a flight's operating aircraft (its tail) and its inbound flight earlier and more accurately than our
free SWIM linkage (the CHAIN methods, see `PROTOCOL_CHAIN.md`)?

## Gate (S0)

- No API call happens until FlightAware gives written permission for an evaluation on the Personal tier or with trial credit,
  or Sean explicitly accepts the risk.
- Record the outcome in HANDOFF.md.
- S0 is satisfied: on 2026-11-22 Sean explicitly accepted the risk of running the evaluation on the Personal tier without asking
  FlightAware first, because FlyRely is not selling anything yet. The Personal tier's terms describe it as non-commercial use, and
  this was weighed.
- Before the run, verify in the AeroAPI portal: prices, the rate limit, when the credit resets, and the data-retention terms.

## Provider and endpoint

- AeroAPI `/flights/{ident}`, queried ONLY by ICAO callsign (flight number) plus a time window, with `max_pages=1`.
- Never query by registration. Never send a tail off the Mac.
- No AeroDataBox arm. That stays a separate $8 pilot decision.

## Days

Tue 2026-12-01 to Thu 2026-12-03 UTC. This avoids Thanksgiving week.

## Sample

- 60 flights per day, 180 in total, departing 15:00 to 01:00 UTC from the 12 A5b hubs.
- Daily quotas: AAL 9, DAL 9, UAL 9, SWA 9, regionals 15, JBU 3, ASA 3, FFT 3. AAY is excluded.
- A fixed-seed random draw picks flight numbers that flew the same origin and time slot on D-2 and D-1.
- The manifest is written before the first call.
- Flights that do not operate on day D are excluded and counted.

## Query times

T-24h, T-6h, T-3h and T-1h before scheduled departure. That is 4 calls per flight, 720 in total, about $3.60 at roughly
$0.005 per result set (to be verified in the portal, S0).

## Guards

- A hard cap of 780 calls.
- At most 1 call every 8 s.
- Stop on a 429, 401 or 403.
- A slot that would fire more than 15 min late is recorded as missed.
- A carrier allow-list.
- Legs in `blocked_legs` are skipped.
- A block-list check before each query and on each response. A response that hits it is discarded unstored.

Honest note: `data/ladd_blocklist.txt` does not exist yet, so the file-based block-list check has nothing to read today.
The safety comes from querying only airline flight numbers from approved scheduled US carriers (the allow-list above).

## Privacy

- The API key comes only from the `AEROAPI_KEY` env var, is sent only in the `x-apikey` header, and is never logged.
- Generic User-Agent, and no user data.
- The sample comes from the public schedule, never from users' saved flights.
- Storage is local only, in `data/linkage/paid_trial.sqlite`. Keep only the parsed `registration`, `inbound_fa_flight_id` and
  `fa_flight_id`. No raw response bodies.
- Purge after scoring, or by the provider's retention limit, whichever is sooner.
- Committed results are aggregates only.

## Truth

- Same truth as CHAIN `load()`: SMES surface-event icao24, else the last SFDPS value for the leg. Legs with no truth are excluded and counted.
- Provider tail: map registration to icao24 locally (from store pairs, else a local N-number to hex function).
- Provider inbound: parse the ident from `inbound_fa_flight_id` and match it to our leg (same callsign, arriving at this leg's origin
  within 12 h before). It counts as correct if that leg's truth equals this leg's truth.

## SWIM comparison

- The CHAIN methods M1/M4/M2/M3, evaluated at the provider's actual request time, rounded down to 5 min.
- This trial never reads the free eval week 2026-11-16 to 2026-11-22.

## Metrics

Reported per cutoff, pooled and split mainline vs regional:

- coverage, precision, and effective (= correct share);
- tail and inbound reported separately;
- for SWIM, for the provider, and for the union (SWIM M1 first, then the provider, then CHAIN M4/M2).

Lead time:

- SWIM M1 lead = scheduled departure minus first SFDPS receive time.
- Provider lead = the earliest cutoff at which it was correct.
- Also report the share of flights where the provider was correct before SFDPS had filed.

## Decision rule

Pooled, applied as written, with no re-tuning:

- **WORTH PAYING**: all three hold:
  - the union meets the CHAIN bar (effective >= 0.50 at T-3h and >= 0.70 at T-1h; precision >= 0.80 at T-3h and >= 0.90 at T-1h);
  - provider precision >= 0.90 at T-3h;
  - the union's effective at T-3h is at least 0.15 above SWIM alone.
- **NOT WORTH IT**: the T-3h gain is < 0.05, or provider precision is < 0.85.
- **PARTIAL**: otherwise. Use AeroAPI only for flights users track.
- T-24h and T-6h are informational only.

The rule decides only whether the data is good enough. Spending stays Sean's decision (the roadmap estimates about
$1,900 to $2,300/month for board-wide use).

## Statistical note

180 flights gives about +/-7 points on a pooled precision. Per-carrier numbers are reported but not used for the decision.

## Outputs

`linkage/results/paid_trial_2026-12-01_2026-12-03.json`, aggregates only.
