# A5b: can free OpenSky live state vectors detect departures and arrivals at our hubs?

## PROTOCOL (pre-registered before any polling; committed first)

**Question.** The T-3h and T-1h features need (a) whether a flight's inbound aircraft has landed, and (b) whether recent departures from the hub are leaving late. Can free OpenSky `/states/all` produce those signals at hub airports, reliably and cheaply enough?

**Data collection.**
- 12 hubs of different sizes: ATL, ORD, DFW, DEN, CLT, LGA, MSP, DTW, SEA (large), MDW, BNA, RDU (medium and focus-city). Includes LGA, ATL and CLT, which behaved very differently in A5.
- One box per hub, about 1.0 x 1.1 degrees centred on the airport (about 1 square degree, so 1 credit per call). Every hub is polled every 120 seconds, starting each round on a 120-second clock.
- Every 5th round, one CONUS call (4 credits) is taken too, to check that boxes and CONUS return the same state vectors (this decides whether boxes are worth their cost against one CONUS call).
- Two windows: an evening window starting at the beginning of the run (about 02:50 UTC on 2026-11-04, which is about 9:50 pm US Eastern), and a daytime window from 15:00 UTC (10:00 am Eastern, 7:00 am Pacific) for about 2.5 hours.
- Optional: up to 4 `/flights/departure` calls (30 credits each, `/flights` bucket kept above 1,500 remaining) for ATL, LGA, CLT, ORD right after the daytime window, to compare with state-vector departure detection.
- **Budget:** at most 1,500 credits from `/states` in this phase (cap is 165 already spent + 1,500), floor of 1,500 remaining in the bucket. The poller stops itself on either condition and on any 429. Every response is cached.

**Event definitions (fixed now).** Airline state vectors are those with a callsign of a BTS carrier's 3-letter ICAO code plus 1 to 4 digits. Within a hub box, per aircraft (icao24) in time order:
- *Departure detected* = an airborne observation (on_ground false) within 15 km of the airport and no more than 1,200 m above airport elevation, either preceded by a ground observation within 5 km (bracketed), or with a climb rate above +2 m/s (first-seen-climbing). Inferred takeoff time = the airborne time minus the time needed to reach that height at the observed climb rate (capped at 120 s); the **bracket width** (airborne time minus last ground time) is reported as the timing uncertainty.
- *Arrival detected* = a ground observation within 5 km preceded by an airborne observation within 25 km, or a descending (below -1 m/s), low (under 600 m above elevation) airborne observation within 12 km as the last sighting. Landing time = first ground observation, with the preceding airborne time as the bracket.

**Denominator (same-weekday recurring schedule) and metrics.**
- Denominator: August 2026 BTS domestic flights on the same weekday, re-timed to today's local schedule with the November UTC offset, averaged over the same-weekday Augusts. Limits: seasonal schedule changes and flights that do not operate on the date (+/-10-15% in aggregate, worse per flight). BTS has no truth for today.
- **Primary metric (aggregate, robust to which flights run):** detected departures divided by expected departures from the hub, and detected arrivals divided by expected arrivals, per hub and per hour, mainline and regional separately. Count only the interior of each window (starting 30 minutes after the window start, ending 30 minutes before its end).
- **Secondary metric (flight identity, lower bound):** share of recurring flights (present on at least 75% of same-weekday Augusts) matched by callsign to a detected event within 60 minutes of scheduled time. Reported unconditionally and with its limits (non-operating flights count as misses).
- **Timing:** bracket width (median, p90) for departures and arrivals; and the difference between the detected takeoff minus scheduled departure distribution and the August BTS distribution (wheels-off minus scheduled), as a distribution-level plausibility check, not truth.
- **Inbound-aircraft linkage:** for each recurring flight, the modal August tail on that flight and weekday (tail stability = share of Augusts with that tail), converted to icao24 with the A5 N-number algorithm. Report (1) tail stability; (2) for departures detected and matched by callsign, how often the observed icao24 equals the modal tail's icao24; (3) the share of detected departures whose aircraft also had an arrival detected at the same hub earlier in the same window (observed turnarounds).
- **Interval sensitivity:** repeat the primary metric on every 2nd and every 3rd round (4- and 6-minute spacing), so the effect of cadence is measured on the same data.
- **Credit cost:** per hub per day at 1, 2, 5 and 10 minute intervals, and the number of hubs that fit in 4,000, compared with one CONUS call per round.
- **Optional comparison:** fraction of `/flights/departure` records (firstSeen near the hub) that the state-vector detector also found, and the signed difference in takeoff time.

**Decision rule written in advance.** "Reliable" for a hub means detected/expected at least 0.85 for departures and arrivals in daytime, with a median bracket width of 3 minutes or less at 2-minute polling. Fewer than 0.7 means not usable. In between is "partial".

---

## RESULTS (written after the poll ran; protocol above unchanged)

Raw output: `backtest/results/a5b_opensky_hubs.json` (`python backtest/opensky_hubs.py analyze`, offline, reproducible from the cached responses). The earlier commit message "WIP ... not yet run" is outdated: polling ran on 2026-11-04.

**What ran.** 112 rounds at 120 s for 12 hubs, ended cleanly ("DONE"). Evening block 02:48-04:02 UTC (38 rounds), daytime block 14:59-17:26 UTC (74 rounds). 46,322 airline-callsign state rows; 1,440 departure and 1,227 arrival events detected at 2-minute cadence. Cost: 1,432 `/states` credits this phase (cap 1,500; 2,403 remaining afterward). The optional `/flights/departure` comparison was NOT run.

**Interior windows.** The evening interior window is only about 14 minutes (03:18-03:32 UTC, 7 rounds, 73 expected departures) because the protocol trims 30 minutes from each end of a 76-minute block. It is too small to judge anything. The daytime interior window is 15:30-16:56 UTC (86 minutes, 640 expected departures, 544 expected arrivals), so **the decision rule is applied to the daytime window only** (the rule says "in daytime").

### Primary metric: detected / expected, daytime interior window, 2-minute cadence

All hubs together: departures 574 / 640.3 = **0.90** (mainline 0.90, regional 0.89); arrivals 529 / 543.8 = **0.97** (mainline 0.99, regional 0.93). Evening (noisy): departures 0.78, arrivals 0.86.

| Hub | Dep all | Dep mainline | Dep regional | Arr all | Arr mainline | Arr regional | Class |
|---|---|---|---|---|---|---|---|
| ATL | 0.66 | 0.65 | 0.72 | 0.79 | 0.80 | 0.71 | not usable |
| ORD | 1.02 | 1.14 | 0.90 | 1.07 | 1.21 | 0.95 | partial |
| DFW | 0.98 | 1.00 | 0.95 | 1.01 | 1.09 | 0.84 | **reliable** |
| DEN | 0.93 | 0.96 | 0.80 | 1.17 | 1.24 | 0.88 | **reliable** |
| CLT | 0.73 | 0.66 | 0.80 | 1.03 | 0.96 | 1.10 | partial |
| LGA | 0.98 | 1.01 | 0.96 | 1.13 | 1.24 | 1.06 | **reliable** |
| MSP | 1.05 | 1.06 | 1.05 | 0.50 | 0.50 | 0.49 | not usable |
| DTW | 1.11 | 1.22 | 1.01 | 0.72 | 0.71 | 0.74 | partial |
| SEA | 1.04 | 1.14 | 0.77 | 0.96 | 0.86 | 1.54 | **reliable** |
| MDW | 0.84 | 0.89 | 0.00 | 0.67 | 0.69 | 0.00 | not usable |
| BNA | 0.79 | 0.79 | 0.75 | 1.00 | 0.94 | 1.11 | partial |
| RDU | 0.66 | 0.43 | 1.33 | 1.10 | 1.37 | 0.60 | not usable |

Per-hub, per-hour, per-carrier-group counts (detected and expected) are in the JSON under `detection_ratio_by_hour` (the daytime window gives one full hour and a 27-minute remainder). Hourly values are noisy (for example MSP arrivals 11/20, then 1/4). Expected counts per hub per hour are roughly 10 to 65, so a single ratio carries large sampling noise on top of the denominator's own +/-10-15%. MDW, BNA, RDU, DTW and the regional columns have expected counts under about 15 to 30 and are indicative only. Ratios above 1 are possible because detections include any BTS-carrier callsign (including international and non-recurring flights) while the denominator is August domestic flights only.

**Numerator definition caveat (read this).** The first version of `analyze` counted only events whose callsign appeared in the BTS schedule for that hub and direction, which gave a much lower total (departures 0.41, arrivals 0.50). That version mixed the secondary (flight-identity) metric into the primary one, contrary to the protocol, which defines the primary metric as an aggregate "robust to which flights run" and defines airline callsigns as a BTS carrier's ICAO code plus 1 to 4 digits. I changed the primary numerator to the protocol definition. This was decided after seeing the first run's numbers, so it is disclosed here. Both versions are saved (`detection_ratio_2min_schedule_callsign_only` is the stricter one). The strict version is a floor, not a detection rate: flight numbers are re-used and re-numbered between August and November, so many real departures fail the callsign test. Under the strict version no hub reaches 0.7 in both directions, so every hub would be "not usable". The truth lies between the two; the secondary metric below shows why the strict figure is pessimistic.

### Secondary metric: flight identity by callsign (lower bound)

Share of recurring flights (present in at least 75% of same-weekday Augusts) matched by callsign to a detected event within 60 minutes of scheduled time, daytime window, departures / arrivals: ATL 0.32 / 0.36, ORD 0.14 / 0.13, DFW 0.09 / 0.25, DEN 0.21 / 0.49, CLT 0.08 / 0.03, LGA 0.40 / 0.39, MSP 0.34 / 0.25, DTW 0.36 / 0.35, SEA 0.59 / 0.56, MDW 0.00 / 0.00, BNA 0.09 / 0.19, RDU 0.10 / 0.31 (both windows in the JSON, `identity_level_lower_bound`). These are low because of August-to-November schedule and flight-number churn plus non-operating days, which count as misses. They are not a detection rate, but they show that **mapping a specific scheduled flight to a callsign via an August schedule is unreliable**; a live feed must key on the callsign observed live, or on the current schedule.

### Timing

- Bracket width (airborne time minus last ground time; bracketed events only), daytime: departures median 132 s, p90 329 s (605 events); arrivals median 119 s, p90 198 s (773 events). Arrival brackets sit at the 120 s polling floor everywhere.
- Only 61% of departures are bracketed (80% of arrivals); the rest were first seen already climbing. Daytime departure bracket medians by hub: ATL 208 s, DFW 158 s, DEN 124 s, LGA 139 s, MSP 123 s, SEA 123 s, BNA 137 s. ORD has only 5 bracketed departures (median 478 s) despite 2,413 ground sightings, and CLT, DTW and MDW have none because those boxes returned very few surface (on-ground) vectors (CLT 8 and DTW 16 ground rows over the whole run, against thousands at ATL, DEN, DFW). That is a coverage property of the OpenSky receiver network, not a code fault.
- Plausibility check (matched departures, daytime, n = 127): detected takeoff minus scheduled has p10 / p50 / p90 of -4.7 / 21 / 52 minutes; August wheels-off minus scheduled for the same flights is 8 / 14.5 / 28.7 minutes. The detected distribution is shifted later and wider, so detected takeoff times are plausible but only a coarse signal. The matched sample is small and biased toward flights whose callsign persisted, so this is a plausibility check only.

### Interval sensitivity (same data, daytime window, all hubs)

| Cadence | Departures | Arrivals |
|---|---|---|
| 2 min (all rounds) | 0.90 | 0.97 |
| 4 min (every 2nd round) | 0.49 | 0.98 |
| 6 min (every 3rd round) | 0.32 | 0.89 |

Arrivals survive coarser polling because the aircraft sits on the ground, and is seen, for a long time. Departures do not: an aircraft is under 1,200 m and within 15 km for only a few minutes, so polling slower than every 2 minutes loses most of them. Departure detection needs 2-minute polling at the slowest; whether 1-minute polling would improve on 0.90 was not tested.

### Inbound-aircraft linkage

- Tail stability in August (share of same-weekday Augusts in which a flight used its modal tail) has a median of 0.25 to 0.33 at every hub, which is the minimum possible with 3 to 4 sample days, so there is no stable tail-to-flight assignment.
- For departures detected and matched by callsign, the observed icao24 equals the modal-August-tail icao24 almost never (0 to 12% across hubs, on a handful of matches each). The August tail is not a usable predictor of the aircraft.
- Observed turnarounds (a detected departure whose aircraft also had an arrival detected at the same hub within the previous 6 hours of the run): ATL 4%, SEA 7%, CLT 9%, ORD 10%, DTW 10%, DFW 15%, MDW 19%, DEN 23%, MSP 23%, LGA 38%, BNA 40%, RDU 75%. This is limited by the short windows, so it is a lower bound, but it shows that linking a departure to its own inbound needs the aircraft to be tracked on its previous leg elsewhere, not just at the hub.

### Box versus one CONUS call

Over 264 box/CONUS comparisons, a hub box and the same area cut from the CONUS response contain essentially the same aircraft (mean icao24 Jaccard 0.98; mean 57.0 versus 57.0 aircraft per box). Boxes carry no information advantage, and one CONUS call at 4 credits covers every hub, so the choice is purely cost: boxes win only below 4 hubs, one CONUS call is better beyond that. (This poll used 12 boxes at 12 credits per round, three times what CONUS would have cost.)

### Credit cost (4,000 credits/day budget)

| Interval | Credits per hub per day (24 h / 16 h) | Hubs fitting (24 h / 16 h) | One CONUS call per round, all hubs (24 h / 16 h) |
|---|---|---|---|
| 1 min | 1,440 / 960 | 2 / 4 | 5,760 / 3,840 |
| 2 min | 720 / 480 | 5 / 8 | 2,880 / 1,920 |
| 5 min | 288 / 192 | 13 / 20 | 1,152 / 768 |
| 10 min | 144 / 96 | 27 / 41 | 576 / 384 |

### Classification (pre-registered rule, daytime window, 2-minute polling)

Rule as applied: **reliable** = departure and arrival ratio both at least 0.85 and median bracket at most 180 s in both directions; **not usable** = either ratio below 0.7; otherwise **partial**. Two interpretation choices, made before classifying: (1) "fewer than 0.7" is read as either direction, since "reliable" needs both; (2) a bracket median that cannot be computed, or that comes from fewer than 10 bracketed events, does not pass. If "not usable" instead required both ratios under 0.7, ATL, MSP, MDW and RDU would be "partial".

- **Reliable (4):** DFW, DEN, LGA, SEA.
- **Partial (4):** ORD (counts fine at 1.02 / 1.07, but almost no bracketed departures), CLT (departures 0.73; no surface vectors, so no brackets), DTW (arrivals 0.72; no surface vectors), BNA (departures 0.79).
- **Not usable (4):** ATL (departures 0.66, median bracket 208 s), MSP (arrivals 0.50), MDW (arrivals 0.67, only 7 expected arrivals, very noisy), RDU (departures 0.66 from only 12 expected, very noisy).

### Verdict

Free OpenSky `/states/all` can feed the T-3h and T-1h features at some hubs, not all. In aggregate it saw 90% of departures and 97% of arrivals at 2-minute polling in the daytime window, but only 4 of 12 hubs pass the pre-registered per-hub rule (DFW, DEN, LGA, SEA); 4 are partial and 4 fail. The failures come from missing ground-vector coverage (CLT, DTW, ORD departures), weak counts at ATL and MSP, and very small expected counts at MDW and RDU.

- **Arrivals** (the "has the inbound landed" signal) are the strong part: about 0.97 overall, 0.98 at 4-minute polling, and 0.89 at 6 minutes.
- **Departures** (the "are departures running late" signal) need 2-minute polling; at 4 minutes the detection ratio halves.
- **Cadence and cost:** 2-minute polling over a 16-hour day costs 1,920 credits as one CONUS call per round covering all hubs (3,840 for 8 hubs as boxes), so CONUS is the sensible design. Arrivals-only at 4-minute CONUS polling would cost about 960 credits for 16 hours.
- **What it cannot do:** link a flight to its inbound aircraft. August tail assignments are unstable (modal tail share 25-33%) and almost never match the observed aircraft, and callsign matching against an August schedule is poor. Tail-level linkage needs a live schedule or tail-assignment source.
- **Limits of this evidence:** one weekday, one 86-minute daytime window, roughly 10 to 95 expected events per hub, a denominator accurate to only +/-10-15%, and an evening window too short to use. Labels for the small hubs (MDW, BNA, RDU, DTW) are low-confidence. Repeating across several days and times of day would be needed before relying on any single hub's label.

### Deviations from the protocol and implementation notes

1. The primary numerator was changed after the first analysis run (see the caveat above). Both versions are reported.
2. The airline callsign pattern was tightened to exactly 1 to 4 digits (the code had allowed one optional trailing letter).
3. Events are attributed to the nearest of all scheduled-service airports, to avoid cross-talk (JFK/EWR near LGA, ORD near MDW). The protocol does not mention this guard; it can only lower detected counts.
4. The secondary metric matches against scheduled time as the protocol says (the first version matched against the median August actual time). Scheduled arrival is approximated as August wheels-on minus arrival delay, which ignores taxi-in (about 5-10 minutes) and is harmless inside the 60-minute window.
5. Bracket widths use bracketed events only; unbracketed events (first seen climbing, or last seen descending) have no bracket, are counted in the detection ratios, and are excluded from timing. Daytime-only medians drive the decision rule.
6. The optional `/flights/departure` comparison was not run.
7. Analysis bug fixed: `rec.tail` resolved to the DataFrame method instead of the tail-number column, which crashed `analyze`.
