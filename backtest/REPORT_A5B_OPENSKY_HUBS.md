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
