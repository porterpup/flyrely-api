# PLAN A6: seeing (nearly) every US airline departure and arrival, free or near-free

Status: **plan and draft pre-registration only.** No polling or paid API calls were made to write this. Research date: 2026-11-05.
Inputs: `REPORT_A5B_OPENSKY_HUBS.md`, `REPORT_A5_OPENSKY.md`, `REPORT_RUNUP.md`, `HANDOFF.md`, the OpenSky credit memory (4,000 credits/day).
A companion guide for Sean is in `../SWIM_SIGNUP.md` (repo parent folder, not committed).

---

## 0. Summary

**Recommendation, in ranked order:**

1. **Sign up for FAA SWIM (SCDS) now.** It is free and self-service, and approval is automatic or within 72 hours. Commercial use is allowed as long as it does not affect the National Airspace System (NAS). Its TFMS flight data covers every IFR flight in the country: flight plans, departure and arrival messages, and tracks. It is the only free source that could give per-flight truth for departure and arrival nationwide. Sean has to make the account himself (about 10 minutes; see `SWIM_SIGNUP.md`). After that, building the consumer is [implementer] work: a Java/JMS client running all the time.
2. **Buy AeroDataBox (from $8/month to pilot, about $75–99/month for ~100 airports)** for "what is scheduled today, and on which tail". Its airport boards return `aircraft.reg` and `aircraft.modeS` (the icao24 hex) plus scheduled, revised and runway times. Commercial use is allowed on every paid plan. Run pilot **P1** (section 4) before A6.
3. **Run A6 as a source-agnostic nationwide ADS-B experiment.**
   - Arm **L** uses adsb.lol. Its licence is ODbL, so commercial use is allowed, and it costs nothing. It can run now.
   - Arm **O** uses OpenSky CONUS `/states/all` at 2-minute cadence: 1,920 credits per 16-hour day, or 2,100 with one 90-minute block at 1-minute cadence. **Arm O is gated on written permission from OpenSky** (see 1.1).
   - Both arms run the A5b detector and the pre-registered looser detector.
   - The denominator comes, in order of preference, from SWIM truth, then AeroDataBox, then BTS August.
4. **Use FlightAware AeroAPI only per user flight at runtime** (about $0.01–0.04 per tracked flight, with `inbound_fa_flight_id` giving the inbound aircraft directly). It is not worth it for nationwide collection: about $400/month for one pass and $1,500–2,300/month with refreshes.
5. **Do not scrape** airport or airline boards (section 5).
6. **Measure value before paying for scale.** A7 (section 7) uses BTS history, which already has tails and actual times, to simulate the "inbound landed / late" feature under the coverage A6 measures. That tells us how much accuracy is at stake.

**Biggest finding: OpenSky's terms forbid what FlyRely is doing without a written licence.** The current OpenSky terms say: *"Any use by a for-profit or commercial entity … requires a written license … regardless of purpose"*. They also say *"Operational REST API use … including integration into a live product … (even if only internal) — requires a previous written agreement."* So A5 and A5b were probably already outside the terms, if FlyRely counts as a for-profit entity, and any live use certainly is. Sean should email `contact@opensky-network.org` before any more OpenSky polling. The rest of this plan does not depend on OpenSky.

---

## 1. Volunteer ADS-B networks and commercial terms

### 1.1 Summary table

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

### 1.2 Merge design (by icao24 and timestamp)

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

## 2. Schedules: "what are we looking for, and when?"

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

### 2.1 AeroDataBox and AviationStack as gap-closers (coordinator request)

| | AeroDataBox | AviationStack |
|---|---|---|
| Free tier | Basic: 400 units/month, 1 req/s, **non-commercial**, FIDS window 12 h | 100 req/month, personal only, **no timetable or schedules** |
| Cheapest commercial tier | Pro $7.50–8 (5k units, 7-day caching) | Basic $49.99 (10k req) |
| ~100-airport production | $75 (Ultra 2) to $99 (Growth) | Professional $149.99 or more. If ~100 results per page hold (unverified), 100 airports × 2 directions × refreshes is far over 50k/month, so Business at $499.99. |
| FIDS covers all major US airports? | Claims 100% schedules and 86% live for the US. The free `/health/services/feeds/{service}/airports` endpoint lists the airports per layer, so check our top 100 at 0 units. | Not documented |
| Tail fill and lead time | `reg` from the live layer "sometimes", usually up to tomorrow. **P1 measures it.** | Unverified; not measurable on the free tier |
| Actual times and source | `revisedTime` and `runwayTime`; sources "aggregated from third-party suppliers" plus ADS-B; `flightPlan` from US ATC filings | `actual`, `actual_runway`; sources undisclosed |
| Verdict | **Pilot it (P1).** | Backup only. Pilot it only if P1 fails, at $49.99 for one month. |

### 2.2 Combined pipeline sketch (schedule + tail + live position → "inbound landed / late")

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

## 3. FAA SWIM: verdict

**Realistic.** I changed my view during the research. The sign-up is self-service. These quotes come from SCDS documents (SAA v1.0, SCDS General Guideline v1.1, 2024-09-11, SWIFT Portal User Guide v4.0, 2023-12; via the [AirsideLabs mirror](https://huggingface.co/datasets/AirsideLabs/SWIFTPortal_help), since the FAA sites returned 403 to automated fetches):

- **Who:** *"SCDS is a publicly accessible cloud-based infrastructure dedicated to providing real-time SWIM data to the public."* The sign-up asks for name, email, password, **company name and phone**. Export-control law applies.
- **Commercial:** *"SWIM data obtained via SCDS is not intended for NAS-Impacting usage. This is not inclusive of operations that solely affect the user's businesses and customers."* *"SCDS data consumers who use SWIM data and provide non-NAS impacting external business service operations will be responsible for notifying their users of planned or unplanned outages."* So a consumer delay-prediction app is allowed. It must not be the *"sole source for any aviation-safety related, law enforcement, or judicial activity."* If we redistribute, *"you may not characterize it as FAA data."*
- **Approval time:** the user guide v4.0 says a subscription is *"created and is approved"* immediately once the product's Service Access Agreement (SAA) is signed in the portal. The Jan 2024 on-boarding sheet says requests are *"reviewed, approved within 72 hours."* So expect minutes to 3 days. SAAs expire after **365 days** and must be re-signed.
- **Obligations:**
  - Honour the monthly **LADD** block list. It affects GA and on-demand aircraft only, but we must be able to block them.
  - Use **JMS 1.1 only** (Solace; the Jumpstart Kit is Java).
  - **Keep compression on.**
  - No duplicate subscriptions: dev and prod must share one, redistributed internally.
  - A volume cap: 200 GB/day or 2 TB/month.
- **CDM data** (some airline-supplied fields) is limited to verified CDM members. We will not get those.

**Products that matter:**

| Product | What we get | Use |
|---|---|---|
| **TFMS** (TFMData "Flight" service; "Flow" service) | Nationwide IFR flight plans, schedule-derived flights, departure and arrival messages, track updates; ground stops, GDPs, reroutes | Per-flight departure and arrival truth (A6 denominator), inbound position; flow programs overlap B5 |
| **SFDPS** (En Route flight data, FIXM) | ERAM flight plans with Item 18, including **`REG/` tail when filed** ([SFDPS manual](https://nsrr.faa.gov/sites/default/files/enroute-ad-pub-v2/SFDPS%20Data%20Consumer%20Reference%20Manual%20v2.1.4.pdf); tail fields added per [2021 briefing](https://www.faa.gov/sites/faa.gov/files/2022-12/2021-November-SWIM-Users-Forum-Briefing.pdf)) | Tail → icao24, so inbound linkage, possibly free |
| **STDDS** (SMES surface events + ASDE-X/ASSC tracks; TDES tower departure events; TAIS terminal tracks) | Surface movement at about 35 ASDE-X + 9 ASSC airports ([FAA ASDE-X](https://www.faa.gov/air_traffic/technology/asde-x)); departure events | Exact pushback, taxi and takeoff at big hubs. Fixes the CLT/DTW surface gap. |
| **TBFM** | Arrival metering (scheduled times of arrival at meter points) | Arrival-delay predictor, later |
| TFDM (TTP) | TOBT/TSAT, EOBT, gate, tail at TFDM sites (rolling deployment) | Later; check availability in the portal |

**Does a flight plan carry the registration?**
- ICAO-format plans carry it in Item 18 `REG/`, which SFDPS exposes.
- Older material says TFMS "terminal" data carries tail numbers ([NASA ATD-2 slide, 2019](https://ntrs.nasa.gov/api/citations/20190025460/downloads/20190025460.pdf)).
- How often US airlines fill `REG/` is **unknown**. It is the first thing to measure after onboarding (build step B-S3).

**Cost of SWIM:** data is free. We need an always-on consumer (Java JMS, or Python via a small Java bridge) and storage. Filtered TFMS flight data plus SFDPS for US airline callsigns should be a few GB/day uncompressed (estimate). Store parsed events only.

Sign-up guide: `/Users/seanbutton/Projects/flyrely/SWIM_SIGNUP.md`.

---

## 4. Pilot P1: AeroDataBox boards (pre-registered; fits inside 5k units)

**Plan:** Pro ($8, 1 month, commercial allowed, 12 h FIDS window). The free Basic plan is non-commercial, so it should not be used by a company even for a test. Pro storage is limited to 7 days: compute the metrics within 7 days, keep only aggregates and derived counts, and delete raw responses on day 7 (build step enforces this).

**Hubs:** the 12 A5b hubs. **Date:** A6 day 1 (D1), a Tue/Wed/Thu. Calls use `direction=Both`, `withCancelled=true`, `withCodeshared=false`, `withCargo=false`, `withPrivate=false`, `withLeg=true`.

| Pull | When (hub local) | Window(s) | Calls | Units |
|---|---|---|---|---|
| L24 | D1−1 at 18:00 | D1 00:00–12:00, 12:00–24:00 | 24 | 48 |
| L06 | D1 06:00 | 06:00–18:00, 18:00–24:00 | 24 | 48 |
| R | D1 09:00, 12:00, 15:00, 18:00 | next 6 h | 48 | 96 |
| POST | D1+1 at 06:00 | D1 00:00–12:00, 12:00–24:00 | 24 | 48 |
| Coverage list | once | `/health/services/feeds/{svc}/airports` | 2–4 | 0 |
| **Total** | | | **≈120** | **≈240** (of 5,000) |

(The coordinator's "about 24 calls" covers one full-day pull. The extra lead-time pulls are needed to measure tail fill by lead time and still use under 5% of the plan.)

**Metrics (all per hub, plus all-hubs):**
1. **Completeness:** operating airline flights on the POST board for D1, against three references:
   - (a) the A5b BTS August same-weekday expected count;
   - (b) ADS-B detected airline departures and arrivals (union detector D1 from A6 on D1): the share of detected callsigns found on the board within ±60 min;
   - (c) SWIM TFMS departures, if available.
2. **Tail fill by lead time:** the share of operating flights with non-null `reg`, by lead-time bucket (>12 h, 6–12, 3–6, 1–3, <1 h, POST), mainline and regional separately. Also the `modeS` fill rate.
3. **Tail → icao24 check:**
   - For flights with `reg`, compare `modeS` (or `n_number_to_icao24(reg)` from `opensky_coverage.py`) with the icao24 that ADS-B observed for that callsign within ±30 min of `runwayTime` (or scheduled time if missing).
   - Report the match rate and the "tail swap" rate (the tail changed between L06 and POST).
4. **Actual times:**
   - Share with `runwayTime` and with `revisedTime`.
   - Median and p90 |`runwayTime` − ADS-B inferred takeoff/landing| for matched events.
   - `quality` mix.
5. **Denominator effect:** recompute A6 D1 detection ratios at the 12 hubs with the board (POST, operating flights, not cancelled) as denominator, against BTS August. Report the change per hub and in the class labels.

**Decision rule (written now):** AeroDataBox is **adopted** as the A6 denominator and the production schedule/tail source if all three hold:
- (i) Board completeness against ADS-B-detected callsigns is ≥0.90 at ≥10 of 12 hubs.
- (ii) Tail fill is ≥0.80 at 1–3 h lead time and ≥0.60 at 3–6 h, for mainline and regional combined.
- (iii) Tail → icao24 agreement with ADS-B is ≥0.90 where checkable (n ≥ 30 per hub-group).

It is **rejected** if (i) is below 0.75 or (ii) is below 0.50 at 1–3 h. Otherwise it is **partial**: use it as the denominator only and keep looking for tails (SWIM `REG/`, AeroAPI per flight).

---

## 5. Scraping airport and airline boards (coordinator request): not recommended

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

---

## 6. A6 pre-registered protocol (draft; to be committed before any A6 polling)

### 6.1 Question
Can nationwide ADS-B polling, from one or more networks, detect airline departures and arrivals at the top-100 US airports reliably, at a cadence that fits the budget? Which detector, which cadence, and how much does each network add?

### 6.2 Gates (must pass before polling)
- **G0 (OpenSky arm O):** written permission or a licence from OpenSky covering evaluation by FlyRely. Without it, arm O does not run and A6 runs on arm L alone. Nothing else changes.
- **G1:** this protocol is committed (hash recorded) before the first poll.
- **G2:** P1 has been run, or is scheduled for D1, so the denominator choice in 6.6 is fixed by rule, not by looking at A6 data.
- **G3 (optional):** SWIM TFMS consumer running on D1. If so, the SWIM-based metrics in 6.7 become primary per-flight truth.

### 6.3 Airports
The top 100 origins by August 2026 BTS departures (list frozen from `data/bts/parquet_v2/2026-08.parquet`; 89.3% of departures). Report results per airport and in tiers: the 12 A5b hubs, ranks 1–30, 31–100.

### 6.4 Collection
- **Days:** five weekdays. **D1 Tue 2026-11-10, D2 Thu 11-12, D3 Tue 11-17, D4 Wed 11-18, D5 Thu 11-19.**
  - Excluded: Veterans Day (Wed 11-11) and Thanksgiving week (11-23 to 11-27).
  - Backups in order: Mon 11-16, Fri 11-20, Tue 12-01. Use a backup only if a day loses more than 10% of rounds.
- **Window:** 11:00–03:00 UTC (6 am–10 pm Eastern / 3 am–7 pm Pacific; covers the main banks). This is 16 h = 480 rounds at 120 s.
- **Arm O (OpenSky):** one `/states/all` CONUS call per round (`lamin 24, lamax 50, lomin −125, lomax −66`; 4 credits). One **1-minute block** each day, 15:00–16:30 UTC, adds 45 extra calls.
  - **Budget:** 1,920 + 180 = **2,100 credits/day**. Hard cap of **2,300/day** for A6 `/states`. Stop if the remaining bucket would fall below **1,500**. Stop on any 429. Never burst-catch-up after a skipped round. Expected remaining at the end of the day is about 1,900 (within the memory's "about 3,000 or less" spend).
  - Other OpenSky jobs must not run on A6 days.
- **Arm L (adsb.lol):** the same round clock.
  - Tiles: `/v2/point/{lat}/{lon}/250` circles chosen by greedy set cover so every top-100 airport's 30 km disc is inside a circle (estimated 12–18 circles; the list is frozen in code before D1).
  - Requests run serially at ≤1 req/s, with exponential backoff on 429 and 5xx, and a User-Agent naming FlyRely and a contact email.
  - Send a courtesy email to the maintainer before D1.
- **Storage:** gzip each raw response, about 1–2 MB per CONUS snapshot, so about 1 GB/day per arm. Keep ledger rows for each call (status, bytes, remaining credits).

### 6.5 Event definitions (fixed now)
Airline callsign and nearest-airport attribution are exactly as in A5b (3-letter BTS carrier ICAO code + 1–4 digits; events attributed to the nearest scheduled-service airport; `MAX_GAP` 600 s; one event per aircraft, type and 30 minutes).

- **Departure detector D0 (A5b):** an airborne observation within 15 km and ≤1,200 m AGL, either preceded within 600 s by a ground observation within 5 km (bracketed) or with vr > +2 m/s and no prior sighting (first-seen-climbing). Takeoff time and bracket as in A5b.
- **Departure detector D1 (looser, from the A5b review):** D0 **OR** a consecutive pair for the same icao24 where the earlier observation is **on ground within 5 km** of the airport and the later is **airborne at any distance or height**, with Δt ≤ 600 s and no ground observation at another airport in between.
  - Inferred takeoff = max(t_ground, t_air − min(600, AGL_air / max(vr, 5 m/s))).
  - Bracket = t_air − t_ground.
- **Arrival detector A0 (A5b):** unchanged.
- **Arrival detector A1 (looser, secondary):** A0 **OR** an airborne observation at any distance followed within 600 s by a ground observation within 5 km.
- Every detector is run on: O alone, L alone, O∪L (merged per §1.2). Each at cadences 1 min (block only), 2, 4 and 6 min, obtained by thinning the same rounds.

### 6.6 Denominator (fixed order, chosen by rule)
1. **SWIM TFMS** departure and arrival messages for airline callsigns at the airport, if G3 holds. This gives per-flight truth.
2. Otherwise the **AeroDataBox POST board**, if P1 adopted it (operating, non-cancelled, non-codeshare airline flights).
3. Otherwise **BTS August same-weekday** (the A5b method).

Whatever is primary, ratios against BTS August are also always reported, for comparability with A5b. The interior-window trimming is the same as A5b (30 min each end).

### 6.7 Metrics
- **Primary (aggregate):** detected ÷ expected, departures and arrivals, per airport per day, then the median across days. Reported per detector, source and cadence. The A5b primary, middle and strict numerators are all reported.
- **Per-flight (when SWIM or AeroDataBox truth exists):**
  - **Recall** = truth flights with a detected event of the same callsign within ±20 min.
  - **Precision** = detected events matching a truth flight.
  - Takeoff and landing time error: median and p90 |detected − truth runway time|.
- **Bracket width:** median and p90 by detector, source and cadence.
- **Source gain:** §1.2 items 1–5.
- **Day-to-day stability:** per-airport standard deviation of the ratio across days, and the class label on each day.
- **Inbound linkage (observational):** the share of D1-detected departures whose icao24 had an A0 or A1 arrival at the same airport earlier the same day (turnaround observed). With AeroDataBox tails, also the share where the board's `reg` matches.
- **Credits:** spent per day, minimum remaining, and any 429s.

### 6.8 Decision rule (written now)
- **Per airport, per detector, source and cadence:**
  - **reliable** = median-over-days departure ratio ≥0.85 **and** arrival ratio ≥0.85 **and** median bracket ≤180 s (bracketed n ≥ 10), **and** reliable on at least 4 of 5 days.
  - **not usable** = either ratio < 0.70.
  - **partial** = anything in between.
  - With per-flight truth, "ratio" is replaced by recall, and precision must also be ≥0.90.
- **Production choice:** the cheapest configuration (fewest credits/requests) whose reliable-airport set is within 5 airports of the best configuration's set.
- **A6 succeeds** if that configuration is reliable at **≥60 of the top 100** airports and has aggregate departure recall/ratio ≥0.90 and arrival ≥0.95.
- **Fallback outcomes:** if no arm meets this but O∪L beats the best single source by ≥0.05 at ≥10 airports, report "merge needed". If nothing reaches 40 airports, report "ADS-B alone insufficient; rely on SWIM/AeroDataBox".
- **Detector:** D1 replaces D0 for production only if, at 4-minute cadence, its departure ratio or recall is ≥0.10 higher than D0's and per-flight precision does not fall below 0.90 (or, without truth, the middle-numerator ratio does not exceed 1.10 at more than 10 airports).

### 6.9 Known limits
- Five weekdays in November; no weekends; no irregular-operations days, unless one happens.
- adsb.lol rate limits are "dynamic", so lost rounds are reported, not imputed.
- BTS August churn (A5b) affects the fallback denominator.
- The AeroDataBox truth is itself partly ADS-B-derived, so it is not independent for actual times. SWIM is independent.

### 6.10 Build steps

| # | Step | Tag |
|---|---|---|
| B0 | Sean: email OpenSky (`contact@opensky-network.org`) asking for written permission or a licence for evaluation and later production (G0). Also email the adsb.lol maintainer as a courtesy. | Sean (main session drafts emails) |
| B1 | Freeze the top-100 airport list and coordinates (from BTS Aug parquet + `data/ref/airports.csv`) into `backtest/a6_airports.json`. | [executor] |
| B2 | Compute the adsb.lol 250 nm tile set by greedy cover; write it to `backtest/a6_tiles.json` with a unit test that every airport disc is covered. | [executor] |
| B3 | Extend `opensky_client.py`: a per-UTC-day cap for an endpoint group (`day_cap`), gzip cache option, and nothing else changed (keep the floor and 429 stop). Tests. | [implementer] |
| B4 | New `backtest/adsblol_client.py`: cached and gzip-compressed, ≤1 req/s, backoff, ledger like OpenSky's, User-Agent with contact. Tests with recorded fixtures (no live calls in tests). | [implementer] |
| B5 | New `backtest/a6_poll.py`: one round clock, arms O and L (O behind a `--arm-o` flag that refuses without an `A6_OPENSKY_PERMISSION=yes` env flag, set only after G0), the 1-min block, pidfile, no burst catch-up. | [implementer] |
| B6 | `backtest/a6_merge.py`: normalise to the unified observation row, cross-source dedupe (§1.2). Tests. | [implementer] |
| B7 | Refactor A5b `detect_events` into a shared module with D0/D1/A0/A1, airport list driven, vectorised per airport. Regression test: D0/A0 reproduce A5b's `a5b_opensky_hubs.json` counts on the cached A5b data. | [implementer] |
| B8 | `backtest/adb_client.py` + `adb_pilot.py` for P1 (pull schedule from §4, 7-day raw deletion, metrics, JSON output). Key in `.env` as `AERODATABOX_KEY`. | [implementer] |
| B9 | `backtest/a6_analyze.py`: all §6.7 metrics, offline, writes `results/a6_nationwide.json`. | [implementer] |
| B10 | Commit the protocol (§6 copied to `REPORT_A6_NATIONWIDE.md`) **before** D1. Run P1 + A6 on schedule. Write the results section. PR → `reviewer`. | main session / [implementer] |
| B-S1 | Sean: SWIM sign-up (`SWIM_SIGNUP.md`). | Sean |
| B-S2 | SCDS consumer: Jumpstart-Kit Java JMS client (compression on) for TFMS Flight + SFDPS, filtered to airline callsigns. Writes parsed events (departure, arrival, flight plan incl. `REG/`) to parquet. LADD block list applied. Runs as one always-on service. | [implementer] |
| B-S3 | One-week SWIM measurement: `REG/` fill rate by airline, departure and arrival message completeness against AeroDataBox and BTS, latency. Pre-register the thresholds in the same style as P1. | [implementer] → `reviewer` |

---

## 7. Which signals actually move accuracy (A7, cheap, recommended before paying for scale)

The run-up backtest already shows the T-3h and T-1h models gain the most (AUC 0.82/0.85) from "where's the plane" information built from BTS tails and actual times. The live question is how much of that survives with **live** coverage gaps.

A7 proposal (pre-register separately):
1. Rebuild the T-3h and T-1h inbound features from BTS history at the as-of time.
2. Degrade them to match measured live coverage: knock out tails at the P1/SWIM fill rate, and drop "inbound landed" for airports A6 labels partial or not usable, with A6's timing error added.
3. Re-score on the existing validation split (no holdout use).

The result is the Brier and AUC loss from imperfect live data. That tells Sean whether $100/month (AeroDataBox) or a paid OpenSky licence is worth it, before buying either for production. [implementer], no external calls.

---

## 8. Open decisions for Sean
1. **OpenSky:** send the licence/permission email? Until there is a reply, A6 runs on adsb.lol only. Also note that A5 and A5b were probably already outside OpenSky's terms for a for-profit entity. I suggest saying so openly in the email.
2. **SWIM sign-up:** about 10 minutes, free (`SWIM_SIGNUP.md`). Which company name and email to use?
3. **AeroDataBox:** approve **$8 for one month** (Pro) for P1, and pre-approve about **$75–99/month** if P1 passes?
4. **adsb.lol ODbL share-alike:** fine to keep any merged flight-event database internal (no obligation beyond attribution)? Attribution line in the app: "Contains data from ADSB.lol, ODbL".
5. **A6 dates:** Nov 10, 12, 17, 18, 19 (no Veterans Day, no Thanksgiving week). OK?
6. **AeroAPI:** keep it for per-flight inbound lookups only (requires the Standard tier, $100/month minimum, for commercial use; the Personal tier is non-commercial). Or drop it if SWIM/AeroDataBox suffice?
7. **Scraping:** confirm we do not scrape airport or airline sites.

## 9. Things I could not verify
- airplanes.live and FAA pages returned 403 to automated fetches. Their wording comes from search snippets and a mirror of FAA documents dated 2023–2024.
- AeroDataBox tail fill and AviationStack fields and fill are unmeasured (P1 measures AeroDataBox).
- The OpenSky licence price is not public.
- The SWIM `REG/` fill rate and current product names in the 2026 portal.
- ADS-B Exchange and FR24 commercial prices.
- The size of the adsb.lol historical archive.
