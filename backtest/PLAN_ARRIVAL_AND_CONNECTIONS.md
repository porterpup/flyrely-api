*Written by `planner` (read-only) on 2026-11-07; saved by the main session.*

## Notes (checked read-only before this plan was written)

1. **The BTS Parquet files have no arrival columns.** `data/bts/parquet_v2` has no ArrDelay, ArrTime, WheelsOn, TaxiIn or DivArrDelay. `PROTOCOL_A7_L0.md` already notes this. The raw Marketing zips (`data/bts/marketing/*.zip`) do have them. So the first task is a new converter (`backtest/prepare_arrival.py`) that writes a separate data set, `data/bts/parquet_arr/`. `prepare.py` stays untouched.
2. **There is no training script for the production model in the repo.** `models/flight_delay_model.joblib` (an 18-feature GradientBoosting model) was added in the initial deployment commit `61f67f5` (2026-02-23). It has no isotonic calibrator: "calib-v2" only refers to the risk-band cut-offs 0.20 and 0.30. `retrain_pipeline.py` and `phase1_retraining.py` did not produce it. The only reproducible "same discipline" in the repo is `backtest.PERIODS` with `backtest.fit_calibrated`, which A7, B5.1 and v3 all use, so the plan uses those.
3. **There is a ready-made contract for the add-flight form.** Branch `feat/flight-lookup` (commit `1276743`, not merged when this was written) adds `GET /flights/lookup`. It returns `matches[]` with `origin`, `dest`, `dep_time`, `arr_time` and `arr_next_day` from `models/route_schedule.csv.gz`. The itinerary contract below is built on those legs.
4. **The `scheduledArrival` trap.** In the app, `flightStore.ts` deletes a field called `scheduledArrival` (it holds invented values from older versions). The new field therefore has to use a different name (`scheduledArrivalLocal`).

---

# Plan: arrival-delay model, missed-connection probability, and the API/app contract

Status: plan and draft pre-registration only. No code was changed, no data was written and no network calls were made. Written 2026-11-07 by `planner` (read-only).

Inputs read:
- API: `main.py`, `backtest/backtest.py`, `backtest/runup.py` (structure only), `backtest/a7_l0.py`, `backtest/holdout.py`, `backtest/paths.py`, `backtest/prepare.py`, `flyrely_features/booking.py`, `copilot/schedule.py`, `prediction_log.py`, `outcomes.py`, the `feat/flight-lookup:flight_lookup.py` branch file, `models/*metadata.json`, `models/route_schedule.csv.gz` (header).
- Backtest docs: `PROTOCOL_ADDENDUM_2026-11.md`, `PROTOCOL_A7_L0.md`, `PLAN_ROADMAP_2026-11.md`, `PLAN_L1_LIVE_TAIL_LINKAGE.md`, `REPORT.md`, `REPORT_V3_VALIDATION.md`, `REPORT_B5_1.md`.
- App: `app/routes/trips/index.tsx`, `src/lib/api.ts`, `src/lib/flightStore.ts`, `src/types/index.ts`.
- Data: the Parquet schema of `2026-03`, and the zip header plus summary statistics of **2025-03**. That month is in the fit window. No evaluation month was looked at.

## 0. Summary for Sean (plain English)

- **What changes for users.** Today FlyRely tells you the chance your flight *leaves* 15+ minutes late. You said what matters is *arriving* late and, on a connecting trip, whether you make your connection. This plan adds both:
  - **Direct flights:** "Chance of arriving 15+ min late" becomes the headline. The departure number stays underneath it.
  - **Connecting trips:** each leg gets its arrival chance. Each connection gets "chance of missing this connection". That number counts the chance your first flight lands late, minus the chance your second flight is also late and so waits for you. Then the final leg's arrival chance.
- **How it works.**
  - We train new models on the same free government (BTS) records, with the same rules as before: train on early 2025, calibrate on late 2025, test on Jan–Aug 2026. The September-onward holdout is never touched.
  - Instead of one yes/no number, the models give a full spread of likely arrival times: early, on time, 15, 30, 60, 120+ minutes late.
  - For a connection we combine the first flight's arrival spread with the second flight's departure spread. We also account for the fact that two flights at the same hub on the same day tend to be late together.
- **How we'll know it's honest.** BTS doesn't record passengers, so we build "pretend connections" from real flights: same airline family, same hub, same day, sensible layover. For each one we check whether it would have been missed, given what actually happened. Before looking at any result, we write down the pass/fail rules. One rule: in each group where we said "10% chance of missing", about 10% must actually have been missed, every month.
- **What we won't claim.**
  - We can't see gates, terminals or the airline's own minimum connection rules. Those are paid data. We use sensible defaults (30–50 min) that you or the user can change.
  - Where we haven't tested a kind of connection (two different airlines, layover over 4 hours, non-US legs), we show no number and say why.
  - The copy says "chance" and "based on past flights", never "you will".
- **Cost and timing.** $0, no scraping, everything fits on the Mac mini (under about 3 GB of memory). About 2 weeks of work. Heavy model training is kept off Nov 17–20 so it doesn't collide with the B5.1 holdout run.
- **What I need from you.** These are listed in section 8:
  - The default connection times.
  - How to show missed-connection risk (percent or "about 1 in 9").
  - Whether to show no number for connections between different airlines.
  - Where to keep the new model file (about 30–60 MB).

---

## 1. Facts that shape the design

| # | Fact | Consequence |
|---|---|---|
| F1 | `parquet_v2` has no arrival columns. The Marketing zips have ArrDelay, ArrTime, WheelsOn, TaxiIn, ArrDel15, DivReachedDest and DivArrDelay. | New module `backtest/prepare_arrival.py` writes a **separate, self-contained** data set `data/bts/parquet_arr/YYYY-MM.parquet`: every `parquet_v2` column plus the arrival columns. It reuses `holdout.is_locked_month` (imported, not edited). The existing loader `backtest.load_months` can read it, because `paths.BTS_PARQUET` honours `FLYRELY_BTS_PARQUET`. No frozen file changes. |
| F2 | Production `/predict` uses the 18 hard-coded schedule/route features (`main.build_features`). `backtest.committed_features` is a vectorised copy, and `check_parity` verifies it matches. | The new models use **exactly these 18 features**. They can be served today with no new rate tables, and feature parity is already tested. |
| F3 | A HistGradientBoosting model retrained with `fit_calibrated` was mis-calibrated on 2026: ECE 0.045, predictions 3–5 points too low (REPORT.md, "2026 was a worse year"). Walk-forward recalibration brought ECE to about 0.015 (REPORT_V3). | The arrival models will probably have the same level shift. The protocol pre-registers a walk-forward recalibration arm and a rule for choosing between the two arms. |
| F4 | In 2025-03 (fit window): mean P(ArrDelay ≥ 15, diverted counted as late) = 0.197, against 0.189 for departure. Arrival quantiles at p10/25/50/75/90/95/99 = −24/−17/−7/8/41/82/216 min. Departure p5/p50 = −11/−3. Departure-arrival correlation 0.968. Cancelled 1.1%, diverted 0.27%. The `Duplicate` flag is "N" on every row. | Most arrivals are early, so the distribution needs thresholds below zero. Severity buckets 15/45/120 sit exactly on threshold points. |
| F5 | The lookup branch returns legs with local `dep_time`, `arr_time` and `arr_next_day`. It also returns 410 for NK (Spirit) and maps HA to AS. | Itinerary legs reuse that shape. The scheduled arrival time comes from the lookup, from the user, or from the same schedule file on the server. |
| F6 | `flightStore.stripLegacyFields` deletes `scheduledArrival`. | The app's new field must be named `scheduledArrivalLocal`. |
| F7 | Run-up models (`runup_*.joblib`) are binary (departure ≥ 15) at T-7d/1d/3h/1h. B5.1's T-3h and T-1h scores use hindsight tails (A7 caveat). | The connection logic consumes a generic "leg forecast" (the ladder in section 2.1), so a run-up ladder can be added later without changing it. |

---

## 2. Design

### 2.1 Item 1: the arrival model ("ladder" of calibrated exceedance probabilities)

**Approaches weighed.**
- (A) **Exceedance ladder.** One calibrated binary model per threshold *t*, giving S(t) = P(delay ≥ t | flight operates). Each is fitted with `backtest.fit_calibrated` (same HGB hyperparameters, isotonic calibration, seed 42). Monotonicity is enforced by sorting across thresholds, and the curve is interpolated piecewise-linearly.
- (B) Quantile regression or a single multiclass HGB over delay bins.

**Pick A.**
- Every point, including P(arrival ≥ 15), is calibrated with exactly the booking-model discipline, so it can be checked with the existing `evaluate`/ECE code.
- The severity buckets and the connection calculation read straight off the curve.
- Option B has no calibration step that matches the existing protocols.

Cost: about 22 small models (roughly 1–3 MB each), measured in T3.

**Thresholds (pre-registered).**
- Arrival ladder (11): −30, −15, 0, 15, 30, 45, 60, 90, 120, 180, 300 minutes.
- Departure ladder (10): −5, 0, 15, 30, 45, 60, 90, 120, 180, 300.
  - Leg 2 needs a departure distribution for the "late departure rescues you" effect, and the production model only gives P(departure ≥ 15).
  - The departure ladder is used **inside the connection calculation only**. `/predict`'s `delay_probability` stays the production model's number.
- Cancellation head: one calibrated binary model on all rows.

**Labels.** Rows are flights that were not cancelled.
- Arrival delay = `ArrDelay`. A diverted flight counts as +∞: it exceeds every threshold. (DOT counts a diversion as not on time.)
- Departure delay = `DepDelay`.

**Tails.** Below the lowest threshold and above the highest, S(x) follows the fit-window empirical shape conditioned on that region (one global table per ladder, stored in the artifact). Values are capped at 600 min.

**Derived outputs.**
- P(arrival ≥ 15).
- Severity given late: minor = (S15 − S45)/S15, moderate = (S45 − S120)/S15, severe = S120/S15. Same buckets as `DelaySeverity`.
- Quantiles q50/q80/q90 by inverting the interpolated curve.
- P(cancel).

**Splits.** Exactly `backtest.PERIODS`:
- fit (model weights) on 2025-01..06;
- isotonic calibration on 2025-07..12;
- evaluation on 2026-01..08 (validation months).

The 18 features use hard-coded tables, so the `rates` windows are not used.

**Recalibration arm (L-R).** On top of arm L, a per-threshold logistic recalibration (intercept and slope on logit(p)).
- In evaluation, month *m* is recalibrated on the 3 most recent months that were published by then (months ≤ m−3). This copies `experiments_v3.recalibrate`, reimplemented in a new module rather than imported.
- In serving, it is refitted on the latest 3 BTS months, which means one executor job per BTS release.

**Memory (6 GB budget).**
- Load one month at a time with `bt.load_months(ym, ym)` and `FLYRELY_BTS_PARQUET=data/bts/parquet_arr`.
- Compute `bt.committed_features` for that month only.
- Write a float32 cache per month to `data/cache/arrival/YYYY-MM.parquet`: 18 features, labels, keys, plus the hub-local scheduled arrival and departure minutes used for proxy pairs.
- Training then concatenates only the cached arrays: about 3.6M rows × 18 × 4 bytes ≈ 260 MB per window. Estimated peak below 3 GB; T3 measures it with `/usr/bin/time -l`.

**Artifact.**
- `models/arrival_ladder_v1.joblib` contains: arrival, departure and cancellation models with their isotonic calibrators; the tail tables; feature names (they must equal `models/feature_names.joblib`); the recalibration coefficients if L-R is chosen; and a `data_through` month.
- Version string: `arr-ladder-v1` (or `arr-ladder-v1r` with recalibration).

### 2.2 Item 2: missed-connection probability

**Definition (the label and the model).**
- For legs 1 → 2 at hub H, the times below are in **H's local time**: leg 1's CRSArrTime and leg 2's CRSDepTime are both local to H, so no time-zone conversion is needed except for leg 1's arrival date.
- Layover L = scheduled departure of leg 2 − scheduled arrival of leg 1.
- M = minimum connect time. It includes walking and the gate door closing (about 10–15 min) before pushback.
- X = leg 1 arrival delay; Y = leg 2 departure delay; s = L − M.
- **Missed connection ⇔ X − Y > s**, given both legs operate. A diverted leg 1 (X = ∞) is a miss.
- Leg 1's arrival date: scheduled departure converted to UTC (zone from `models/airports_ref.csv`) plus CRSElapsedTime, then converted to H's local time.

**Approaches weighed.**
- (A) **Structural:** the two marginal ladders (section 2.1) plus a Gaussian copula with dependence ρ, estimated from BTS proxy pairs, and optionally one isotonic recalibration of the miss probability.
- (B) A direct classifier trained on proxy pairs.

**Pick A.**
- It works for any provider (booking now, run-up or a live ETA later) by swapping the marginals.
- It covers any layover, MCT override or N-leg trip.
- It keeps the "late leg 2 rescues you" mechanism explicit.
- B is run as a **diagnostic only**: if it beats A by a large margin, that tells us A is missing structure.

**Calculation (pure numpy, under 2 ms).**
1. Put X on a 1-minute grid from −60 to 600, using the interpolated ladder.
2. z₁ = Φ⁻¹(F₁(x)).
3. P(Y < x − s | X = x) = Φ((Φ⁻¹(F₂(x − s)) − ρz₁)/√(1−ρ²)).
4. P(miss) = Σₓ p(x) · that probability, plus P(leg 1 diverted).

Integer-minute convention: a miss means Y ≤ X − s − 1. This is unit-tested.

**Dependence variants (pre-registered).**
- R0: independence (ρ = 0).
- R1: one global ρ.
- R2: ρ by hub (the top-30 hubs, shrunk toward the global value by n/(n+5000)) × layover bucket (< 60, 60–120, 120–240 min).
- ρ is the correlation of normal scores of the PIT values (randomised within the interpolation bin, seed 42) of (X, Y) on proxy pairs, using **the provider's own marginals**. This matters: conditioning on run-up features changes the remaining correlation, so ρ is always estimated per provider and horizon.

**Airline holds.** Holds show up as extra correlation in same-carrier pairs with tight layovers. Proxies dilute this, because most proxy pairs carry no real connecting passengers. The report includes a diagnostic: ρ and P(Y ≥ 10 | X − s ∈ [0, 20]) by layover bucket, for same operating carrier versus a different carrier within the same network. It is not used in a decision.

**Minimum connect time (M).** There is no free, complete source: the IATA MCT database is sold through OAG/Cirium. Airline websites are not scraped. Defaults are stored in `models/mct_defaults.csv` with a `source="flyrely_default_estimate"` column:
- 30 min at airports with fewer than 100 daily departures (2025 BTS);
- 35 min at 100–400 daily departures;
- 40 min at the top 30;
- 50 min at a curated "large campus" list (ATL, ORD, DFW, DEN, LAX, JFK, EWR, IAH, SFO, CLT), which Sean confirms.

Legs on different marketing networks get +30 min and `validated=false`. Users can override per connection with `min_connect_minutes`, and the response always echoes M and its source. BTS has no terminal data, so terminals only enter through the override.

**Cancellations.**
- P(trip disrupted) = 1 − (1 − c₁)(1 − c₂)(1 − P(miss)), where c₁ and c₂ are the legs' cancellation probabilities.
- Assuming the two cancellations are independent is an approximation. It is **tested** (section 4, rule C5) because same-day hub storms correlate them.
- Diversion is already inside P(miss), as X = ∞.

**Trips with 3 or more legs.** P(make all) = Πₖ(1 − P(missₖ)) × Πᵢ(1 − cᵢ), using unconditional marginals. The response returns `approximate: true`. Only 2-leg connections are validated in this protocol.

**Through flights.** If leg 1 and leg 2 have the same operating carrier and flight number, it is the same aircraft. The connection gets `kind: "through_flight"` and no P(miss), with copy "same flight number continues". Through flights are excluded from proxy pairs.

**Consuming run-up and L1 later (interface, in new package `flyrely_dist/`).**
```python
@dataclass(frozen=True)
class Ladder:            # thresholds (minutes, ascending), S = P(delay >= t | operates), tail tables
    def sf(self, x) -> np.ndarray: ...
    def quantile(self, q) -> float: ...
@dataclass(frozen=True)
class LegForecast:
    arr: Ladder; dep: Ladder; p_cancel: float
    provider: str          # "booking-ladder-v1" | later "runup-3h-ladder" | "live-eta"
    horizon: str           # "booking" | "7d" | "1d" | "3h" | "1h" | "live"
class LegForecaster(Protocol):
    def forecast(self, legs: list[LegSpec], as_of: datetime | None) -> list[LegForecast]: ...
def miss_probability(f1: LegForecast, f2: LegForecast, layover_min: int, mct_min: int,
                     dep: DependenceModel) -> float: ...
```
- `DependenceModel` is looked up by `(provider, horizon)`. A new provider needs (i) a ladder trained by the same `train_ladders.py` on its feature matrix, and (ii) a ρ table fitted on its proxy-pair marginals, under a new protocol.
- When L1 or SWIM shows leg 1 airborne with an ETA, a `live-eta` provider returns a narrow arrival ladder. The engine needs no change.

### 2.3 Item 3: API and app contract

**`/predict`: backward compatible.**
- No existing field changes meaning.
- The request gains one optional field, `arrival_time` (scheduled local arrival, ISO without offset). It is echoed back and not used by the model. Arrival needs no new inputs because the model uses the same 18 features.
- The response gains one nullable block:
```json
"arrival": {
  "probability_late_15": 0.23,          // P(arrive 15+ min after schedule | flight operates)
  "risk_level": "medium",               // same 0.20 / 0.30 cut-offs, tier ordering checked (rule A3)
  "severity": { ...same shape as DelaySeverity, from the arrival ladder... },
  "quantiles_minutes": {"p50": -6, "p80": 14, "p90": 38},   // vs scheduled arrival
  "exceedance": [{"minutes": -30, "probability": 0.94}, ...],  // the ladder
  "cancel_probability": 0.012,
  "model_version": "arr-ladder-v1"
}
```
- `arrival` is `null` (plus a warning "Arrival forecast unavailable") when the ladder failed to load. `/predict` still returns 200 with the departure prediction, exactly as `delay_severity: null` works today. This is not a fallback, because no number is invented.
- `/health` adds `arrival_model_loaded`. `GET /` lists the new endpoint.
- Arrival is shown only if the gates in section 4 pass. Otherwise the artifact is not deployed and `arrival` stays null.

**`POST /predict/itinerary`.**
- Mounted only when `FLYRELY_FEATURE_ITINERARY=1`, like the copilot. It lives in a new module, `itinerary.py`, with `mount(app)` (the `flight_lookup.mount` pattern).

Request:
```json
{
  "itinerary_id": "trip-123",                       // optional, echoed
  "legs": [                                          // 1..4, in travel order
    {"flight_id": "f1", "flight_number": "UA648", "airline": "UA",
     "origin": "SFO", "destination": "DEN",
     "departure_time": "2026-12-03T07:10:00",        // local, as /predict
     "arrival_time": "2026-12-03T10:42:00"},         // local at destination; optional (resolved below)
    {"flight_number": "UA1442", "origin": "DEN", "destination": "BNA",
     "departure_time": "2026-12-03T11:35:00", "arrival_time": "2026-12-03T15:05:00"}
  ],
  "connections": [{"min_connect_minutes": 50}]       // optional, length = legs-1, per-connection override
}
```
- If `arrival_time` is missing, it is resolved from the typical schedule by flight number and date (`BtsTypicalSchedule.flights_by_number`, from the lookup branch). If it is still unknown, the endpoint returns 422 `{"reason": "schedule_unknown"}`.
- Leg *i*'s destination must equal leg *i+1*'s origin, else 422 `not_connected`.
- A layover ≤ 0 or > 24 h returns 422 `invalid_layover`. More than 4 legs returns 422.

Response:
```json
{
  "itinerary_id": "trip-123",
  "legs": [ <full PredictionResponse per leg, including "arrival"> ],
  "connections": [{
    "at": "DEN", "kind": "connection" | "through_flight",
    "scheduled_layover_minutes": 53,
    "min_connect_minutes": 40, "mct_source": "default_large_hub" | "user",
    "miss_probability": 0.11,            // P(miss | both legs operate); null when not validated
    "risk_level": "low" | "medium" | "high" | null,
    "validated": true,
    "not_validated_reason": null,        // e.g. "different_airlines", "layover_over_4h", "airport_not_tested"
    "warnings": ["Layover is shorter than the usual minimum at DEN"]
  }],
  "trip": {"disruption_probability": 0.15, "approximate": false},  // any miss or cancellation
  "warnings": [],
  "model_version": "itin-v1 (arr-ladder-v1, dep-copula-R?)"
}
```
- Connections are `validated=true` only when all hold:
  - the hub is in the evaluated top-30;
  - both legs are on the same marketing network (HA is treated as AS);
  - M ≤ L ≤ 240 min;
  - it is not a through flight;
  - both airports are in BTS.

  Otherwise `miss_probability` is null and the reason is given.
- No partial answers: if any leg's model fails, the whole call returns 503 with `model_unavailable` or `prediction_failed`. If the ladder artifact is missing, `/predict/itinerary` returns 503 `model_unavailable` while `/predict` keeps working.
- **Logging.** Each leg is written through the existing `prediction_store.safe_write`. New migration `migrations/004_arrival_itinerary.sql` adds:
  - `arr_probability`, `cancel_probability` and `arr_model_version` to predictions;
  - an `itinerary_predictions` table (itinerary_id, leg flight_ids, at, layover, mct, mct_source, miss_probability, validated, model_version, ts_utc);
  - `arr_delay_min` and `diverted` to `outcomes`, so R1 live accuracy can later score arrival and connections.

**App (`flyrely-app`).** Implemented after the add-flight rebuild lands, behind an app flag in the same style as `src/copilot/flags.ts`.
- **Contract with the add-flight agent.** The form outputs a `LegInput`:
  `{flightNumber, airline, date, origin, destination, departureLocal, arrivalLocal, arrivalSource: 'lookup'|'manual'}`.
  - From the lookup: `arrivalLocal` = date + `arr_time` (+1 day if `arr_next_day`).
  - After saving a leg, an "Add connecting flight" button pre-fills date = arrival date and origin = previous destination.
  - Nothing else is required of that agent.
- **Types.**
  - `Flight` gains `scheduledArrivalLocal?`, `arrivalSource?`, `tripId?`, `legIndex?` and `arrival?: ArrivalForecast | null`.
  - New `Trip {id, flightIds[], connections: {mctOverride?: number}[], lastResult?: ItineraryResponse, checkedAt}` stored under `flyrely_trips`.
  - Do **not** use the field name `scheduledArrival` (F6).
- **`src/lib/api.ts`.** Add an `arrival` normaliser (missing → null) and `predictItinerary()`.
- **Trips route.** Group by trip first, then by date for single flights.
- **Flight detail.** Arrival headline plus departure line.

**What the app shows (copy rules: always "chance", whole percents, under 2% shown as "under 2%", no "will" or "guaranteed").**

Direct flight:
- Headline: **"Arriving 15+ min late: 23% chance"**, with a risk colour.
- Second line: "Leaving 15+ min late: 21%".
- Severity: "If it's late, it's usually under 45 min" (most likely bucket).
- Cancellation: "Cancellation chance: about 1%".
- Footer: "Based on how this airline, route and time of day performed in US government flight records. It doesn't yet know today's weather or where your plane is."

Connecting trip: legs stacked, with a connection row between them.
- Connection row:
  - **"Connection in Denver · 53 min"**, with sub-text "Minimum we assume: 40 min (change)".
  - **"Chance of missing it: 11% (about 1 in 9)"**.
  - Info line: "Counts the chance your first flight lands late, and that your next flight leaves late too."
- When a connection isn't validated: "We haven't tested connections like this one (different airlines), so we don't show a number."
- Through flight: "Same flight number continues. Usually the same plane."
- Final leg: its arrival headline.
- Trip line: "Chance something on this trip goes wrong (missed connection or cancellation): 15%".
- Never say the airline will rebook or hold the flight. The existing copilot `DisruptionPanel` stays the "what are my options" path.

---

## 3. Proxy itineraries from BTS (the calibration target)

**Hubs.** The top 30 airports by 2025 departures (the frozen list is written to the protocol at T0).

**Pairs.** Leg 1 (arrives at H) and leg 2 (departs H). All of these must hold:
- same `Marketing_Airline_Network` (AS and HA merged; NK excluded);
- leg 2's FlightDate equals leg 1's arrival date in H's local time;
- M_H ≤ L ≤ 240;
- leg 2's destination ≠ leg 1's origin;
- great-circle distance from leg 1's origin to leg 2's destination ≥ 150 mi;
- detour (d₁ + d₂)/d(O, D) ≤ 1.6;
- not a through flight;
- not leg 2's own same-tail inbound (the same aircraft would be a rotation, not a passenger connection).

**Sampling (seed 42).** For each leg 1, draw at most one leg 2 from each layover bucket [M, M+30), [M+30, 60+M), [60+M, 120), [120, 240], uniformly within the bucket. This oversamples tight connections, where misses happen. Metrics are reported both unweighted and reweighted to an all-pairs layover mix. Expect about 0.5–1M pairs per month. Pairs are written month by month to `data/cache/arrival/pairs_YYYY-MM.parquet`.

**Outcomes per pair.**
- `miss` (both legs operated and X − Y > s, a diversion counting as a miss);
- `c1` (leg 1 cancelled);
- `c2` (leg 2 cancelled);
- `disrupted` = miss or c1 or c2.

**Honest limit.** These are physically possible, schedule-sold-shaped connections, not booked passengers. Real minimum connect times, gate distances and airline-protected connections are not observed. The proxies calibrate "the inbound arrives too late for the outbound". They cannot check passenger behaviour. Live itineraries scored later through R1 (SWIM arrivals, then BTS) are the real-world check.

---

## 4. Pre-registered protocol (commit verbatim as `backtest/PROTOCOL_ARRIVAL_CONNECTIONS.md` before any evaluation is run)

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

## 5. Tasks (`[executor]` = mechanical run or commit; `[implementer]` = code)

**T0 [executor]: commit the protocol.** 11-08, after Sean reads section 0.
- Commit section 4 as `backtest/PROTOCOL_ARRIVAL_CONNECTIONS.md`, with the frozen top-30 hub list and the large-campus MCT list filled in from 2025 data only.
- Accept: commit hash recorded at the top of the file; `git diff master --stat -- flyrely_features backtest/runup.py backtest/backtest.py backtest/experiments_b5.py backtest/holdout.py backtest/fetch_faa_adv.py` is empty.

**T1 [implementer]: `backtest/prepare_arrival.py`.** About 0.5 day.
- Same zip reader as `prepare.py` (a copied helper, not an edit), the `MARKETING_COLUMNS` plus arrival columns, written to `data/bts/parquet_arr/`. It refuses locked months and has no `--holdout` flag.
- Tests, in `tests/test_prepare_arrival.py`:
  - a fixture zip converts;
  - a 2026-09 zip is refused even with the unlock variable set (the script exits if the variable is present);
  - shared columns equal `parquet_v2` for a fixture.
- Accept (executor run): 20 months built (2025-01..2026-08); per month, row count and shared columns equal `parquet_v2`; extra disk under 200 MB.

**T2 [implementer]: `backtest/arrival_data.py`.** About 1 day.
- Month-by-month loader via `bt.load_months(ym, ym)` with `FLYRELY_BTS_PARQUET` set. Calls `bt.committed_features`, the labels, and the hub-local scheduled-minute columns (zoneinfo from `airports_ref.csv`). Writes a float32 cache per month.
- Also: `pairs.py` to build proxy pairs as in section 3.
- Tests, in `tests/test_arrival_data.py` and `tests/test_proxy_pairs.py`:
  - on a synthetic frame: same network only; destination ≠ origin; layover window; detour rule; through flights and same-tail rotations excluded; cross-midnight arrival date and DST day; label boundary (X − Y == s is not a miss);
  - diverted means a miss;
  - no date ≥ 2026-09 (assert);
  - `check_parity`-style test: 500 cached rows equal `main.build_features`.

**T3 [implementer], then [executor] run: `flyrely_dist/ladder.py` and `backtest/train_ladders.py`.** About 1 day of code plus about 2 h of run.
- Ladder dataclass with sf/quantile/severity/tails, rearrangement, and both arms (L and L-R).
- Tests, in `tests/test_ladder.py`:
  - sf is monotone after rearrangement;
  - interpolation is exact at thresholds;
  - severity sums to 1 and matches the threshold points;
  - quantile inverts sf;
  - tails stay in [0, 1];
  - feature names equal `models/feature_names.joblib`.
- Accept: peak RSS ≤ 6 GB (`/usr/bin/time -l`, logged); artifact loads under scikit-learn 1.7.2; artifact size logged (target ≤ 80 MB, otherwise see decision D4); predicting one row for all 22 models takes ≤ 40 ms on the Mac mini.
- Run on 11-10 to 11-12, never during 11-17 to 11-20.

**T4 [implementer]: `flyrely_dist/connections.py`, `flyrely_dist/mct.py`, `flyrely_dist/providers.py`.** About 1.5 days.
- Copula engine, R0–R2 fitting from PIT values, N-leg product, MCT table and override, `LegForecaster` with `BookingLadderProvider`.
- Tests, in `tests/test_connections.py`:
  - point-mass marginals give a 0 or 1 answer;
  - ρ = 0 matches brute-force numerical convolution within 1e-3;
  - P(miss) decreases with layover and increases with MCT;
  - shifting Y later lowers P(miss) (the rescue effect);
  - ρ close to 1 with identical marginals gives P(miss) → P(0 > s);
  - a diverted mass is counted as a miss;
  - the 3-leg case returns `approximate=True`.

**T5 [implementer], then [executor] single run: `backtest/eval_arrival_connections.py`.** About 1 day of code plus about 1 h of run, on 11-15 to 11-16.
- Implements section 4 exactly and writes the JSON. A `--quick` mode runs on 2025 months only.
- Accept: JSON contains every metric and decision in section 4; nothing dated ≥ 2026-09 is loaded (assert plus test); the report is drafted.
- Then a `reviewer` pass on the report.

**T6 [implementer]: API.** About 1.5 days. Can run in parallel from 11-13; touches no frozen files.
- `main.py`: optional `arrival_time`; `arrival` block; `/health` field.
- `itinerary.py` behind the flag; migration 004; logging.
- Tests, in `tests/test_api_arrival.py` and `tests/test_api_itinerary.py`:
  - the old `/predict` keys and values are unchanged for a fixed request (snapshot);
  - `arrival` is null when the ladder is missing and the call still returns 200;
  - `/predict/itinerary` returns 503 when the ladder is missing;
  - 422 for `not_connected`, `schedule_unknown`, more than 4 legs, `invalid_layover`;
  - through flight; `validated=false` for different networks and layovers over 240 min;
  - the MCT override is echoed back;
  - CORS is unchanged;
  - the route is absent when the flag is off;
  - codeshare: the model uses the operating carrier.
- Accept: the itinerary endpoint is deployed only if section 4 passes; arrival is exposed only if A1–A3 pass.

**T7 [implementer]: app.** About 2 days. After the add-flight rebuild merges, from about 11-21.
- Types, `api.ts` normaliser and `predictItinerary`, `flyrely_trips` store, trips grouping, the arrival headline on flight detail, the connection row, all behind the flag.
- Tests (vitest):
  - normaliser: missing `arrival` → null;
  - `scheduledArrivalLocal` survives `stripLegacyFields`;
  - trip grouping;
  - copy snapshot (no "will", percent formatting, the under-2% rule).

**T8 [executor]: operations.**
- If L-R is chosen, a monthly recalibration refresh when BTS publishes.
- Docs: README endpoints, `HANDOFF.md`.
- Add the arrival label to the R1 outcomes loader (`arr_delay_min`).

## 6. Timeline (today is Sat 2026-11-07)

| Date | Work |
|---|---|
| 11-08 | T0 protocol commit |
| 11-09 – 11-10 | T1 prepare_arrival; T2 starts |
| 11-10 – 11-12 | T2 caches and pairs; T3 training run (in the background, overnight) |
| 11-12 – 11-14 | T4 engine; T6 API code starts (behind the flag) |
| 11-15 – 11-16 | T5 single evaluation run, report, reviewer |
| 11-17 – 11-20 | **No heavy compute (B5.1 holdout H1).** T6 tests and review only |
| 11-21 – 11-24 | T6 merge if the rules pass; T7 app (after add-flight merges) |
| 11-25 → | Flag on for Sean only; live itinerary logging feeds R1 |

## 7. Risks

- **Level shift (F3).** It is handled by the pre-registered L-R arm. If A1 still fails, arrival is not shown.
- **Proxy pairs are not passengers.** This is stated in the report and the copy. R1 provides the live check.
- **Artifact size and latency.** Measured in T3. If the models are too large, reduce the thresholds under a new protocol amendment, never silently.
- **The departure ladder at 15 min disagrees with the production model.** This is reported. The headline shown is always the production number, and the ladder is used only inside the connection maths.
- **Hawaiian/Alaska merge and the end of Spirit.** HA is treated as AS in the network rule. NK is excluded from pairs and reported separately for legs.

## 8. Decisions for Sean

- **D1.** MCT defaults (30/35/40/50 min, plus the large-campus list), with a user override.
- **D2.** How to show miss risk: percent plus "about 1 in N" (recommended), and the bands < 5% / 5–15% / ≥ 15%.
- **D3.** Hide the number for connections between different airlines (recommended) or show it marked "not tested".
- **D4.** Where to keep the ladder file (30–80 MB): in git, or gitignored with a sha256 manifest copied to the Mac mini (recommended).
- **D5.** Whether arrival replaces departure as the headline for direct flights once A1–A3 pass (recommended yes, per your note).

## 9. Not verified

- Training time and memory on the Mac mini, and the artifact size (measured in T3).
- Whether the lookup branch's `flights_by_number` stays as it is now (the branch is unmerged).
- The actual proxy miss base rate. It hasn't been computed, deliberately, before pre-registration.

## 10. Acceptance commands

```
cd /Users/seanbutton/Projects/flyrely/flyrely-api
venv/bin/python -m pytest tests/test_prepare_arrival.py tests/test_arrival_data.py tests/test_proxy_pairs.py tests/test_ladder.py tests/test_connections.py tests/test_api_arrival.py tests/test_api_itinerary.py tests/test_holdout_guard.py tests/test_feature_parity.py tests/test_api.py -q
venv/bin/python -m pytest -q          # full suite stays green
git diff master --stat -- flyrely_features backtest/runup.py backtest/backtest.py backtest/experiments_b5.py backtest/holdout.py backtest/fetch_faa_adv.py   # must be empty
env | grep FLYRELY_HOLDOUT_UNLOCK      # must print nothing during every run
cd ../flyrely-app && npx vitest run
```

### Critical Files for Implementation
- /Users/seanbutton/Projects/flyrely/flyrely-api/backtest/backtest.py (imported only: `load_months`, `committed_features`, `fit_calibrated`, `PERIODS`, `ece_and_deciles`)
- /Users/seanbutton/Projects/flyrely/flyrely-api/backtest/prepare.py (the pattern for the new `prepare_arrival.py`; do not edit)
- /Users/seanbutton/Projects/flyrely/flyrely-api/main.py (the `/predict` arrival block, `/health`, mounting `itinerary.py`)
- /Users/seanbutton/Projects/flyrely/flyrely-api/copilot/schedule.py and the `feat/flight-lookup:flight_lookup.py` branch file (leg shape, filling in a missing scheduled arrival)
- /Users/seanbutton/Projects/flyrely/flyrely-app/src/lib/api.ts and /Users/seanbutton/Projects/flyrely/flyrely-app/src/lib/flightStore.ts (the app contract; the `scheduledArrival` field-name trap)
