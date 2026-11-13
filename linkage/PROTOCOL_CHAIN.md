# Protocol CHAIN: how much of the departure board can free data link to its aircraft by walking forward from a known aircraft?

Dated 2026-11-13. Pre-registered: this file is committed BEFORE any eval day exists (eval days start 2026-11-16).
Offline, on the live linkage store, $0. Sean approved this bar on 2026-11-13.

## Question

At T-3h and T-1h before scheduled departure, what share of departures can be correctly linked to their operating aircraft
or inbound leg, using only facts known by then (as-of `first_recv` <= cutoff)?

## Methods, in priority order

- **M1**: the leg's own SFDPS registration or icao24.
- **M4, anchor walk**: the aircraft was identified at the cutoff on an earlier leg of its day. Walk forward, and at each step take, in order:
  1. a departure already filed with the same aircraft;
  2. else M2 on the allow-list;
  3. else the pairing prior (same weekday where available, else the previous day).

  Apply an aircraft-type-family filter, and exclude departures already filed with another aircraft. Limit the walk to 6 steps
  within 18 h. Pattern consistency: a prior step counts as high confidence only if the previous step matched the prior line.
- **M2**: same flight number continuing. Allow-list (BTS calibration, T4): WN, DL, AA, OO, MQ, 9E, OH, PT, YX, AS, QX.
  UA, G4, G7, C5, YV are excluded.
- **M3**: prior alone.
- **none.**

## Truth

SMES surface-event icao24, else the last SFDPS value for the leg. Report the share of legs with no truth; those are excluded.

## Days

- Dev days (tuning allowed): 2026-11-08 to 2026-11-12.
- Eval days (frozen): 2026-11-16 to 2026-11-22 UTC.
- A day is excluded, and replaced by the next complete day, if SFDPS or TFMS has a gap over 15 minutes between 11:00 and 03:00 UTC (check archive or status timestamps).
- Known outage on 2026-11-13, 09:49 to 21:50 UTC.

## Population

Scheduled domestic chain legs (`Store.chain_legs()`) departing the top-100 airports.

Definitions the code needs (taken from the L1 plan, section 3.2, not new rules): the top-100 list is the frozen BTS-August-2026 list of
busiest origins; a day's legs are those scheduled 13:00 UTC of that day to 03:00 UTC of the next, so that T-3h falls inside the
11:00 to 03:00 collection window.

## Outputs, per carrier group x horizon

- coverage, precision, and effective linkage (correct share);
- high-confidence share and its precision;
- the anchor upper bound;
- the Brier gain interpolated on the A7 curve, given raw and wrong-link-discounted (effective = correct - wrong).

The A7 curve, at T-3h / T-1h: +0.009 / +0.012 at 29% linkage, +0.015 / +0.022 at 50%, +0.023 / +0.033 at 75%, +0.031 / +0.044 at 100%.

## Decision rule

Pooled, applied as written, with no re-tuning on eval days:

- **GO**: effective linkage >= 0.50 at T-3h AND >= 0.70 at T-1h, AND precision of linked legs >= 0.80 at T-3h AND >= 0.90 at T-1h.
- **T-1h-only GO**: only the T-1h conditions hold.
- **STOP**: neither holds. Keep M1 + M2 only and wait for a paid source.
