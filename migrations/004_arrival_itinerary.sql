-- 004_arrival_itinerary.sql : arrival forecast + itinerary logging (plan: backtest/PLAN_ARRIVAL_AND_CONNECTIONS.md 2.3).
-- Run in the Supabase SQL editor (or any Postgres). Safe to run twice. Server-side only: RLS on, no policies.
-- The API also adds these columns itself the first time it logs an arrival forecast (same self-migrating approach as
-- the CREATE TABLE statements in prediction_log.py), so running this by hand is optional but recommended.

-- Arrival forecast beside the departure probability (written only when the arrival flags are on).
alter table predictions add column if not exists arr_probability    double precision;   -- P(arrive 15+ min late)
alter table predictions add column if not exists cancel_probability double precision;
alter table predictions add column if not exists arr_model_version  text;

-- One row per connection of every /predict/itinerary call.
create table if not exists itinerary_predictions (
    id                   bigint generated always as identity primary key,
    itinerary_id         text not null,
    ts_utc               timestamptz not null,
    connection_index     integer not null,
    leg_flight_ids       text[] not null,       -- [arriving leg, departing leg]
    at                   text not null,         -- hub airport
    kind                 text not null,         -- 'connection' | 'through_flight'
    layover_min          integer,
    mct_min              integer,
    mct_source           text,
    miss_probability     double precision,      -- null when not validated
    validated            boolean not null,
    not_validated_reason text,
    model_version        text not null
);
create index if not exists idx_itinerary_predictions_itinerary on itinerary_predictions (itinerary_id);

alter table itinerary_predictions enable row level security;
-- Intentionally no policies.

do $$
begin
    if exists (select 1 from pg_roles where rolname = 'anon') then
        revoke all on itinerary_predictions from anon;
    end if;
    if exists (select 1 from pg_roles where rolname = 'authenticated') then
        revoke all on itinerary_predictions from authenticated;
    end if;
end $$;

-- Final arrival outcome, so live accuracy (R1) can later score arrival and connections.
-- (`outcomes.diverted` already exists from 002.)
alter table outcomes add column if not exists arr_delay_min double precision;
