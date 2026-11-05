-- FlyRely final flight outcomes (BTS on-time records), joined to `predictions` for live accuracy scoring.
-- Run in the Supabase SQL editor (or any Postgres). The app/scorer also creates this table on first use.
-- Server-side only: RLS is enabled with NO policies.
--
-- One row per operating carrier + flight number + origin + local scheduled departure date + source.
-- label_d15 is the training label (departure delay >= 15 min); NULL when the flight was cancelled.

create table if not exists outcomes (
    carrier           text not null,   -- operating carrier
    flight_number     text not null,   -- operating flight number, digits only, no leading zeros
    origin            text not null,
    sched_dep_date    date not null,   -- local scheduled departure date
    source            text not null,   -- 'bts' (final); later 'swim', 'aeroapi' (provisional)
    marketing_airline text,
    marketing_number  text,
    dest              text,
    sched_dep_local   timestamp,
    dep_delay_min     double precision,
    cancelled         boolean not null,
    diverted          boolean not null,
    label_d15         smallint,
    label_quality     text not null,   -- 'exact' (gate OUT) or 'estimated' (OFF minus taxi-out)
    fetched_utc       timestamptz not null,
    primary key (carrier, flight_number, origin, sched_dep_date, source)
);

create index if not exists idx_outcomes_marketing on outcomes (marketing_airline, marketing_number, origin, sched_dep_date);

alter table outcomes enable row level security;
-- Intentionally no policies.

do $$
begin
    if exists (select 1 from pg_roles where rolname = 'anon') then
        revoke all on outcomes from anon;
    end if;
    if exists (select 1 from pg_roles where rolname = 'authenticated') then
        revoke all on outcomes from authenticated;
    end if;
end $$;
