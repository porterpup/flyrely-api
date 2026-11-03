-- FlyRely prediction log. Run in the Supabase SQL editor (or any Postgres).
-- Server-side only: RLS is enabled with NO policies, so the anon/authenticated
-- roles cannot read or write. The API connects with the database connection
-- string (DATABASE_URL), which bypasses RLS.

create table if not exists predictions (
    prediction_id        uuid primary key,
    ts_utc               timestamptz not null,
    flight_id            text not null,
    client               text not null,
    origin               text not null,
    dest                 text not null,
    sched_dep_local      timestamp not null,
    sched_dep_utc        timestamptz,
    marketing_airline    text,
    marketing_number     text,
    operating_airline    text,
    operating_number     text,
    horizon              text not null,
    minutes_to_departure integer,
    delay_probability    double precision not null,
    risk_level           text not null,
    model_version        text not null,
    rates_as_of          date,
    features_json        jsonb not null,
    live_inputs_json     jsonb,
    weather_available    boolean not null
);

create index if not exists idx_predictions_flight_id on predictions (flight_id);
create index if not exists idx_predictions_sched_dep_utc on predictions (sched_dep_utc);

alter table predictions enable row level security;
-- Intentionally no policies.

-- Also remove PostgREST role grants (guarded so it works on non-Supabase Postgres).
do $$
begin
    if exists (select 1 from pg_roles where rolname = 'anon') then
        revoke all on predictions from anon;
    end if;
    if exists (select 1 from pg_roles where rolname = 'authenticated') then
        revoke all on predictions from authenticated;
    end if;
end $$;
