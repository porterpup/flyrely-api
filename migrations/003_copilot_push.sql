-- FlyRely disruption copilot: web push watches (docs/PLAN_DISRUPTION_COPILOT.md §4.4, §8).
-- Run in the Supabase SQL editor (or any Postgres). The API/watcher also creates this table on first use.
-- Server-side only: RLS is enabled with NO policies. Holds no account id, email, IP or home airport.
-- Rows are deleted at expires_at (scheduled departure + 36h, see copilot/push_store.py), on a 404/410 from the
-- push service, or when the user turns alerts off.

create table if not exists copilot_push_watches (
    watch_id          text primary key,
    endpoint          text not null,            -- push service URL; a capability, never log it
    p256dh            text not null,
    auth              text not null,
    client_flight_id  text,
    airline           text not null,
    flight_number     text not null,
    origin            text not null,
    destination       text not null,
    departure_time    timestamp not null,       -- local wall clock, as sent by the app
    expires_at        timestamptz not null,
    risk_level        text,                     -- last score seen by the watcher
    delay_probability double precision,
    created_at        timestamptz not null,
    heads_up_sent_at  timestamptz,              -- set atomically before sending: exactly-once
    cancelled_sent_at timestamptz,
    unique (endpoint, airline, flight_number, origin, departure_time)
);

create index if not exists idx_copilot_push_expires on copilot_push_watches (expires_at);
create index if not exists idx_copilot_push_endpoint on copilot_push_watches (endpoint);

alter table copilot_push_watches enable row level security;
-- Intentionally no policies.

do $$
begin
    if exists (select 1 from pg_roles where rolname = 'anon') then
        revoke all on copilot_push_watches from anon;
    end if;
    if exists (select 1 from pg_roles where rolname = 'authenticated') then
        revoke all on copilot_push_watches from authenticated;
    end if;
end $$;
