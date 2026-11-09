-- 005_user_flights_trips.sql : FlyRely cloud-saved flights and trips.
-- Paste into Supabase > SQL Editor > New query > Run. Safe to run twice.
-- Unlike 001-003, these tables ARE read/written by the app (role `authenticated`) under RLS: each user sees only
-- their own rows. Server jobs (later alerts) read via DATABASE_URL or a dedicated read-only role (see bottom).

-- ---------- tables ----------
create table if not exists public.user_trips (
    user_id           uuid not null default auth.uid() references auth.users(id) on delete cascade,
    id                uuid not null,                                -- generated on the device (crypto.randomUUID)
    name              text check (name is null or char_length(name) between 1 and 80),
    client_updated_at timestamptz not null default now(),           -- device edit time: newest edit wins
    created_at        timestamptz not null default now(),
    updated_at        timestamptz not null default now(),           -- server receive time (trigger)
    deleted_at        timestamptz,
    primary key (user_id, id)
);

create table if not exists public.user_flights (
    user_id                   uuid not null default auth.uid() references auth.users(id) on delete cascade,
    id                        text not null check (char_length(id) between 1 and 64),  -- app Flight.id
    airline_code              text not null check (airline_code ~ '^[A-Z0-9]{2,3}$'),
    flight_number             text not null check (char_length(flight_number) between 2 and 10
                                                   and flight_number = upper(flight_number)),  -- e.g. UA648
    origin                    text not null check (origin ~ '^[A-Z0-9]{3,4}$'),
    destination               text not null check (destination ~ '^[A-Z0-9]{3,4}$'),
    scheduled_departure_local timestamp not null,                   -- local wall clock at origin
    scheduled_arrival_local   timestamp,                            -- local wall clock at destination
    trip_id                   uuid,
    leg_index                 smallint check (leg_index between 0 and 3),          -- 4 legs max (plan 2.3)
    connect_mct_override      smallint check (connect_mct_override between 10 and 240), -- min connect before THIS leg
    data                      jsonb not null default '{}'::jsonb
                              check (jsonb_typeof(data) = 'object' and pg_column_size(data) <= 32768),
    client_updated_at         timestamptz not null default now(),
    created_at                timestamptz not null default now(),
    updated_at                timestamptz not null default now(),
    deleted_at                timestamptz,
    primary key (user_id, id),
    foreign key (user_id, trip_id) references public.user_trips (user_id, id),
    check ((trip_id is null) = (leg_index is null)),
    check (connect_mct_override is null or (trip_id is not null and leg_index > 0)),
    check (origin <> destination)
);

-- ---------- indexes ----------
-- No duplicates: one live row per user + flight + origin + departure (catches the same flight added on two devices).
create unique index if not exists ux_user_flights_natural
    on public.user_flights (user_id, flight_number, origin, scheduled_departure_local) where deleted_at is null;
-- One flight per leg position in a trip.
create unique index if not exists ux_user_flights_trip_leg
    on public.user_flights (user_id, trip_id, leg_index) where trip_id is not null and deleted_at is null;
-- Server-side alert scans ("flights departing in the next N hours").
create index if not exists ix_user_flights_upcoming
    on public.user_flights (scheduled_departure_local) where deleted_at is null;
create index if not exists ix_user_flights_trip on public.user_flights (user_id, trip_id) where trip_id is not null;

-- ---------- sync guard trigger (conflict rule lives in the database) ----------
create or replace function public.flyrely_sync_guard() returns trigger
language plpgsql set search_path = '' as $$
begin
    -- A device with a wrong clock can't lock out future edits.
    if new.client_updated_at > now() + interval '5 minutes' then
        new.client_updated_at := now();
    end if;
    if tg_op = 'UPDATE' then
        if old.deleted_at is not null then
            return null;                               -- deleted rows stay deleted (skip the update)
        end if;
        if new.deleted_at is null and new.client_updated_at < old.client_updated_at then
            return null;                               -- an older edit never overwrites a newer one
        end if;                                        -- (a delete always wins)
        new.user_id := old.user_id;
        new.created_at := old.created_at;
    else
        new.created_at := now();
    end if;
    new.updated_at := now();
    return new;
end $$;

drop trigger if exists trg_user_flights_guard on public.user_flights;
create trigger trg_user_flights_guard before insert or update on public.user_flights
    for each row execute function public.flyrely_sync_guard();
drop trigger if exists trg_user_trips_guard on public.user_trips;
create trigger trg_user_trips_guard before insert or update on public.user_trips
    for each row execute function public.flyrely_sync_guard();

-- ---------- row level security ----------
alter table public.user_flights enable row level security;
alter table public.user_trips   enable row level security;

drop policy if exists "own flights select" on public.user_flights;
drop policy if exists "own flights insert" on public.user_flights;
drop policy if exists "own flights update" on public.user_flights;
create policy "own flights select" on public.user_flights for select to authenticated
    using ((select auth.uid()) = user_id);
create policy "own flights insert" on public.user_flights for insert to authenticated
    with check ((select auth.uid()) = user_id);
create policy "own flights update" on public.user_flights for update to authenticated
    using ((select auth.uid()) = user_id) with check ((select auth.uid()) = user_id);

drop policy if exists "own trips select" on public.user_trips;
drop policy if exists "own trips insert" on public.user_trips;
drop policy if exists "own trips update" on public.user_trips;
create policy "own trips select" on public.user_trips for select to authenticated
    using ((select auth.uid()) = user_id);
create policy "own trips insert" on public.user_trips for insert to authenticated
    with check ((select auth.uid()) = user_id);
create policy "own trips update" on public.user_trips for update to authenticated
    using ((select auth.uid()) = user_id) with check ((select auth.uid()) = user_id);
-- No DELETE policy: the app soft-deletes. Rows go away only when the auth user is deleted (cascade).

revoke all on public.user_flights, public.user_trips from anon;
revoke all on public.user_flights, public.user_trips from authenticated;
grant select, insert, update on public.user_flights, public.user_trips to authenticated;

-- ---------- RPC: one-time import of a device's old localStorage flights (idempotent, no duplicates) ----------
create or replace function public.import_flights(p_rows jsonb) returns integer
language plpgsql security invoker set search_path = '' as $$
declare n integer;
begin
    if auth.uid() is null then raise exception 'not signed in'; end if;
    if jsonb_typeof(p_rows) <> 'array' or jsonb_array_length(p_rows) > 500 then
        raise exception 'expected an array of at most 500 flights';
    end if;
    insert into public.user_flights (user_id, id, airline_code, flight_number, origin, destination,
                                     scheduled_departure_local, scheduled_arrival_local, data, client_updated_at)
    select auth.uid(), r.id, r.airline_code, r.flight_number, r.origin, r.destination,
           r.scheduled_departure_local, r.scheduled_arrival_local, coalesce(r.data, '{}'::jsonb),
           coalesce(r.client_updated_at, now())
    from jsonb_to_recordset(p_rows) as r(id text, airline_code text, flight_number text, origin text,
         destination text, scheduled_departure_local timestamp, scheduled_arrival_local timestamp,
         data jsonb, client_updated_at timestamptz)
    on conflict do nothing;          -- same id OR same flight already in the account: skip it
    get diagnostics n = row_count;
    return n;
end $$;

-- ---------- RPC: set a trip's legs in order, all at once (also dissolves a trip when given < 2 legs) ----------
create or replace function public.set_trip_legs(p_trip_id uuid, p_flight_ids text[], p_client_updated_at timestamptz)
returns void
language plpgsql security invoker set search_path = '' as $$
declare uid uuid := auth.uid(); i int; n int := coalesce(array_length(p_flight_ids, 1), 0);
begin
    if uid is null then raise exception 'not signed in'; end if;
    if n > 4 then raise exception 'a trip has at most 4 flights'; end if;
    if not exists (select 1 from public.user_trips where user_id = uid and id = p_trip_id and deleted_at is null) then
        raise exception 'trip not found';
    end if;
    if n >= 2 and (select count(*) from public.user_flights
                   where user_id = uid and id = any(p_flight_ids) and deleted_at is null) <> n then
        raise exception 'flight not found';
    end if;
    -- Detach current legs of this trip, and the requested flights from any other trip.
    update public.user_flights
       set trip_id = null, leg_index = null, connect_mct_override = null,
           client_updated_at = greatest(client_updated_at, p_client_updated_at)
     where user_id = uid and deleted_at is null and (trip_id = p_trip_id or id = any(p_flight_ids));
    if n < 2 then
        update public.user_trips set deleted_at = now(),
               client_updated_at = greatest(client_updated_at, p_client_updated_at)
         where user_id = uid and id = p_trip_id;
        return;
    end if;
    for i in 1..n loop
        update public.user_flights
           set trip_id = p_trip_id, leg_index = i - 1,
               client_updated_at = greatest(client_updated_at, p_client_updated_at)
         where user_id = uid and id = p_flight_ids[i];
    end loop;
end $$;

revoke all on function public.import_flights(jsonb) from public, anon;
revoke all on function public.set_trip_legs(uuid, text[], timestamptz) from public, anon;
grant execute on function public.import_flights(jsonb) to authenticated;
grant execute on function public.set_trip_legs(uuid, text[], timestamptz) to authenticated;

-- ---------- LATER (alerts plan), do NOT run now ----------
-- create role flyrely_alerts login password '<Sean picks>';     -- read-only, for the Mac mini alerts job
-- grant usage on schema public to flyrely_alerts;
-- grant select on public.user_flights, public.user_trips to flyrely_alerts;
-- create policy "alerts read" on public.user_flights for select to flyrely_alerts using (true);
-- create policy "alerts read" on public.user_trips   for select to flyrely_alerts using (true);
