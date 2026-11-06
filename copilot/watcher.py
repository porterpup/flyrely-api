"""Push watcher (plan §4.7): one-time T-24h "high delay risk" heads-up, and a "cancelled" push once a live status
source exists (a stub until roadmap R3, so today `cancelled` never fires in production).

Each cycle:
  1. delete watches past `expires_at`;
  2. for each watch departing within 24 h that has not had its heads-up: score it; if `high`, send once;
  3. if a status source is configured: on a new `cancelled`, send once;
  4. 404/410 from the push service deletes every watch for that endpoint.

Exactly once: a send is claimed with an atomic UPDATE (`push_store.claim`) before it happens, so two processes (the
in-process task and the CLI, or two replicas) cannot both send. A transient failure releases the claim; a crash
between claim and send loses that one push rather than duplicating it. Sends are rate limited (`send_delay_s`,
`max_sends`). Nothing logs endpoints, keys or payloads: counts only.

Runs in-process (started by `copilot.mount` only when FLYRELY_FEATURE_DISRUPTION_COPILOT=1, FLYRELY_COPILOT_PUSH=1
and FLYRELY_COPILOT_WATCHER=1, never under pytest) or as its own process: `python -m copilot.watcher`.
"""
import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from . import config, push, push_store, status as status_mod
from .schemas import FlightRef

logger = logging.getLogger("flyrely.copilot.watcher")

HEADS_UP_WINDOW = timedelta(hours=24)
FAILURES_BEFORE_ALERT = 3


def _departure_utc(watch: push_store.Watch) -> datetime:
    """Local departure -> UTC via the origin's zone; unknown airport falls back to reading it as UTC."""
    from .options import _airports   # lazy: reads models/airports_ref.csv
    ap = _airports().get(watch.origin.upper())
    dep = watch.departure_time
    return dep.replace(tzinfo=ZoneInfo(ap.tz)).astimezone(timezone.utc) if ap else dep.replace(tzinfo=timezone.utc)


class Watcher:
    def __init__(self, store: push_store.PushStore, score: Optional[Callable], sender: push.Sender,
                 status_source=None, clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                 sleep: Callable[[float], None] = time.sleep, alert: Optional[Callable[[str, str], object]] = None,
                 send_delay_s: float = 0.2, max_sends: int = 200):
        self.store, self.score, self.sender = store, score, sender
        self.status_source = status_source
        self.clock, self.sleep, self.alert = clock, sleep, alert
        self.send_delay_s, self.max_sends = send_delay_s, max_sends
        self._failures = 0
        self._alerted = False

    # ------------------------------------------------------------------ one cycle
    def run_once(self) -> dict:
        now = self.clock()
        stats = {"expired": 0, "checked": 0, "sent": 0, "gone": 0, "errors": 0, "attempted": 0}
        stats["expired"] = self.store.delete_expired(now)
        use_status = self.status_source is not None and getattr(self.status_source, "source", "none") != "none"
        for watch in self.store.active(now):
            if stats["sent"] >= self.max_sends:
                logger.info("watcher: send cap reached, remaining watches wait for the next cycle")
                break
            stats["checked"] += 1
            try:
                dep = _departure_utc(watch)
                if watch.heads_up_sent_at is None and timedelta(0) < dep - now <= HEADS_UP_WINDOW:
                    prob = self._risk(watch)
                    if prob is not None and self._send(watch, push_store.KIND_HEADS_UP, prob, now, stats) == "gone":
                        continue
                if use_status and watch.cancelled_sent_at is None and dep > now - timedelta(hours=6):
                    if self._cancelled(watch):
                        self._send(watch, push_store.KIND_CANCELLED, None, now, stats)
            except Exception as e:   # one bad row must not stop the cycle
                stats["errors"] += 1
                logger.warning("watcher: watch %s failed: %s", watch.watch_id, type(e).__name__)
        logger.info("watcher cycle: %s", stats)
        return stats

    def _risk(self, watch: push_store.Watch) -> Optional[float]:
        """Returns the delay probability if the flight scores `high`, else None."""
        if self.score is None:
            return None
        prob, level = self.score(watch.origin, watch.destination, watch.departure_time, watch.airline)
        self.store.record_risk(watch.watch_id, level, float(prob))
        return float(prob) if level == "high" else None

    def _cancelled(self, watch: push_store.Watch) -> bool:
        ref = FlightRef(airline=watch.airline, flight_number=watch.flight_number, origin=watch.origin,
                        destination=watch.destination, departure_time=watch.departure_time.isoformat())
        try:
            return self.status_source.status(ref).status == "cancelled"
        except status_mod.SourceNotConfigured:
            self.status_source = None   # stub until SWIM R3: stop asking this run
            return False

    def _send(self, watch, kind, prob, now, stats) -> str:
        if not self.store.claim(watch.watch_id, kind, now):
            return "skipped"   # another process already sent it
        stats["attempted"] += 1
        try:
            self.sender(watch.subscription, push.payload_for(watch, kind, prob))
        except push.PushGone:
            self.store.delete_endpoint(watch.endpoint)
            stats["gone"] += 1
            return "gone"
        except push.PushError as e:
            self.store.release(watch.watch_id, kind)
            stats["errors"] += 1
            logger.warning("watcher: send failed for %s: %s", watch.watch_id, e)
            return "error"
        stats["sent"] += 1
        self.sleep(self.send_delay_s)   # rate limit
        return "sent"

    # ------------------------------------------------------------------ loop
    def cycle_with_alerts(self) -> Optional[dict]:
        """run_once that never raises; reports repeated failures through the alert hook (once per outage)."""
        try:
            stats = self.run_once()
            failed = stats["attempted"] > 0 and stats["sent"] + stats["gone"] == 0 and stats["errors"] > 0
        except Exception as e:
            logger.error("watcher cycle failed: %s", type(e).__name__)
            stats, failed = None, True
        self._failures = self._failures + 1 if failed else 0
        if self._failures >= FAILURES_BEFORE_ALERT and not self._alerted:
            self._alerted = True
            self._send_alert("FlyRely copilot watcher failing",
                             f"{self._failures} consecutive failed cycles (see the watcher log; no endpoints logged).")
        elif self._failures == 0:
            self._alerted = False
        return stats

    def _send_alert(self, subject: str, body: str) -> None:
        try:
            if self.alert:
                self.alert(subject, body)
            else:
                import alerts   # repo-root module, only imported when needed
                alerts.send_alert(subject, body)
        except Exception as e:
            logger.warning("watcher: alert hook failed: %s", type(e).__name__)

    async def run_forever(self, interval_s: float) -> None:
        while True:
            await asyncio.to_thread(self.cycle_with_alerts)
            await asyncio.sleep(interval_s)


# ---------------------------------------------------------------------- in-process wiring
def _under_pytest() -> bool:
    return "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules


def build_default(score: Optional[Callable]) -> Watcher:
    return Watcher(push_store.PushStore(), score, push.webpush_sender(), status_mod.get_status_source())


def attach(app, score: Optional[Callable], watcher: Optional[Watcher] = None, force: bool = False) -> bool:
    """Register startup/shutdown hooks on `app`. Returns whether the watcher was wired. Never wires under
    pytest (unless `force`), when any flag is off, or when VAPID is not configured."""
    if not force and (_under_pytest() or not config.watcher_enabled()):
        return False
    if watcher is None and not push.vapid_configured():
        logger.warning("copilot watcher requested but VAPID keys are not configured; not starting")
        return False
    w = watcher or build_default(score)
    interval = config.watch_interval_s()

    async def _start():
        app.state.copilot_watcher_task = asyncio.create_task(w.run_forever(interval))

    async def _stop():
        task = getattr(app.state, "copilot_watcher_task", None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    app.router.on_startup.append(_start)
    app.router.on_shutdown.append(_stop)
    return True


# ---------------------------------------------------------------------- CLI
def _load_env_file() -> None:
    from pathlib import Path
    path = Path(os.environ.get("FLYRELY_ENV_FILE", Path(__file__).resolve().parent.parent / ".env"))
    if path.is_file():
        from dotenv import load_dotenv
        load_dotenv(path, override=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="FlyRely copilot push watcher (separate process).")
    ap.add_argument("--once", action="store_true", help="run one cycle and exit (for cron/launchd interval jobs)")
    ap.add_argument("--interval", type=float, default=None, help="seconds between cycles (default: env or 900)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    _load_env_file()
    if not config.watcher_enabled():
        print("copilot watcher is off: set FLYRELY_FEATURE_DISRUPTION_COPILOT=1, FLYRELY_COPILOT_PUSH=1 and "
              "FLYRELY_COPILOT_WATCHER=1", file=sys.stderr)
        return 0
    if not push.vapid_configured():
        print("copilot watcher: VAPID keys are not configured (scripts/gen_vapid_keys.py)", file=sys.stderr)
        return 0
    import main as api_main   # loads the models; only the CLI does this, the copilot package never imports main
    w = build_default(api_main._copilot_scorer())
    if args.once:
        return 0 if w.cycle_with_alerts() is not None else 1
    interval = args.interval or config.watch_interval_s()

    async def _run():
        task = asyncio.create_task(w.run_forever(interval))
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, task.cancel)
        try:
            await task
        except asyncio.CancelledError:
            pass
    asyncio.run(_run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
