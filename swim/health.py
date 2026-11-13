"""Health check for the SWIM consumers. Run every few minutes (launchd / systemd timer / cron).

Alerts (ntfy via swim/ntfy.py, once per incident plus "recovered", when FLYRELY_NTFY_TOPIC is set; also through alerts.py: ntfy, Discord or SMTP from environment variables; a logged no-op when none
is set) when, for any watched source:
  * no message has arrived for --max-silence-min minutes, or the status file stopped updating
    (consumer down);
  * reconnects repeat: at least --reconnects in the last --reconnect-window-min minutes;
  * the consumer is parked after an authentication failure;
local API /health or web / not answering, and when free disk space is below --min-free-gb.
Each problem alerts once per --cooldown-min (state in <status-dir>/health_state.json) and re-arms
when it clears. Exit code 0 healthy, 1 problems found.

Also:  python -m swim.health --status           (one summary line per source)
       python -m swim.health --unpark tfms        (after fixing credentials, allow the consumer to run again)
"""
import argparse
import json
import logging
import os
import sys
import time
import urllib.request
from pathlib import Path

from swim import monitor
from swim.retention import free_gb

log = logging.getLogger("swim.health")


def check(status_dir: Path, sources, root: Path, now: float, max_silence_min=10.0, reconnects=5,
          reconnect_window_min=30.0, min_free_gb=10.0, stale_status_min=3.0) -> dict:
    """Returns {problem_key: message}. Pure function of the files on disk (offline, testable)."""
    problems = {}
    for src in sources:
        sp = monitor.park_path(status_dir, src)
        if sp.exists():
            try:
                since = json.loads(sp.read_text()).get("parked_at", "?")
            except (OSError, ValueError):
                since = "?"
            problems[f"{src}:parked"] = (f"{src}: PARKED since {since} after an authentication failure. "
                                         f"Fix the credentials, then run: python -m swim.health --unpark {src}")
            continue
        stp = monitor.status_path(status_dir, src)
        try:
            st = json.loads(stp.read_text())
        except (OSError, ValueError):
            problems[f"{src}:nostatus"] = f"{src}: no readable status file (consumer never started, or it is down)."
            continue
        if st.get("state") == "config_error":
            problems[f"{src}:config"] = f"{src}: configuration error ({st.get('last_error')}). The consumer is waiting; fix it and restart the service."
            continue
        upd = st.get("updated_epoch") or 0
        if now - upd > stale_status_min * 60:
            problems[f"{src}:down"] = f"{src}: status not updated for {(now - upd) / 60:.0f} min (consumer down or hung)."
            continue
        ref = st.get("last_message_epoch") or st.get("started_epoch") or upd
        if now - ref > max_silence_min * 60:
            problems[f"{src}:silence"] = (f"{src}: no messages for {(now - ref) / 60:.0f} min "
                                          f"(state={st.get('state')}, reconnects={st.get('reconnect_count')}"
                                          + (f", tls_reason={st['tls_reason']}" if st.get("tls_reason") else "") + ").")
        recent = [t for t in st.get("reconnect_epochs_24h", []) if now - t <= reconnect_window_min * 60]
        if len(recent) >= reconnects:
            problems[f"{src}:reconnects"] = (f"{src}: {len(recent)} reconnects in the last "
                                             f"{reconnect_window_min:g} min (threshold {reconnects}).")
    try:
        free = free_gb(root)
        if free < min_free_gb:
            problems["disk:low"] = f"disk low: {free:.1f} GB free under the SWIM archive (threshold {min_free_gb:g} GB)."
    except OSError:
        pass
    return problems


def check_http(urls: dict, timeout: float = 8.0, retries: int = 2) -> dict:
    """{name: url} -> problems for urls that do not answer below HTTP 400 (after retries)."""
    problems = {}
    for name, url in urls.items():
        err = "?"
        for i in range(retries + 1):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "flyrely-health"})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    err = None if r.status < 400 else f"HTTP {r.status}"
            except Exception as e:
                code = getattr(e, "code", None)
                err = f"HTTP {code}" if code else type(e).__name__
            if err is None:
                break
            if i < retries:
                time.sleep(2)
        if err:
            problems[f"http:{name}"] = f"{name} check failed ({url}): {err}."
    return problems


def due(problems: dict, state: dict, now: float, cooldown_min: float):
    """(messages to send now, new state). Problems that cleared are forgotten so they re-alert."""
    new_state, send = {}, []
    for k, msg in problems.items():
        last = state.get(k)
        if last is None or now - last >= cooldown_min * 60:
            send.append(msg)
            new_state[k] = now
        else:
            new_state[k] = last
    return send, new_state


def _test_alert() -> int:
    """Push one fake failure, a duplicate (must not send), then the recovery, through the real dedup path.
    Uses a temp state file; no service is touched."""
    import tempfile
    from swim import ntfy
    sf = Path(tempfile.mkdtemp()) / "ntfy_state.json"
    fake = {"test:fake": "TEST ONLY: simulated failure, ignore."}
    ntfy.notify_transitions(fake, sf)
    ntfy.notify_transitions(fake, sf)
    ntfy.notify_transitions({}, sf)
    print("test alert done (1 failure + 1 recovery expected)")
    return 0


def main(argv=None) -> int:
    from swim.consumer import DATA_ROOT, ENV_PATH
    env = os.environ
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DATA_ROOT)
    ap.add_argument("--status-dir", type=Path, default=None, help="default <root>/status")
    ap.add_argument("--sources", default=env.get("SWIM_SOURCES", "tfms,sfdps"), help="comma separated")
    ap.add_argument("--max-silence-min", type=float, default=float(env.get("SWIM_HEALTH_SILENCE_MIN", 15)))
    ap.add_argument("--reconnects", type=int, default=5)
    ap.add_argument("--reconnect-window-min", type=float, default=30)
    ap.add_argument("--min-free-gb", type=float, default=float(env.get("SWIM_MIN_FREE_GB", 10)))
    ap.add_argument("--cooldown-min", type=float, default=60)
    ap.add_argument("--api-url", default=env.get("SWIM_HEALTH_API_URL", "http://127.0.0.1:8000/health"),
                    help="local API health URL; empty string disables the check")
    ap.add_argument("--web-url", default=env.get("SWIM_HEALTH_WEB_URL", "http://127.0.0.1:3000/"),
                    help="local web URL; empty string disables the check")
    ap.add_argument("--test-alert", action="store_true",
                    help="send a simulated failure then recovery via a temp state file; checks nothing real")
    ap.add_argument("--dry-run", action="store_true", help="print problems; send nothing, keep no state")
    ap.add_argument("--status", action="store_true", help="print one summary line per source and exit")
    ap.add_argument("--unpark", metavar="SOURCE", help="remove the auth-failure park marker and exit")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    status_dir = args.status_dir or args.root / "status"

    if args.status:
        for src in [x for x in args.sources.split(",") if x]:
            try:
                d = json.loads(monitor.status_path(status_dir, src).read_text())
            except (OSError, ValueError):
                print(f"{src}: no status file")
                continue
            print(f"{src}: {d.get('state')} received/min={d.get('msgs_per_min_5m')} kept/min={d.get('kept_per_min_5m')} "
                  f"reconnects={d.get('reconnect_count')} last_message={d.get('last_message_at')} codec={d.get('archive_codec')}")
        return 0

    if args.unpark:
        print("unparked" if monitor.unpark(status_dir, args.unpark) else "was not parked")
        return 0

    # alert settings: optional .env.alerts next to the main env file (FLYRELY_ALERT_*), then the env file
    try:
        from dotenv import load_dotenv
        load_dotenv(ENV_PATH.with_name(ENV_PATH.name + ".alerts"))
        load_dotenv(ENV_PATH)
    except ImportError:
        pass
    if args.test_alert:
        return _test_alert()
    now = time.time()
    problems = check(status_dir, [s for s in args.sources.split(",") if s], args.root, now,
                     args.max_silence_min, args.reconnects, args.reconnect_window_min, args.min_free_gb)
    urls = {n: u for n, u in (("api", args.api_url), ("web", args.web_url)) if u}
    problems.update(check_http(urls))
    for msg in problems.values():
        log.warning("%s", msg)
    if args.dry_run:
        print("healthy" if not problems else f"{len(problems)} problem(s)")
        return 1 if problems else 0

    state_file = status_dir / "health_state.json"
    try:
        state = json.loads(state_file.read_text())
    except (OSError, ValueError):
        state = {}
    send, new_state = due(problems, state, now, args.cooldown_min)
    if send and any(os.environ.get(k) for k in ("FLYRELY_ALERT_DISCORD_WEBHOOK", "FLYRELY_ALERT_NTFY_TOPIC",
                                                "FLYRELY_ALERT_SMTP_HOST")):
        from alerts import send_alert  # legacy channels (cooldown re-alerts); ntfy below is once per incident
        send_alert("FlyRely SWIM alert", "\n".join(send))
    from swim import ntfy
    ntfy.notify_transitions(problems, status_dir / "ntfy_state.json")
    try:
        monitor.write_json_atomic(state_file, new_state)
    except OSError:
        pass
    if not problems:
        log.info("healthy")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
