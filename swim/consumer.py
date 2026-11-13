"""FAA SWIM SCDS consumer (tfms/sfdps/stdds): archiver.

Binds to the persistent Solace queue, filters each message at receive time
(swim/filters.py), appends the kept ones to one long-lived compressed JSONL stream per
process run per hour (swim/archive.py, zstd or gzip), fsyncs, and only then
acknowledges the whole batch (dropped messages are acknowledged too).
Writes a heartbeat file (swim/monitor.py). Exit codes: 0 clean, 2 config error,
3 authentication failure, 4 safety-net exit (too many consecutive failures or open file descriptors: launchd
KeepAlive restarts a clean process). On a config error or auth failure the service parks IN-PROCESS (status
`config_error` / `parked`, no connection attempts, waits until SIGTERM); an auth park resumes by itself after
`python -m swim.health --unpark <source>`. --exit-on-park exits with 2/3 instead (manual runs).

Credentials come from the .env file (SCDS_<SOURCE>_*) and are never logged:
all log output and error strings pass through redact().

Usage:  python -m swim.consumer --source tfms --minutes 10
Env:    FLYRELY_ENV_FILE (default: <repo>/.env), SCDS_TRUST_STORE_DIR,
        SCDS_COMPRESSION_LEVEL (1-9, default 1; 0 is refused: FAA requires compression),
        SWIM_ARCHIVE_CODEC (auto|zstd|gzip), SWIM_ARCHIVE_LEVEL, SWIM_FILTER_CONFIG, SWIM_FILTER=off,
        SWIM_BLOCKLIST_FILE (LADD block list, default data/ladd_blocklist.txt)
"""
import argparse
import base64
import hashlib
import json
import logging
import logging.handlers
import os
import re
import signal
import socket
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from swim import archive, filters, monitor

REPO = Path(__file__).resolve().parent.parent
ENV_PATH = Path(os.environ.get("FLYRELY_ENV_FILE", REPO / ".env"))
DATA_ROOT = REPO / "data" / "swim"
SOURCES = ("tfms", "sfdps", "stdds")
SOURCE = "tfms"  # set by main()
FIELDS = ("URL", "USER", "PASSWORD", "QUEUE", "VPN")
AUTH_RE = re.compile(r"login[_ ]failure|authentication|unauthori[sz]ed|not authori[sz]ed|"
                     r"authori[sz]ation failed|invalid (user|client) ?name", re.I)
TLS_RE = re.compile(r"CERTIFICATE|SSL|TLS|handshake", re.I)
HOSTNAME_RE = re.compile(r"COMMON_?NAME", re.I)  # wrong host / name mismatch: a configuration error, not transient
SUBCODE_RE = re.compile(r"SOLCLIENT_SUBCODE_[A-Z0-9_]+")
BACKOFF_BASE_S = 5.0
BACKOFF_CAP_S = 600.0  # 10 min
MAX_CONSECUTIVE_FAILURES = 30  # env SWIM_MAX_CONSECUTIVE_FAILURES overrides (read in main())
TLS_FAILURES_BEFORE_RESTART = 2  # a TLS error that repeats means this process is wedged (e.g. emptied trust store)
EXIT_RESTART = 4
BATCH_MAX = 100
BATCH_SECONDS = 1.0
RUN_ID = f"{datetime.now(timezone.utc):%H%M%S}p{os.getpid()}"  # one file per process run

log = logging.getLogger("swim")
STOP = threading.Event()


class AuthError(Exception):
    pass


class ConfigError(Exception):
    pass


class TlsError(Exception):
    """Certificate / TLS handshake failure on connect. Retried with backoff, but a repeat means this process is
    wedged, so main() exits for a clean restart. Known cause (2026-11-13): the trust-store folder was in $TMPDIR and
    macOS temp cleanup emptied it, so every later handshake said 'untrusted certificate'; a restart rebuilt it."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class Backoff:
    """Exponential delay for consecutive failures, capped; reset() after a successful connect."""

    def __init__(self, base=BACKOFF_BASE_S, cap=BACKOFF_CAP_S):
        self.base, self.cap, self.failures = base, cap, 0

    def next(self) -> float:
        delay = min(self.base * 2 ** min(self.failures, 30), self.cap)
        self.failures += 1
        return delay

    def reset(self):
        self.failures = 0


def open_fds() -> int:
    """Open file descriptors of this process (-1 if unknown)."""
    for d in ("/dev/fd", "/proc/self/fd"):
        try:
            return len(os.listdir(d))
        except OSError:
            continue
    return -1


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def max_open_fds() -> int:
    """Safety-net threshold: SWIM_MAX_OPEN_FDS, else half the soft RLIMIT_NOFILE (default 500 if unknown)."""
    if os.environ.get("SWIM_MAX_OPEN_FDS", "").strip().isdigit():
        return int(os.environ["SWIM_MAX_OPEN_FDS"])
    try:
        import resource
        soft = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        if soft > 0:
            return max(int(soft * 0.5), 50)
    except (ImportError, OSError, ValueError):
        pass
    return 500


def secret_keys(sources=SOURCES):
    return tuple(f"SCDS_{s.upper()}_{f}" for s in sources for f in FIELDS)


# ---------------------------------------------------------------- redaction
def _hosts(url: str):
    for part in url.split(","):
        part = part.strip()
        host = part.split("://", 1)[-1].split("/")[0]
        host = host.rsplit(":", 1)[0] if host.count(":") == 1 else host
        if host:
            yield host


_IP_CACHE: dict = {}  # host -> resolved IPs; filled once at startup (no DNS inside log calls)


def warm_ip_cache():
    """Resolve configured hosts once so redact() can also mask their IPs."""
    for s in SOURCES:
        for h in _hosts(os.environ.get(f"SCDS_{s.upper()}_URL", "")):
            try:
                _IP_CACHE[h] = tuple(socket.gethostbyname_ex(h)[2])
            except OSError:
                _IP_CACHE[h] = ()


def secret_values() -> list:
    vals = set()
    for k in secret_keys():
        v = os.environ.get(k)
        if not v:
            continue
        vals.add(v)
        if k.endswith("_URL"):
            vals.update(p.strip() for p in v.split(","))
            for h in _hosts(v):
                vals.add(h)
                vals.update(_IP_CACHE.get(h, ()))
    return sorted((x for x in vals if len(x) >= 3), key=len, reverse=True)


def redact(text: str) -> str:
    """Mask every credential / host / VPN / queue / resolved-IP value (case-insensitive)."""
    vals = secret_values()
    if not vals:
        return text
    return re.sub("|".join(re.escape(v) for v in vals), "***", text, flags=re.I)


class RedactingFilter(logging.Filter):
    def filter(self, record):
        try:
            record.msg = redact(record.getMessage())
            record.args = None
        except Exception:
            record.msg, record.args = "<unprintable log message>", None
        return True


def setup_logging(log_file=None):
    """Console logging, or (with log_file) ONLY a rotating file: 10 MB x 5, so an always-on process can
    never fill the disk with logs (launchd/systemd capture of stderr would be unbounded)."""
    root = logging.getLogger()
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        for h in list(root.handlers):
            root.removeHandler(h)
        fh = logging.handlers.RotatingFileHandler(log_file, maxBytes=10_000_000, backupCount=5)
        fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        root.addHandler(fh)
        root.setLevel(logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for h in root.handlers:
        if not any(isinstance(f, RedactingFilter) for f in h.filters):
            h.addFilter(RedactingFilter())
    logging.getLogger("solace").setLevel(logging.WARNING)


# ------------------------------------------------------------------- config
def require_tcps(url: str) -> str:
    parts = [p.strip() for p in url.split(",") if p.strip()]
    if not parts or not all(p.lower().startswith("tcps://") for p in parts):
        raise ConfigError("SCDS URL must start with tcps:// (refusing to send credentials unencrypted)")
    return url


def compression_level() -> int:
    try:
        lvl = int(os.environ.get("SCDS_COMPRESSION_LEVEL", "1"))
    except ValueError:
        raise ConfigError("SCDS_COMPRESSION_LEVEL must be an integer 1-9") from None
    if not 1 <= lvl <= 9:
        raise ConfigError("SCDS_COMPRESSION_LEVEL must be 1-9 (the FAA requires compression)")
    return lvl


def is_auth_error(message: str) -> bool:
    return bool(AUTH_RE.search(message))


def is_tls_error(message: str) -> bool:
    return bool(TLS_RE.search(message))


def tls_reason(message: str) -> str:
    """Short, secret-free reason: the Solace sub-code name if present, else a generic label."""
    m = SUBCODE_RE.search(message)
    return m.group(0) if m else "TLS_ERROR"


TRUST_ROOT = DATA_ROOT / "trust"  # persistent (data/ is gitignored); NOT $TMPDIR, which macOS cleans periodically


LINUX_CERT_DIR = Path("/etc/ssl/certs")


def _build_trust_dir(out: Path):
    import certifi
    out.mkdir(parents=True, exist_ok=True)
    pems = re.findall(r"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----",
                      Path(certifi.where()).read_text(), re.S)
    for i, pem in enumerate(pems):
        (out / f"ca{i:03d}.pem").write_text(pem + "\n")


def _clean_stale_tmp_trust_dirs():
    """Remove empty swim_trust_* folders left in $TMPDIR by older versions (best effort)."""
    for d in Path(tempfile.gettempdir()).glob("swim_trust_*"):
        try:
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
        except OSError:
            pass


def trust_store_dir() -> str:
    """Solace wants a directory of PEM files. Called on EVERY build_service(), so a folder that vanished or was
    emptied is rebuilt instead of failing every later handshake.
    1. SCDS_TRUST_STORE_DIR if set.
    2. Linux: the system directory /etc/ssl/certs (needs the ca-certificates package).
    3. Otherwise (macOS, python.org Python has no PEM directory of its own): certifi's CA bundle split into
       data/swim/trust/<source>/ (persistent), rebuilt when missing or empty."""
    d = os.environ.get("SCDS_TRUST_STORE_DIR")
    if d:
        return d
    if sys.platform.startswith("linux") and LINUX_CERT_DIR.is_dir() and any(LINUX_CERT_DIR.iterdir()):
        return str(LINUX_CERT_DIR)
    out = TRUST_ROOT / SOURCE
    if not (out.is_dir() and any(out.glob("*.pem"))):
        log.info("trust store missing or empty: rebuilding")
        _build_trust_dir(out)
        _clean_stale_tmp_trust_dirs()
    return str(out)


# ------------------------------------------------------------------ archive
def encode_record(received_at: datetime, props: dict, body: bytes, meta=None) -> bytes:
    rec = {"recv": received_at.isoformat(), "props": props,
           "dedupe": {**(meta or {}), "sha256": hashlib.sha256(body).hexdigest()}}
    try:
        rec["body"] = body.decode("utf-8")
    except UnicodeDecodeError:
        rec["body_b64"] = base64.b64encode(body).decode("ascii")
    return (json.dumps(rec, default=str) + "\n").encode("utf-8")


def message_meta(msg) -> dict:
    """Dedupe identifiers (best effort; any may be absent)."""
    meta = {}
    for name, fn in (("rgmid", "get_replication_group_message_id"), ("app_id", "get_application_message_id"),
                     ("sender_ts", "get_sender_timestamp")):
        try:
            v = getattr(msg, fn)()
            meta[name] = None if v is None else (v if isinstance(v, (int, str)) else str(v))
        except Exception:
            meta[name] = None
    return meta


# ------------------------------------------------------------------- solace
def build_service():
    from solace.messaging.messaging_service import MessagingService, RetryStrategy
    from solace.messaging.config.transport_security_strategy import TLS
    from solace.messaging.config.solace_properties import (
        authentication_properties as ap, service_properties as sp, transport_layer_properties as tp)

    p = "SCDS_" + SOURCE.upper() + "_"
    props = {
        tp.HOST: require_tcps(os.environ[p + "URL"]),
        sp.VPN_NAME: os.environ[p + "VPN"],
        ap.SCHEME_BASIC_USER_NAME: os.environ[p + "USER"],
        ap.SCHEME_BASIC_PASSWORD: os.environ[p + "PASSWORD"],
        tp.COMPRESSION_LEVEL: compression_level(),
    }
    return (MessagingService.builder().from_properties(props)
            # full certificate + hostname validation against the CA bundle
            .with_transport_security_strategy(
                TLS.create().with_certificate_validation(False, True, trust_store_dir()))
            .with_reconnection_retry_strategy(RetryStrategy.parametrized_retry(20, 3000))
            .build())


def _flush(receiver, writer, pending) -> int:
    """pending: [(msg, received_at, line_or_None)]. None = filtered out (still acked)."""
    writer.write_batch([(t, line) for _, t, line in pending if line is not None])
    for m, _, _ in pending:  # ack only after fsync
        receiver.ack(m)
    return len(pending)


def _release(service, receiver):
    """Free everything one attempt created (receiver, then the messaging service and its Solace context).
    Runs in a finally on every attempt: a leaked context holds sockets for the life of the process."""
    if receiver is not None:
        try:
            receiver.terminate(grace_period=0)  # unacked prefetched msgs are redelivered later
        except Exception as e:
            log.debug("receiver terminate: %s", type(e).__name__)
    if service is not None:
        try:
            service.disconnect()
        except Exception as e:
            log.warning("service disconnect failed: %s", type(e).__name__)
        # solace-pubsubplus 1.11: a failed connect or dropped session marks the service DISCONNECTED before
        # connect() returns, so disconnect() returns early ("already disconnected") and never frees the Solace
        # context (about 2 fds and 1 thread leaked per attempt). session_cleanup() does free it and is safe to
        # call twice. Private API, so guarded: if it ever disappears we only lose this extra release.
        try:
            service._session.session_cleanup()
        except Exception as e:
            log.debug("session cleanup: %s", type(e).__name__)


def consume(writer, flt, stats, deadline) -> int:
    n = 0
    service = receiver = None
    try:
        service = build_service()
        try:
            service.connect()
        except Exception as e:
            msg = str(e)
            if is_auth_error(msg):
                raise AuthError("authentication failed (check the SCDS user / password / VPN settings)") from None
            if HOSTNAME_RE.search(msg):
                raise ConfigError("TLS hostname / common-name mismatch (check the SCDS URL)") from None
            if is_tls_error(msg):
                raise TlsError(tls_reason(msg)) from None
            raise
        from solace.messaging.resources.queue import Queue
        log.info("connected")
        receiver = service.create_persistent_message_receiver_builder().build(
            Queue.durable_non_exclusive_queue(os.environ["SCDS_" + SOURCE.upper() + "_QUEUE"]))
        receiver.start()
        log.info("bound to queue")
        if getattr(stats, "on_connected", None):
            stats.on_connected()  # only now is the attempt a success (resets backoff / failure count)
        stats.set_state("connected")
        pending, batch_start = [], time.monotonic()
        while not STOP.is_set() and (deadline is None or time.time() < deadline):
            msg = receiver.receive_message(timeout=1000)
            now = datetime.now(timezone.utc)
            if msg is not None:
                body = msg.get_payload_as_bytes()
                if body is None:
                    body = (msg.get_payload_as_string() or "").encode("utf-8")
                body = bytes(body)
                props = dict(msg.get_properties() or {})
                kept_body, reason = body, "keep:binary"
                try:
                    text = body.decode("utf-8")
                except UnicodeDecodeError:
                    text = None
                if text is None:
                    if not flt.check_binary(body):
                        kept_body, reason = None, "drop:ladd_text"
                else:
                    out, reason = flt.apply(props, text, now.timestamp())
                    kept_body = None if out is None else (body if out is text else out.encode("utf-8"))
                props["_redelivered"] = msg.is_redelivered()
                line = None if kept_body is None else encode_record(now, props, kept_body, message_meta(msg))
                stats.message(line is not None, reason)
                pending.append((msg, now, line))
            if pending and (len(pending) >= BATCH_MAX or time.monotonic() - batch_start >= BATCH_SECONDS):
                n += _flush(receiver, writer, pending)
                stats.bytes_written = writer.bytes_written
                pending = []
            if not pending:
                batch_start = time.monotonic()
            stats.write()
        if pending:
            n += _flush(receiver, writer, pending)
    finally:
        _release(service, receiver)
    return n


def alert_auth_failure(source: str):
    """Best-effort alert (no secrets in the text); a no-op when no channel is configured."""
    try:
        from alerts import send_alert
        send_alert(f"FlyRely SWIM {source}: authentication failed",
                   f"The {source} consumer was rejected by SWIM and is PARKED: it will not retry. "
                   f"Check the SCDS credentials, then run: python -m swim.health --unpark {source}")
    except Exception as e:
        log.warning("auth alert failed: %s", type(e).__name__)


def self_check(source: str, out: Path, flt_cfg: dict) -> int:
    """Offline checks (never connects). Prints setting NAMES and OK/FAIL only."""
    bad = 0

    def line(ok, what):
        nonlocal bad
        bad += (not ok)
        print(("OK   " if ok else "FAIL ") + what)

    missing = [k for k in secret_keys([source]) if not os.environ.get(k)]
    line(not missing, "settings present: " + (", ".join(missing) + " MISSING" if missing else "all SCDS_" + source.upper() + "_*"))
    try:
        require_tcps(os.environ.get(f"SCDS_{source.upper()}_URL", ""))
        line(True, "URL is tcps://")
    except ConfigError as e:
        line(False, str(e))
    try:
        compression_level()
        line(True, "SCDS_COMPRESSION_LEVEL valid")
    except ConfigError as e:
        line(False, str(e))
    try:
        from solace.messaging.messaging_service import MessagingService, RetryStrategy  # noqa: F401
        from solace.messaging.config.transport_security_strategy import TLS  # noqa: F401
        from solace.messaging.config.solace_properties import (  # noqa: F401
            authentication_properties, service_properties, transport_layer_properties)
        from solace.messaging.resources.queue import Queue  # noqa: F401
        from solace.messaging.errors.pubsubplus_client_error import PubSubPlusClientError  # noqa: F401
        line(True, "solace-pubsubplus imports (every name the consumer uses)")
    except Exception as e:
        line(False, f"solace-pubsubplus import: {type(e).__name__}")
    try:
        td = trust_store_dir()
        line(Path(td).is_dir() and any(Path(td).iterdir()), f"trust store directory ({'system' if td == str(LINUX_CERT_DIR) else 'certifi split or override'})")
    except Exception as e:
        line(False, f"trust store: {type(e).__name__}")
    try:
        codec = archive.resolve_codec(os.environ.get("SWIM_ARCHIVE_CODEC", "auto"))
        line(True, f"archive codec: {codec}")
    except ValueError as e:
        line(False, str(e))
    try:
        out.mkdir(parents=True, exist_ok=True)
        probe = out / ".write_probe"
        probe.write_bytes(b"x")
        probe.unlink()
        line(True, "archive directory writable")
    except OSError:
        line(False, "archive directory not writable")
    line(True, f"filter: {'on' if flt_cfg.get('enabled', True) else 'OFF'}; "
               f"block list: {'present' if blocklist_path().exists() else 'absent (no LADD filtering)'}")
    return 0 if not bad else 2


def blocklist_path() -> Path:
    return Path(os.environ.get("SWIM_BLOCKLIST_FILE", DATA_ROOT.parent / "ladd_blocklist.txt"))


PARK_POLL_S = 10.0


def sleep_interruptible(seconds: float, stats, step: float = 5.0):
    """Wait out a backoff, still refreshing the status file (so health sees 'silence', not 'down') and
    returning early on SIGTERM/SIGINT."""
    left = seconds
    while left > 0 and not STOP.is_set():
        STOP.wait(min(step, left))
        left -= step
        stats.write()


def hold(stats, state, err, status_dir, exit_on_park, code, resumable):
    """Park IN-PROCESS: record the state, then wait (no reconnecting, no exit) until SIGTERM/SIGINT.
    resumable (auth park): returns None as soon as the park marker is removed (`swim.health --unpark`),
    so no restart is needed. Returns the exit code otherwise. With exit_on_park it returns `code` at once."""
    stats.set_state(state, err)
    if exit_on_park:
        return code
    log.error("%s: waiting (no connection attempts) until %s", state,
              "unparked or stopped" if resumable else "stopped; fix the configuration and restart")
    while not STOP.wait(PARK_POLL_S):
        if resumable and not monitor.park_path(status_dir, SOURCE).exists():
            log.info("unparked: resuming")
            return None
        stats.write(force=True)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=float, default=None, help="stop cleanly after N minutes")
    parser.add_argument("--source", choices=SOURCES, required=True)
    parser.add_argument("--out", type=Path, default=None, help="default data/swim/<source>")
    parser.add_argument("--status-dir", type=Path, default=None, help="default data/swim/status")
    parser.add_argument("--filter-config", default=None, help="JSON, default $SWIM_FILTER_CONFIG or built-in")
    parser.add_argument("--log-file", type=Path, default=None, help="log only to this rotating file")
    parser.add_argument("--exit-on-park", action="store_true",
                        help="manual runs: exit (2 config error, 3 auth failure/parked) instead of waiting in-process")
    parser.add_argument("--check", action="store_true", help="offline self-check; never connects")
    args = parser.parse_args(argv)
    global SOURCE
    SOURCE = args.source
    out = args.out or DATA_ROOT / SOURCE
    status_dir = args.status_dir or DATA_ROOT / "status"
    setup_logging(args.log_file)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: STOP.set())

    from dotenv import load_dotenv
    load_dotenv(ENV_PATH)
    load_dotenv(ENV_PATH.with_name(ENV_PATH.name + ".alerts"))  # optional FLYRELY_ALERT_* settings
    stats = monitor.Stats(SOURCE, status_dir)

    def config_error(msg):
        log.error("%s", msg)
        return hold(stats, "config_error", msg, status_dir, args.exit_on_park, 2, resumable=False)

    try:
        flt_cfg = filters.load_config(args.filter_config)[SOURCE]
    except (OSError, ValueError) as e:
        return config_error(f"filter config: {type(e).__name__}")
    if args.check:
        return self_check(SOURCE, out, flt_cfg)
    missing = [k for k in secret_keys([SOURCE]) if not os.environ.get(k)]
    if missing:
        return config_error("missing settings: " + ", ".join(missing))
    try:
        require_tcps(os.environ[f"SCDS_{SOURCE.upper()}_URL"])
        compression_level()
        codec = archive.resolve_codec(os.environ.get("SWIM_ARCHIVE_CODEC", "auto"))
    except (ConfigError, ValueError) as e:
        return config_error(str(e))
    warm_ip_cache()

    flt = filters.MessageFilter(SOURCE, flt_cfg, filters.Blocklist(blocklist_path()))
    writer = archive.ArchiveWriter(out, SOURCE, RUN_ID, codec, os.environ.get("SWIM_ARCHIVE_LEVEL"))
    stats.codec = writer.codec
    log.info("source=%s archive codec=%s filter=%s", SOURCE, writer.codec, "on" if flt_cfg.get("enabled", True) else "off")

    deadline = time.time() + args.minutes * 60 if args.minutes else None
    total, first = 0, True
    backoff = Backoff()

    def connected():
        nonlocal tls_streak
        tls_streak = 0
        backoff.reset()
        stats.tls_reason = None
        stats.consecutive_failures = 0
        stats.backoff_s = 0

    stats.on_connected = connected
    max_failures = _env_int("SWIM_MAX_CONSECUTIVE_FAILURES", MAX_CONSECUTIVE_FAILURES)
    fd_limit = max_open_fds()
    tls_streak = 0

    def give_up(why):
        log.error("safety net: %s; exiting %d so the service manager restarts a clean process", why, EXIT_RESTART)
        stats.set_state("restarting", why)
        return EXIT_RESTART
    try:
        while not STOP.is_set() and (deadline is None or time.time() < deadline):
            if monitor.park_path(status_dir, SOURCE).exists():
                log.error("PARKED after an authentication failure: not connecting. Fix the credentials, then run: "
                          "python -m swim.health --unpark %s", SOURCE)
                r = hold(stats, "parked", "authentication failure", status_dir, args.exit_on_park, 3, resumable=True)
                if r is not None:
                    return r
                continue
            fds = open_fds()
            if fds > fd_limit:
                return give_up(f"{fds} open file descriptors (limit {fd_limit})")
            if not first:
                stats.reconnect()
            first = False
            try:
                got = consume(writer, flt, stats, deadline)
                total += got
                if STOP.is_set() or (deadline is not None and time.time() >= deadline):
                    break
            except ConfigError as e:
                return config_error(str(e))
            except AuthError as e:
                log.error("%s", e)
                monitor.park(status_dir, SOURCE, "authentication failure")
                alert_auth_failure(SOURCE)
                continue  # the loop head parks (waits) before any further connection attempt
            except Exception as e:
                tls = isinstance(e, TlsError)
                tls_streak = tls_streak + 1 if tls else 0
                err = f"TlsError: {e.reason}" if tls else f"{type(e).__name__}: {redact(str(e))[:200]}"
                delay = backoff.next()
                stats.consecutive_failures, stats.backoff_s = backoff.failures, delay
                if tls:
                    stats.tls_reason = e.reason
                    log.warning("tls_error (%s): attempt %d failed, retrying in %.0fs", e.reason, backoff.failures, delay)
                else:
                    log.warning("transient error (%s): attempt %d failed, reconnecting in %.0fs", err, backoff.failures, delay)
                stats.set_state("tls_error" if tls else "reconnecting", err)
                if tls_streak >= TLS_FAILURES_BEFORE_RESTART:
                    return give_up(f"{tls_streak} consecutive TLS errors ({e.reason})")
                if backoff.failures >= max_failures:
                    return give_up(f"{backoff.failures} consecutive connection failures")
                sleep_interruptible(delay, stats)
        stats.set_state("stopped")
    finally:
        writer.close()
    log.info("done: %d messages received, %d kept", total, stats.kept)
    return 0


if __name__ == "__main__":
    sys.exit(main())
