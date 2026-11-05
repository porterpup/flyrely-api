"""FAA SWIM SCDS consumer (tfms/sfdps/stdds): archiver.

Binds to the persistent Solace queue, filters each message at receive time
(swim/filters.py), appends the kept ones to one long-lived compressed JSONL stream per
process run per hour (swim/archive.py, zstd or gzip), fsyncs, and only then
acknowledges the whole batch (dropped messages are acknowledged too).
Writes a heartbeat file (swim/monitor.py). Exit codes: 0 clean, 2 config error,
3 authentication failure (the consumer parks itself: it will not reconnect until
`python -m swim.health --unpark <source>`; see --auth-exit-code).

Credentials come from the .env file (SCDS_<SOURCE>_*) and are never logged:
all log output and error strings pass through redact().

Usage:  python -m swim.consumer --source tfms --minutes 10
Env:    FLYRELY_ENV_FILE (default: <repo>/.env), SCDS_TRUST_STORE_DIR,
        SCDS_COMPRESSION_LEVEL (1-9, default 1; 0 is refused: FAA requires compression),
        SWIM_ARCHIVE_CODEC (auto|zstd|gzip), SWIM_ARCHIVE_LEVEL, SWIM_FILTER_CONFIG, SWIM_FILTER=off,
        SWIM_BLOCKLIST_FILE (LADD block list, default data/ladd_blocklist.txt)
"""
import argparse
import atexit
import base64
import hashlib
import json
import logging
import logging.handlers
import os
import re
import shutil
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
BATCH_MAX = 100
BATCH_SECONDS = 1.0
RUN_ID = f"{datetime.now(timezone.utc):%H%M%S}p{os.getpid()}"  # one file per process run

log = logging.getLogger("swim")
STOP = threading.Event()


class AuthError(Exception):
    pass


class ConfigError(Exception):
    pass


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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if log_file:  # rotating: 10 MB x 5, so an always-on process can never fill the disk with logs
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(log_file, maxBytes=10_000_000, backupCount=5)
        fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logging.getLogger().addHandler(fh)
    for h in logging.getLogger().handlers:
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


_TRUST_DIR = None


LINUX_CERT_DIR = Path("/etc/ssl/certs")


def trust_store_dir() -> str:
    """Solace wants a directory of PEM files.
    1. SCDS_TRUST_STORE_DIR if set.
    2. Linux: the system directory /etc/ssl/certs (needs the ca-certificates package).
    3. Otherwise (macOS, python.org Python has no PEM directory of its own): split certifi's CA
       bundle into a temp dir once per process, removed on exit."""
    global _TRUST_DIR
    d = os.environ.get("SCDS_TRUST_STORE_DIR")
    if d:
        return d
    if sys.platform.startswith("linux") and LINUX_CERT_DIR.is_dir() and any(LINUX_CERT_DIR.iterdir()):
        return str(LINUX_CERT_DIR)
    if _TRUST_DIR is None:
        import certifi
        out = Path(tempfile.mkdtemp(prefix="swim_trust_"))
        atexit.register(shutil.rmtree, out, True)
        pems = re.findall(r"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----",
                          Path(certifi.where()).read_text(), re.S)
        for i, pem in enumerate(pems):
            (out / f"ca{i:03d}.pem").write_text(pem + "\n")
        _TRUST_DIR = str(out)
    return _TRUST_DIR


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


def consume(writer, flt, stats, deadline) -> int:
    from solace.messaging.resources.queue import Queue
    from solace.messaging.errors.pubsubplus_client_error import PubSubPlusClientError

    n = 0
    service = build_service()
    try:
        try:
            service.connect()
        except PubSubPlusClientError as e:
            if is_auth_error(str(e)):
                raise AuthError("authentication failed (check the SCDS user / password / VPN settings)") from None
            raise
        log.info("connected")
        receiver = service.create_persistent_message_receiver_builder().build(
            Queue.durable_non_exclusive_queue(os.environ["SCDS_" + SOURCE.upper() + "_QUEUE"]))
        receiver.start()
        log.info("bound to queue")
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
                if text is not None:
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
        try:
            receiver.terminate(grace_period=0)  # unacked prefetched msgs are redelivered later
        except Exception:
            pass
    finally:
        try:
            service.disconnect()
        except Exception:
            pass
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=float, default=None, help="stop cleanly after N minutes")
    parser.add_argument("--source", choices=SOURCES, required=True)
    parser.add_argument("--out", type=Path, default=None, help="default data/swim/<source>")
    parser.add_argument("--status-dir", type=Path, default=None, help="default data/swim/status")
    parser.add_argument("--filter-config", default=None, help="JSON, default $SWIM_FILTER_CONFIG or built-in")
    parser.add_argument("--log-file", type=Path, default=None, help="also log to this rotating file")
    parser.add_argument("--auth-exit-code", type=int, default=3,
                        help="exit code after an auth failure (launchd uses 0 so KeepAlive does not respawn)")
    parser.add_argument("--check", action="store_true", help="offline self-check; never connects")
    args = parser.parse_args(argv)
    global SOURCE
    SOURCE = args.source
    out = args.out or DATA_ROOT / SOURCE
    status_dir = args.status_dir or DATA_ROOT / "status"
    setup_logging(args.log_file)

    from dotenv import load_dotenv
    load_dotenv(ENV_PATH)
    load_dotenv(ENV_PATH.with_name(ENV_PATH.name + ".alerts"))  # optional FLYRELY_ALERT_* settings
    try:
        flt_cfg = filters.load_config(args.filter_config)[SOURCE]
    except (OSError, ValueError) as e:
        log.error("filter config: %s", type(e).__name__)
        return 2
    if args.check:
        return self_check(SOURCE, out, flt_cfg)
    missing = [k for k in secret_keys([SOURCE]) if not os.environ.get(k)]
    if missing:
        log.error("missing settings: %s", ", ".join(missing))
        return 2
    try:
        require_tcps(os.environ[f"SCDS_{SOURCE.upper()}_URL"])
        compression_level()
        codec = archive.resolve_codec(os.environ.get("SWIM_ARCHIVE_CODEC", "auto"))
    except (ConfigError, ValueError) as e:
        log.error("%s", e)
        return 2
    warm_ip_cache()

    if monitor.park_path(status_dir, SOURCE).exists():
        log.error("PARKED after an authentication failure: not connecting. Fix the credentials, then run: "
                  "python -m swim.health --unpark %s", SOURCE)
        return args.auth_exit_code

    flt = filters.MessageFilter(SOURCE, flt_cfg, filters.Blocklist(blocklist_path()))
    writer = archive.ArchiveWriter(out, SOURCE, RUN_ID, codec, os.environ.get("SWIM_ARCHIVE_LEVEL"))
    stats = monitor.Stats(SOURCE, status_dir, writer.codec)
    log.info("source=%s archive codec=%s filter=%s", SOURCE, writer.codec, "on" if flt_cfg.get("enabled", True) else "off")

    deadline = time.time() + args.minutes * 60 if args.minutes else None
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: STOP.set())

    total, backoff, first = 0, 5, True
    try:
        while not STOP.is_set() and (deadline is None or time.time() < deadline):
            if not first:
                stats.reconnect()
            first = False
            try:
                got = consume(writer, flt, stats, deadline)
                total += got
                if got:
                    backoff = 5
            except AuthError as e:
                log.error("%s", e)
                monitor.park(status_dir, SOURCE, "authentication failure")
                stats.set_state("parked", "authentication failure")
                alert_auth_failure(SOURCE)
                return args.auth_exit_code
            except Exception as e:
                err = f"{type(e).__name__}: {redact(str(e))[:200]}"
                log.warning("transient error (%s): reconnecting in %ss", err, backoff)
                stats.set_state("reconnecting", err)
                STOP.wait(backoff)  # interruptible by SIGTERM/SIGINT
                backoff = min(backoff * 2, 60)
        stats.set_state("stopped")
    finally:
        writer.close()
    log.info("done: %d messages received, %d kept", total, stats.kept)
    return 0


if __name__ == "__main__":
    sys.exit(main())
