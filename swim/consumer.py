"""FAA SWIM SCDS consumer (tfms/sfdps/stdds): archiver.

Binds to the persistent Solace queue, appends each batch of messages to a
gzip JSONL file (one file per process run per hour), fsyncs, and only then
acknowledges them.

Credentials come from the .env file (SCDS_<SOURCE>_*) and are never logged:
all log output and error strings pass through redact().

Usage:  python -m swim.consumer --source tfms --minutes 10
Env:    FLYRELY_ENV_FILE (default: <repo>/.env), SCDS_TRUST_STORE_DIR,
        SCDS_COMPRESSION_LEVEL (1-9, default 1; 0 is refused: FAA requires compression)
"""
import argparse
import atexit
import base64
import gzip
import hashlib
import json
import logging
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


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
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


def trust_store_dir() -> str:
    """Solace wants a directory of PEM files. Use SCDS_TRUST_STORE_DIR, else split
    certifi's CA bundle into a temp dir once per process, removed on exit."""
    global _TRUST_DIR
    d = os.environ.get("SCDS_TRUST_STORE_DIR")
    if d:
        return d
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
def hour_path(root: Path, ts: datetime) -> Path:
    d = root / ts.strftime("%Y-%m-%d")
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{SOURCE}_{ts.strftime('%Y%m%dT%H')}_{RUN_ID}.jsonl.gz"


def encode_record(received_at: datetime, props: dict, body: bytes, meta=None) -> bytes:
    rec = {"recv": received_at.isoformat(), "props": props,
           "dedupe": {**(meta or {}), "sha256": hashlib.sha256(body).hexdigest()}}
    try:
        rec["body"] = body.decode("utf-8")
    except UnicodeDecodeError:
        rec["body_b64"] = base64.b64encode(body).decode("ascii")
    return (json.dumps(rec, default=str) + "\n").encode("utf-8")


def _fsync_dir(path: Path):
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_batch(root: Path, records: list) -> None:
    """records: list of (received_at, line_bytes). Each batch is one gzip member,
    fsynced (and the directory too, for new files) before returning, so acking
    afterwards is safe."""
    by_file = {}
    for ts, line in records:
        by_file.setdefault(hour_path(root, ts), []).append(line)
    for path, lines in by_file.items():
        new = not path.exists()
        with open(path, "ab") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb") as gz:
                gz.write(b"".join(lines))
            raw.flush()
            os.fsync(raw.fileno())
        if new:
            _fsync_dir(path.parent)
            _fsync_dir(path.parent.parent)


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


def _flush(receiver, root, pending) -> int:
    write_batch(root, [(t, line) for _, t, line in pending])
    for m, _, _ in pending:  # ack only after fsync
        receiver.ack(m)
    return len(pending)


def consume(root: Path, deadline) -> int:
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
        pending, batch_start = [], time.monotonic()
        while not STOP.is_set() and (deadline is None or time.time() < deadline):
            msg = receiver.receive_message(timeout=1000)
            now = datetime.now(timezone.utc)
            if msg is not None:
                body = msg.get_payload_as_bytes()
                if body is None:
                    body = (msg.get_payload_as_string() or "").encode("utf-8")
                props = dict(msg.get_properties() or {})
                props["_redelivered"] = msg.is_redelivered()
                pending.append((msg, now, encode_record(now, props, bytes(body), message_meta(msg))))
            if pending and (len(pending) >= BATCH_MAX or time.monotonic() - batch_start >= BATCH_SECONDS):
                n += _flush(receiver, root, pending)
                pending = []
            if not pending:
                batch_start = time.monotonic()
        if pending:
            n += _flush(receiver, root, pending)
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=float, default=None, help="stop cleanly after N minutes")
    parser.add_argument("--source", choices=SOURCES, required=True)
    parser.add_argument("--out", type=Path, default=None, help="default data/swim/<source>")
    args = parser.parse_args(argv)
    global SOURCE
    SOURCE = args.source
    out = args.out or DATA_ROOT / SOURCE
    setup_logging()

    from dotenv import load_dotenv
    load_dotenv(ENV_PATH)
    missing = [k for k in secret_keys([SOURCE]) if not os.environ.get(k)]
    if missing:
        log.error("missing settings: %s", ", ".join(missing))
        return 2
    try:
        require_tcps(os.environ[f"SCDS_{SOURCE.upper()}_URL"])
        compression_level()
    except ConfigError as e:
        log.error("%s", e)
        return 2
    warm_ip_cache()

    deadline = time.time() + args.minutes * 60 if args.minutes else None
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: STOP.set())

    total, backoff = 0, 5
    while not STOP.is_set() and (deadline is None or time.time() < deadline):
        try:
            got = consume(out, deadline)
            total += got
            if got:
                backoff = 5
        except AuthError as e:
            log.error("%s", e)
            return 3
        except Exception as e:
            log.warning("transient error (%s): %s; reconnecting in %ss",
                        type(e).__name__, redact(str(e))[:200], backoff)
            STOP.wait(backoff)  # interruptible by SIGTERM/SIGINT
            backoff = min(backoff * 2, 60)
    log.info("done: %d messages archived", total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
