"""FAA SWIM SCDS consumer (tfms/sfdps/stdds): archiver.

Binds to the persistent Solace queue, appends each batch of messages to an
hourly-rotated gzip JSONL file, fsyncs, and only then acknowledges them.

Credentials come from flyrely-api/.env (SCDS_<SOURCE>_*) and are never logged;
every error string is passed through redact() first.

Usage:  python -m swim.consumer --source tfms --minutes 10
"""
import argparse
import base64
import gzip
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ENV_PATH = Path(os.environ.get("FLYRELY_ENV_FILE", "/Users/seanbutton/Projects/flyrely/flyrely-api/.env"))
DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "swim"
SOURCES = ("tfms", "sfdps", "stdds")
SOURCE = "tfms"  # set by main()
FIELDS = ("URL", "USER", "PASSWORD", "QUEUE", "VPN")


def secret_keys(sources=SOURCES):
    return tuple(f"SCDS_{s.upper()}_{f}" for s in sources for f in FIELDS)
AUTH_HINTS = ("login failure", "authoriz", "authentic", "unauthorized", "invalid username", "401", "403")
BATCH_MAX = 100
BATCH_SECONDS = 1.0

log = logging.getLogger("tfms")


class AuthError(Exception):
    pass


def redact(text: str) -> str:
    """Replace any credential/host/queue value in text with ***."""
    for k in secret_keys():
        v = os.environ.get(k)
        if v:
            text = text.replace(v, "***")
            if "://" in v:  # bare hostname inside a tcps://host:port URL
                text = text.replace(v.split("://", 1)[1].split(":")[0], "***")
    return text


def hour_path(root: Path, ts: datetime) -> Path:
    d = root / ts.strftime("%Y-%m-%d")
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{SOURCE}_{ts.strftime('%Y%m%dT%H')}.jsonl.gz"


def encode_record(received_at: datetime, props: dict, body: bytes) -> bytes:
    rec = {"recv": received_at.isoformat(), "props": props}
    try:
        rec["body"] = body.decode("utf-8")
    except UnicodeDecodeError:
        rec["body_b64"] = base64.b64encode(body).decode("ascii")
    return (json.dumps(rec, default=str) + "\n").encode("utf-8")


def write_batch(root: Path, records: list) -> None:
    """records: list of (received_at, line_bytes). Each batch is one gzip member,
    fsynced before returning, so acking afterwards is safe."""
    by_file = {}
    for ts, line in records:
        by_file.setdefault(hour_path(root, ts), []).append(line)
    for path, lines in by_file.items():
        with open(path, "ab") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb") as gz:
                gz.write(b"".join(lines))
            raw.flush()
            os.fsync(raw.fileno())


def trust_store_dir() -> str:
    """Solace wants a directory of PEM files. Use SCDS_TRUST_STORE_DIR, else split
    certifi's CA bundle into a temp dir (macOS has no /etc/ssl/certs)."""
    d = os.environ.get("SCDS_TRUST_STORE_DIR")
    if d:
        return d
    import re
    import tempfile
    import certifi
    out = Path(tempfile.mkdtemp(prefix="swim_trust_"))
    pems = re.findall(r"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----",
                      Path(certifi.where()).read_text(), re.S)
    for i, pem in enumerate(pems):
        (out / f"ca{i:03d}.pem").write_text(pem + "\n")
    return str(out)


def build_service():
    from solace.messaging.messaging_service import MessagingService, RetryStrategy
    from solace.messaging.config.transport_security_strategy import TLS
    from solace.messaging.config.solace_properties import (
        authentication_properties as ap, service_properties as sp, transport_layer_properties as tp)

    props = {
        tp.HOST: os.environ["SCDS_" + SOURCE.upper() + "_URL"],
        sp.VPN_NAME: os.environ["SCDS_" + SOURCE.upper() + "_VPN"],
        ap.SCHEME_BASIC_USER_NAME: os.environ["SCDS_" + SOURCE.upper() + "_USER"],
        ap.SCHEME_BASIC_PASSWORD: os.environ["SCDS_" + SOURCE.upper() + "_PASSWORD"],
        tp.COMPRESSION_LEVEL: int(os.environ.get("SCDS_COMPRESSION_LEVEL", "1")),  # FAA requires compression
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


def consume(root: Path, deadline, stop: dict) -> int:
    from solace.messaging.resources.queue import Queue
    from solace.messaging.errors.pubsubplus_client_error import PubSubPlusClientError

    n = 0
    service = build_service()
    try:
        try:
            service.connect()
        except PubSubPlusClientError as e:
            if any(h in redact(str(e)).lower() for h in AUTH_HINTS):
                raise AuthError("authentication failed (check the SCDS user / password / VPN settings)") from None
            raise
        log.info("connected")
        receiver = service.create_persistent_message_receiver_builder().build(
            Queue.durable_non_exclusive_queue(os.environ["SCDS_" + SOURCE.upper() + "_QUEUE"]))
        receiver.start()
        log.info("bound to queue")
        pending, batch_start = [], time.monotonic()
        while not stop["flag"] and (deadline is None or time.time() < deadline):
            msg = receiver.receive_message(timeout=1000)
            now = datetime.now(timezone.utc)
            if msg is not None:
                body = msg.get_payload_as_bytes()
                if body is None:
                    body = (msg.get_payload_as_string() or "").encode("utf-8")
                props = dict(msg.get_properties() or {})
                props["_redelivered"] = msg.is_redelivered()
                pending.append((msg, now, encode_record(now, props, bytes(body))))
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
    import faulthandler
    faulthandler.enable()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from dotenv import load_dotenv
    load_dotenv(ENV_PATH)
    missing = [k for k in secret_keys([SOURCE]) if not os.environ.get(k)]
    if missing:
        log.error("missing settings: %s", ", ".join(missing))
        return 2

    deadline = time.time() + args.minutes * 60 if args.minutes else None
    stop = {"flag": False}
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.update(flag=True))

    total, backoff = 0, 5
    while not stop["flag"] and (deadline is None or time.time() < deadline):
        try:
            got = consume(out, deadline, stop)
            total += got
            if got:
                backoff = 5
        except AuthError as e:
            log.error("%s", e)
            return 3
        except Exception as e:
            log.warning("transient error (%s): %s; reconnecting in %ss",
                        type(e).__name__, redact(str(e))[:200], backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
    log.info("done: %d messages archived", total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
