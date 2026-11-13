"""Heartbeat / status file and the auth-failure 'park' marker for one SWIM consumer process.

Status file: <status_dir>/<source>.json, rewritten atomically about every 15 s. Contains counts
and times only (never credentials, hostnames or message content). The health check reads it.
Park marker: <status_dir>/<source>.parked. While it exists the consumer refuses to connect, so a
reboot or a service manager can never retry bad credentials in a loop. Remove it with
`python -m swim.health --unpark <source>` after fixing the credentials.
"""
import json
import os
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

WRITE_EVERY_S = 15.0
RECONNECT_HISTORY_S = 24 * 3600


def iso(epoch):
    return None if epoch is None else datetime.fromtimestamp(epoch, timezone.utc).isoformat(timespec="seconds")


def status_path(status_dir, source) -> Path:
    return Path(status_dir) / f"{source}.json"


def park_path(status_dir, source) -> Path:
    return Path(status_dir) / f"{source}.parked"


def write_json_atomic(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def park(status_dir, source, reason: str, now=None):
    write_json_atomic(park_path(status_dir, source), {"parked_at": iso(now or time.time()), "reason": reason})


def unpark(status_dir, source) -> bool:
    try:
        park_path(status_dir, source).unlink()
        return True
    except FileNotFoundError:
        return False


class Stats:
    def __init__(self, source, status_dir, codec="", clock=time.time):
        self.source, self.status_dir, self.codec, self.clock = source, Path(status_dir), codec, clock
        self.started = clock()
        self.state = "starting"
        self.last_error = None
        self.last_message = None
        self.received = self.kept = 0
        self.reasons = Counter()
        self.reconnects = deque()  # epoch of each reconnect
        self.reconnect_total = 0
        self._minutes = deque()  # (minute_epoch, received, kept)
        self._next_write = 0.0
        self.bytes_written = 0
        self.tls_reason = None  # sub-code name of the last TLS failure; cleared on connect
        self.consecutive_failures = 0
        self.backoff_s = 0
        self.first_started = self.started  # carried across restarts: when this silence began at the latest
        self.restarts = []  # epochs this process (re)started, carried across restarts, last 24 h
        self.last_tls_exit = None  # epoch of the last TLS fast-exit (restart cool-down)

    def carry_forward(self, now=None):
        """Load what must survive a restart from the previous status file: last_message, restart history,
        last TLS exit. Without this each new process resets the silence clock and health never fires."""
        now = now or self.clock()
        try:
            prev = json.loads(status_path(self.status_dir, self.source).read_text())
        except (OSError, ValueError):
            return
        self.last_message = prev.get("last_message_epoch")
        self.first_started = prev.get("first_started_epoch") or prev.get("started_epoch") or self.started
        self.last_tls_exit = prev.get("last_tls_exit_epoch")
        self.restarts = [t for t in prev.get("restart_epochs_24h", []) if now - t <= RECONNECT_HISTORY_S] + [round(now)]
        self.write(force=True)

    def message(self, kept: bool, reason: str = ""):
        now = self.clock()
        self.last_message = now
        self.received += 1
        self.kept += bool(kept)
        if reason:
            self.reasons[reason] += 1
        m = int(now // 60) * 60
        if self._minutes and self._minutes[-1][0] == m:
            _, r, k = self._minutes.pop()
            self._minutes.append((m, r + 1, k + bool(kept)))
        else:
            self._minutes.append((m, 1, int(bool(kept))))
        while self._minutes and self._minutes[0][0] < now - 15 * 60:
            self._minutes.popleft()

    def reconnect(self):
        now = self.clock()
        self.reconnect_total += 1
        self.reconnects.append(now)
        while self.reconnects and self.reconnects[0] < now - RECONNECT_HISTORY_S:
            self.reconnects.popleft()

    def set_state(self, state, error=None):
        self.state = state
        if error is not None:
            self.last_error = error
        self.write(force=True)

    def _per_min(self, idx, window_min=5):
        now = self.clock()
        cur = int(now // 60) * 60
        rows = [r for r in self._minutes if cur - window_min * 60 <= r[0] < cur]  # complete minutes only
        return round(sum(r[idx] for r in rows) / window_min, 1)

    def snapshot(self) -> dict:
        now = self.clock()
        return {
            "source": self.source, "pid": os.getpid(), "state": self.state,
            "started_at": iso(self.started), "started_epoch": self.started, "updated_at": iso(now), "updated_epoch": now,
            "last_message_at": iso(self.last_message), "last_message_epoch": self.last_message,
            "messages_received": self.received, "messages_kept": self.kept,
            "msgs_per_min_5m": self._per_min(1), "kept_per_min_5m": self._per_min(2),
            "reconnect_count": self.reconnect_total,
            "reconnect_epochs_24h": [round(t) for t in self.reconnects],
            "archive_codec": self.codec, "archive_bytes_written": self.bytes_written,
            "decisions": dict(self.reasons.most_common(25)), "last_error": self.last_error,
            "tls_reason": self.tls_reason, "consecutive_failures": self.consecutive_failures,
            "backoff_s": self.backoff_s,
            "first_started_epoch": self.first_started, "restart_epochs_24h": self.restarts,
            "last_tls_exit_epoch": self.last_tls_exit,
        }

    def write(self, force=False):
        now = self.clock()
        if not force and now < self._next_write:
            return
        self._next_write = now + WRITE_EVERY_S
        try:
            write_json_atomic(status_path(self.status_dir, self.source), self.snapshot())
        except OSError:
            pass  # a full or read-only disk must not take the consumer down; health will notice
