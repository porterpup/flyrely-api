"""ntfy.sh push notifications with once-per-incident de-duplication.

Env: FLYRELY_NTFY_TOPIC (unset = log only, never an error), FLYRELY_NTFY_SERVER (default https://ntfy.sh).
The topic acts as the password, so it is never logged.

State is a small JSON file {problem_key: message} of the problems currently open. `transitions` compares it
with the current problems: new keys alert once, keys that disappeared send "recovered".
"""
import json
import logging
import os
import urllib.request
from pathlib import Path

log = logging.getLogger("swim.ntfy")


def transitions(problems: dict, state: dict):
    """(opened, recovered, new_state). opened/recovered are lists of messages."""
    opened = [m for k, m in problems.items() if k not in state]
    recovered = [state[k] for k in state if k not in problems]
    return opened, recovered, dict(problems)


def load_state(path: Path) -> dict:
    try:
        d = json.loads(Path(path).read_text())
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def save_state(path: Path, state: dict) -> None:
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(state))
    os.replace(tmp, p)


def send(title: str, body: str, priority: str = "default", tags: str = "", env=None) -> bool:
    """POST to ntfy. Returns True if sent. Never raises; no topic means a logged no-op."""
    env = os.environ if env is None else env
    topic = env.get("FLYRELY_NTFY_TOPIC", "").strip()
    if not topic:
        log.info("ntfy not configured (FLYRELY_NTFY_TOPIC unset): %s | %s", title, body)
        return False
    server = (env.get("FLYRELY_NTFY_SERVER") or "https://ntfy.sh").rstrip("/")
    headers = {"Title": title.encode("ascii", "replace").decode(), "Priority": priority,
               "User-Agent": "flyrely-alerts"}
    if tags:
        headers["Tags"] = tags
    try:
        req = urllib.request.Request(f"{server}/{topic}", data=body.encode(), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            r.read()
        return True
    except Exception as e:  # type only: URL errors can echo the topic
        log.warning("ntfy send failed: %s", type(e).__name__)
        return False


def notify_transitions(problems: dict, state_file: Path, env=None, dry=False) -> bool:
    """Alert on newly opened problems, send 'recovered' for cleared ones, persist the state.
    A notification that fails to send is retried on the next run (state is not advanced for it).
    With no topic configured the state is not advanced, so adding the topic later still alerts."""
    env = os.environ if env is None else env
    state = load_state(state_file)
    opened, recovered, new_state = transitions(problems, state)
    if not (opened or recovered):
        return True
    if not env.get("FLYRELY_NTFY_TOPIC", "").strip():
        log.info("ntfy not configured: %d opened, %d recovered (not sent)", len(opened), len(recovered))
        return False
    ok = True
    if opened and not send("FlyRely problem", "\n".join(opened), "high", "rotating_light", env):
        ok = False
        new_state = {k: m for k, m in new_state.items() if k in state}
    if recovered and not send("FlyRely recovered", "\n".join(f"recovered: {m}" for m in recovered),
                              "default", "white_check_mark", env):
        ok = False
        for k in state:
            if k not in problems:
                new_state[k] = state[k]
    if not dry:
        save_state(state_file, new_state)
    return ok
