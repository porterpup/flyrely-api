"""Pluggable alert hook. No channel chosen yet, so with nothing configured it only logs.

Channels (any combination), configured by environment variables:
  FLYRELY_ALERT_DISCORD_WEBHOOK   Discord webhook URL
  FLYRELY_ALERT_NTFY_TOPIC        ntfy topic (FLYRELY_ALERT_NTFY_SERVER, default https://ntfy.sh)
  FLYRELY_ALERT_SMTP_HOST         SMTP host, with FLYRELY_ALERT_SMTP_PORT (587), _USER, _PASSWORD,
                                  _FROM, _TO (comma separated); STARTTLS is used unless port is 465 (SSL)
Secrets are never logged.
"""

import json
import logging
import os
import smtplib
import urllib.request
from email.message import EmailMessage

logger = logging.getLogger("flyrely.alerts")


def configured_channels(env=None) -> list:
    env = os.environ if env is None else env
    ch = []
    if env.get("FLYRELY_ALERT_DISCORD_WEBHOOK"):
        ch.append("discord")
    if env.get("FLYRELY_ALERT_NTFY_TOPIC"):
        ch.append("ntfy")
    if env.get("FLYRELY_ALERT_SMTP_HOST") and env.get("FLYRELY_ALERT_SMTP_TO"):
        ch.append("smtp")
    return ch


def _post(url: str, data: bytes, headers: dict) -> None:
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=10) as resp:
        resp.read()


def send_alert(subject: str, body: str, env=None) -> list:
    """Send to every configured channel. Never raises. Returns the channels that succeeded
    (empty list, with an INFO log, when nothing is configured)."""
    env = os.environ if env is None else env
    channels = configured_channels(env)
    if not channels:
        logger.info(f"ALERT (no channel configured): {subject}\n{body}")
        return []
    ok = []
    for ch in channels:
        try:
            if ch == "discord":
                text = f"**{subject}**\n{body}"[:1900]
                _post(env["FLYRELY_ALERT_DISCORD_WEBHOOK"], json.dumps({"content": text}).encode(),
                      {"Content-Type": "application/json", "User-Agent": "flyrely-alerts"})
            elif ch == "ntfy":
                server = env.get("FLYRELY_ALERT_NTFY_SERVER", "https://ntfy.sh").rstrip("/")
                _post(f"{server}/{env['FLYRELY_ALERT_NTFY_TOPIC']}", body.encode(),
                      {"Title": subject.encode("ascii", "replace").decode(), "User-Agent": "flyrely-alerts"})
            elif ch == "smtp":
                msg = EmailMessage()
                msg["Subject"] = subject
                msg["From"] = env.get("FLYRELY_ALERT_SMTP_FROM") or env.get("FLYRELY_ALERT_SMTP_USER", "flyrely@localhost")
                msg["To"] = env["FLYRELY_ALERT_SMTP_TO"]
                msg.set_content(body)
                port = int(env.get("FLYRELY_ALERT_SMTP_PORT", "587"))
                cls = smtplib.SMTP_SSL if port == 465 else smtplib.SMTP
                with cls(env["FLYRELY_ALERT_SMTP_HOST"], port, timeout=10) as s:
                    if port != 465:
                        s.starttls()
                    if env.get("FLYRELY_ALERT_SMTP_USER"):
                        s.login(env["FLYRELY_ALERT_SMTP_USER"], env.get("FLYRELY_ALERT_SMTP_PASSWORD", ""))
                    s.send_message(msg)
            ok.append(ch)
        except Exception as e:  # message only: exception text for URL errors can echo the webhook
            logger.warning(f"Alert via {ch} failed: {type(e).__name__}")
    return ok


def evaluate(report: dict, recent_prediction_count: int, env=None) -> list:
    """Alert messages for a score report: zero prediction volume, or calibration drift.

    Thresholds (env, defaults): FLYRELY_ALERT_ECE_MAX=0.05, FLYRELY_ALERT_MIN_N=500,
    FLYRELY_ALERT_BASELINE_BRIER (unset = skip), FLYRELY_ALERT_BRIER_MARGIN=0.02,
    FLYRELY_ALERT_VOLUME_HOURS=24. The drift check uses the report's `recent` groups (rolling window)."""
    env = os.environ if env is None else env
    msgs = []
    hours = float(env.get("FLYRELY_ALERT_VOLUME_HOURS", "24"))
    if recent_prediction_count == 0:
        msgs.append(f"No predictions logged in the last {hours:g} hours.")
    ece_max = float(env.get("FLYRELY_ALERT_ECE_MAX", "0.05"))
    min_n = int(env.get("FLYRELY_ALERT_MIN_N", "500"))
    base = env.get("FLYRELY_ALERT_BASELINE_BRIER")
    margin = float(env.get("FLYRELY_ALERT_BRIER_MARGIN", "0.02"))
    for g in report.get("recent", []):
        if g["n"] < min_n:
            continue
        label = f"{g['model_version']} / {g['horizon']} (n={g['n']}, last {report['recent_days']}d)"
        if g["ece"] is not None and g["ece"] > ece_max:
            msgs.append(f"Calibration drift: {label} ECE {g['ece']:.3f} > {ece_max:.3f}.")
        if base and g["brier"] is not None and g["brier"] > float(base) + margin:
            msgs.append(f"Accuracy drift: {label} Brier {g['brier']:.4f} > baseline {float(base):.4f} + {margin:g}.")
    return msgs


def run_checks(report: dict, recent_prediction_count: int, env=None) -> list:
    """Evaluate and send one combined alert if anything fired. Returns the messages."""
    msgs = evaluate(report, recent_prediction_count, env)
    if msgs:
        send_alert("FlyRely accuracy alert", "\n".join(msgs), env)
    return msgs
