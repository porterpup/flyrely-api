import os
import sys
import tempfile
from pathlib import Path

import pytest

# Keep test predictions out of the real usage log
os.environ["FLYRELY_USAGE_LOG"] = str(Path(tempfile.mkdtemp()) / "usage_log.jsonl")
os.environ["FLYRELY_PREDICTION_DB"] = str(Path(tempfile.mkdtemp()) / "predictions.sqlite")
os.environ.pop("DATABASE_URL", None)
os.environ.pop("WEATHER_API_KEY", None)
os.environ["WEATHER_API_PROVIDER"] = "none"  # no network in tests; weather tests inject fixtures

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Env vars that decide whether a real push/alert can leave the machine. Never let a test inherit them.
NOTIFY_ENV_VARS = ("FLYRELY_NTFY_TOPIC", "FLYRELY_NTFY_SERVER", "FLYRELY_NTFY_ALLOW_IN_TESTS")


def _notify_env_names():
    return [k for k in os.environ if k.startswith("FLYRELY_ALERT_")] + list(NOTIFY_ENV_VARS)


class RealSendBlocked(BaseException):
    """BaseException, not Exception, so the send paths' `except Exception` cannot swallow it: the test fails."""


def _blocked_post(url, data, headers):
    raise RealSendBlocked("a test tried to send a real alert; patch swim.ntfy._post or alerts._post")


def pytest_collection_finish(session):
    # Test modules imported during collection can load a real .env (python-dotenv writes os.environ).
    for k in _notify_env_names():
        os.environ.pop(k, None)


@pytest.fixture(autouse=True)
def _no_real_alerts(monkeypatch, tmp_path):
    from swim import consumer, ntfy
    import alerts

    for k in _notify_env_names():
        monkeypatch.delenv(k, raising=False)
    no_env_file = tmp_path / "no-such-dir" / "flyrely.env"  # nonexistent: no real .env is ever read
    monkeypatch.setenv("FLYRELY_ENV_FILE", str(no_env_file))
    monkeypatch.setattr(consumer, "ENV_PATH", no_env_file)  # swim.health and swim.retention read this at call time
    monkeypatch.setattr(ntfy, "_post", _blocked_post)
    monkeypatch.setattr(alerts, "_post", _blocked_post)
    yield
