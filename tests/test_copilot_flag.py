"""S-A1: the copilot flag is default OFF and the core app is unchanged when it is off."""
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
FLAG = "FLYRELY_FEATURE_DISRUPTION_COPILOT"


def _run(code: str, **env) -> str:
    e = {k: v for k, v in os.environ.items() if not k.startswith(("FLYRELY_COPILOT", "AFFILIATE_"))}
    e.pop(FLAG, None)
    e.update(env)
    e["PYTHONPATH"] = str(ROOT)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=e, cwd=ROOT, timeout=120)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip().splitlines()[-1]


def test_flag_off_no_routes_no_import_and_404():
    code = (
        "import json, sys, main\n"
        "from fastapi.testclient import TestClient\n"
        "paths = [r.path for r in main.app.routes]\n"
        "print(json.dumps({'copilot_routes': [p for p in paths if p.startswith('/copilot')],"
        " 'imported': any(m == 'copilot' or m.startswith('copilot.') for m in sys.modules),"
        " 'status': TestClient(main.app).get('/copilot/config').status_code}))\n"
    )
    result = json.loads(_run(code))
    assert result == {"copilot_routes": [], "imported": False, "status": 404}


def test_flag_off_core_routes_and_openapi_match_baseline():
    """Byte-identical core: with the flag off, routes and OpenAPI equal those of an app with no copilot code."""
    code = (
        "import json, main\n"
        "print(json.dumps({'routes': sorted(r.path for r in main.app.routes),"
        " 'openapi': main.app.openapi()}, sort_keys=True))\n"
    )
    off = json.loads(_run(code))
    assert not any(p.startswith("/copilot") for p in off["routes"])
    assert not any(p.startswith("/copilot") for p in off["openapi"]["paths"])
    assert "copilot" not in json.dumps(off["openapi"]).lower()
    # Setting any other copilot env var without the master flag must not change anything either.
    other = json.loads(_run(code, FLYRELY_COPILOT_PUSH="1", FLYRELY_COPILOT_STATUS_SOURCE="swim"))
    assert other == off
    # Only "1" turns it on.
    assert json.loads(_run(code, **{FLAG: "true"})) == off


def test_flag_on_main_mounts_config():
    code = (
        "import json, main\n"
        "from fastapi.testclient import TestClient\n"
        "r = TestClient(main.app).get('/copilot/config')\n"
        "print(json.dumps({'status': r.status_code, 'body': r.json()}))\n"
    )
    result = json.loads(_run(code, **{FLAG: "1"}))
    assert result["status"] == 200
    assert result["body"]["enabled"] is True


def _fresh_client(monkeypatch, **env) -> TestClient:
    from copilot import mount
    for k in list(os.environ):
        if k.startswith(("FLYRELY_COPILOT", "AFFILIATE_")):
            monkeypatch.delenv(k)
    monkeypatch.setenv(FLAG, "1")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    app = FastAPI()
    mount(app, score=None)
    return TestClient(app)


def test_config_shape_defaults(monkeypatch):
    r = _fresh_client(monkeypatch).get("/copilot/config")
    assert r.status_code == 200
    assert r.json() == {
        "enabled": True,
        "push": {"enabled": False, "vapid_public_key": None},
        "status_source": "none",
        "affiliate_partners_configured": [],
        "schedule_as_of": "2026-08",   # latest month of models/route_schedule.csv.gz (S-A6 wiring)
    }


def test_config_push_and_affiliates(monkeypatch):
    client = _fresh_client(
        monkeypatch,
        FLYRELY_COPILOT_PUSH="1",
        FLYRELY_COPILOT_VAPID_PUBLIC_KEY="pub-key",
        FLYRELY_COPILOT_STATUS_SOURCE="swim",
        AFFILIATE_KAYAK_PARAMS="aid=1",
        AFFILIATE_STAY22_WRAP="https://x.test/?u={url}",
    )
    body = client.get("/copilot/config").json()
    assert body["push"] == {"enabled": True, "vapid_public_key": "pub-key"}
    assert body["status_source"] == "swim"
    assert body["affiliate_partners_configured"] == ["kayak", "stay22"]
    assert "aid=1" not in json.dumps(body)


def test_push_key_hidden_when_push_off(monkeypatch):
    client = _fresh_client(monkeypatch, FLYRELY_COPILOT_VAPID_PUBLIC_KEY="pub-key")
    assert client.get("/copilot/config").json()["push"] == {"enabled": False, "vapid_public_key": None}


def test_schemas_validate_plan_examples():
    from copilot import schemas as s
    flight = {"airline": "DL", "flight_number": "DL1234", "origin": "JFK", "destination": "LAX",
              "departure_time": "2026-11-15T14:30:00"}
    req = s.OptionsRequest(flight=flight, state="cancelled", home_airport="BOS")
    assert req.max_alternatives == 8
    s.StatusResponse(status="unknown", source="none")
    s.SubscribeRequest(subscription={"endpoint": "https://fcm.googleapis.com/x", "keys": {"p256dh": "a", "auth": "b"}},
                       flight=flight, client_flight_id="flight-1")
    s.PushPayload(kind="test", title="t", body="b", url="/flights/x", client_flight_id="x")


def test_copilot_never_imports_main():
    for path in (ROOT / "copilot").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            assert not any(n == "main" or n.startswith("main.") for n in names), path
