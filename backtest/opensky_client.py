"""Budgeted, cached OpenSky REST client (OAuth2 client_credentials).

Every raw response is cached to disk so reruns cost nothing. Every network call is appended
to a ledger with the X-Rate-Limit-Remaining header. The client refuses to call when the last
known remaining credits minus the expected cost would fall below the floor. The secret is
never logged or printed.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "opensky_cache"
LEDGER = CACHE_DIR / "ledger.jsonl"
TOKEN_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
API = "https://opensky-network.org/api"
FLOOR = 1500
TOTAL_CAP = 2500   # hard cap on credits spent by this experiment (all endpoint groups)
DAILY = 4000


class BudgetStop(Exception):
    pass


def _load_env():
    p = ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def cache_key(path, params):
    s = path + "?" + "&".join(f"{k}={params[k]}" for k in sorted(params))
    return hashlib.sha1(s.encode()).hexdigest()[:20], s


def group_of(path):
    return "/" + path.strip("/").split("/")[0]


def ledger_rows():
    if not LEDGER.exists():
        return []
    out = []
    for line in LEDGER.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def last_remaining(group):
    """Last X-Rate-Limit-Remaining seen in the ledger for an endpoint group (None if never called)."""
    rem = None
    for r in ledger_rows():
        if r.get("group") == group and r.get("remaining") is not None:
            rem = r["remaining"]
    return rem


def credits_spent():
    """Credits spent across all groups, from the ledger (observed cost, else expected cost)."""
    return sum((r.get("cost_observed") if r.get("cost_observed") is not None else r.get("expected_cost", 0))
               for r in ledger_rows() if r.get("status") == 200 or r.get("status") == 404)


class OpenSky:
    def __init__(self, floor=FLOOR, total_cap=TOTAL_CAP):
        _load_env()
        self.cid = os.environ.get("OPENSKY_USERNAME")
        self.sec = os.environ.get("OPENSKY_PASSWORD")
        self.floor = floor
        self._tok = None
        self._tok_exp = 0.0
        self.total_cap = total_cap
        self.remaining = None
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _token(self):
        if self._tok and time.time() < self._tok_exp - 120:
            return self._tok
        data = urllib.parse.urlencode({"grant_type": "client_credentials",
                                       "client_id": self.cid, "client_secret": self.sec}).encode()
        with urllib.request.urlopen(urllib.request.Request(TOKEN_URL, data=data), timeout=30) as resp:
            j = json.loads(resp.read())
        self._tok = j["access_token"]
        self._tok_exp = time.time() + int(j.get("expires_in", 1800))
        return self._tok

    def get(self, path, params, expected_cost):
        """Return (json, from_cache). Raises BudgetStop rather than dipping below the floor."""
        key, desc = cache_key(path, params)
        f = CACHE_DIR / f"{key}.json"
        if f.exists():
            return json.loads(f.read_text())["body"], True
        grp = group_of(path)
        prev = last_remaining(grp)
        if prev is None:
            prev = DAILY
        if prev - expected_cost < self.floor:
            raise BudgetStop(f"{grp}: remaining={prev} cost={expected_cost} floor={self.floor}")
        if credits_spent() + expected_cost > self.total_cap:
            raise BudgetStop(f"experiment cap {self.total_cap} would be exceeded")
        req = urllib.request.Request(API + path + "?" + urllib.parse.urlencode(params),
                                     headers={"Authorization": "Bearer " + self._token()})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                status, hdr, raw = resp.status, resp.headers, resp.read()
        except urllib.error.HTTPError as e:
            status, hdr, raw = e.code, e.headers, e.read()
        rem = hdr.get("X-Rate-Limit-Remaining")
        rem = int(float(rem)) if rem not in (None, "") else None
        cost_obs = (prev - rem) if (prev is not None and rem is not None and prev >= rem) else None
        with open(LEDGER, "a") as lf:
            lf.write(json.dumps({"ts": time.time(), "req": desc, "group": grp, "status": status, "remaining": rem,
                                 "cost_observed": cost_obs, "expected_cost": expected_cost,
                                 "retry_after": hdr.get("X-Rate-Limit-Retry-After-Seconds"),
                                 "bytes": len(raw)}) + "\n")
        self.remaining = rem
        if status == 429:
            raise BudgetStop(f"429 received; retry-after={hdr.get('X-Rate-Limit-Retry-After-Seconds')}")
        if status == 404:  # OpenSky returns 404 for 'no flights found'
            body = []
        else:
            if status != 200:
                raise RuntimeError(f"HTTP {status} for {desc}")
            body = json.loads(raw)
        f.write_text(json.dumps({"req": desc, "group": grp, "status": status, "remaining": rem, "body": body}))
        if self.remaining is not None and self.remaining < self.floor:
            raise BudgetStop(f"remaining {self.remaining} < floor {self.floor}")
        time.sleep(1.0)  # gentle pacing, never a tight loop
        return body, False
