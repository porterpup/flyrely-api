"""FAA traffic-management programs (ground stops, ground delay programs) known at the cutoff.

Built from ATCSCC advisories (``backtest/fetch_faa_adv.py``). An advisory is visible from its send time. Within one
(airport, kind) chain, an advisory stays in force until the NEXT advisory of that chain is sent (a revision replaces
it, a cancellation ends it). Nothing sent after the cutoff is ever used. Serving: the FAA NAS Status feed
(https://nasstatus.faa.gov/api/airport-status-information) lists programs currently in force.

GS and GDP both control ARRIVALS into the control element, so the destination features are the direct ones; the origin
features capture programs that are inbound-restricting the departure airport (a congestion signal).
"""

import numpy as np
import pandas as pd

from ._common import DAY0, SHIFT, T

GS_DEFAULT_MIN = 180      # period unknown (no body): assume a ground stop lasts 3 h from its send time
GDP_DEFAULT_MIN = 720     # period unknown: assume a GDP lasts 12 h from its send time
FEATURES = ["gs_now", "gdp_now", "gdp_prop", "gdp_cover"]
KINDS = ("GS", "GDP")


def _minutes(ts: pd.Series) -> np.ndarray:
    return ((pd.to_datetime(ts) - DAY0) / pd.Timedelta(minutes=1)).to_numpy(dtype="float64")


class Chain:
    """Advisories of one kind, per airport, in force from send time until the next advisory of the chain."""

    def __init__(self, prog: pd.DataFrame, airports: pd.Index, default_min: int):
        p = prog.copy()
        p["ap"] = airports.get_indexer(p["element"])
        p = p[p["ap"] >= 0]
        sent = _minutes(p["sent_utc"])
        ps, pe = _minutes(p["period_start"]), _minutes(p["period_end"])
        self.n_missing_period = int(np.isnan(pe).sum())
        ps = np.where(np.isnan(ps), sent, ps)
        pe = np.where(np.isnan(pe), sent + default_min, pe)
        key = p["ap"].to_numpy(np.int64) * T + np.floor(sent).astype(np.int64)
        order = np.argsort(key, kind="stable")
        self.key = key[order]
        self.sent = sent[order]
        self.ps, self.pe = ps[order], pe[order]
        self.cancel = (p["status"].to_numpy() == "cancel")[order]
        self.proposed = (p["status"].to_numpy() == "proposed")[order]
        ap = self.key // T
        nxt = np.full(len(ap), np.inf)
        same = ap[1:] == ap[:-1]
        nxt[:-1] = np.where(same, self.sent[1:], np.inf)
        self.next_sent = nxt

    def lookup(self, ap: np.ndarray, c: np.ndarray, x: np.ndarray) -> dict:
        """For each airport/cutoff c: is an advisory in force (sent <= c, not yet replaced) whose period covers time x."""
        if len(self.key) == 0:
            z = np.zeros(len(ap), dtype=bool)
            return {"active": z, "proposed": z}
        q = np.where(ap >= 0, ap * T + np.floor(np.nan_to_num(c)).astype(np.int64), 0)
        j = np.searchsorted(self.key, q, side="right") - 1
        jj = np.clip(j, 0, len(self.key) - 1)
        in_force = (ap >= 0) & (j >= 0) & (self.key[jj] // T == ap) & (c < self.next_sent[jj]) & ~self.cancel[jj]
        covers = in_force & (self.ps[jj] <= x) & (x <= self.pe[jj])
        return {"active": covers & ~self.proposed[jj], "proposed": covers & self.proposed[jj]}


class Programs:
    def __init__(self, prog: pd.DataFrame, airports: pd.Index, covered: list):
        """`prog` columns: element, kind, status, sent_utc, period_start, period_end (UTC).
        `covered`: airports whose advisories were collected; every other airport gets NaN, not 0."""
        self.ok = len(prog) > 0
        self.covered = np.zeros(len(airports), dtype=bool)
        idx = airports.get_indexer(list(covered))
        self.covered[idx[idx >= 0]] = True
        self.chains = {"GS": Chain(prog[prog["kind"] == "GS"], airports, GS_DEFAULT_MIN),
                       "GDP": Chain(prog[prog["kind"] == "GDP"], airports, GDP_DEFAULT_MIN)}
        self.n_airports = int(self.covered.sum())

    def at(self, ap: np.ndarray, c: np.ndarray, x: np.ndarray) -> dict:
        gs = self.chains["GS"].lookup(ap, c, c)            # a ground stop matters while it is on
        gdp_now = self.chains["GDP"].lookup(ap, c, c)
        gdp_cov = self.chains["GDP"].lookup(ap, c, x)      # does the program, as advised by c, cover the flight's time
        ok = np.where(ap >= 0, self.covered[np.clip(ap, 0, None)], False)
        f = lambda a: np.where(ok, a.astype("float64"), np.nan)
        return {"gs_now": f(gs["active"]), "gdp_now": f(gdp_now["active"]),
                "gdp_prop": f(gdp_cov["proposed"]), "gdp_cover": f(gdp_cov["active"])}

    def features(self, A, rows, h_minutes: int) -> dict:
        utc = lambda t: t - SHIFT + A["utc_shift"][rows]
        c = utc(A["s_dep"][rows] - h_minutes)
        out = {}
        for side, ap, x in (("o", A["o_id"][rows], utc(A["s_dep"][rows])), ("d", A["d_id"][rows], utc(A["s_arr"][rows]))):
            for k, v in self.at(ap, c, x).items():
                out[f"faa_{side}_{k}"] = v
        return out


def serve_row(P: Programs, query: dict, h_minutes: int) -> dict:
    """One flight. `query` holds 1-element o_id, d_id, s_dep, s_arr and utc_shift arrays. Same code path as the vectorised call."""
    out = P.features(query, np.array([0]), h_minutes)
    return {k: float(v[0]) for k, v in out.items()}
