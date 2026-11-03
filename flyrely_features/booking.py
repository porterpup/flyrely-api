"""Booking-time features: smoothed historical delay/cancel rates by route, airline, hour and month."""

import pandas as pd

SMOOTHING = 50  # pseudo-flights pulling sparse groups toward the global rate

DELAY_KEYS = (["Origin"], ["Dest"], ["Origin", "Dest"], ["Reporting_Airline"],
              ["Reporting_Airline", "Origin"], ["Origin", "dep_hour"],
              ["Reporting_Airline", "dep_hour"], ["Origin", "month"],
              ["Reporting_Airline", "Origin", "Dest", "dep_hour"])
CANCEL_KEYS = (["Origin"], ["Reporting_Airline"], ["Origin", "month"])


def days_to_holiday(dates: pd.Series) -> pd.Series:
    import main  # production holiday calendar; imported lazily because it loads the models
    lookup = {d: main.get_days_to_holiday(pd.Timestamp(d).to_pydatetime()) for d in dates.unique()}
    return dates.map(lookup).astype(int)


def smoothed_rate(hist: pd.DataFrame, keys: list[str], target: str, prior: float) -> pd.DataFrame:
    g = hist.groupby(keys, observed=True)[target].agg(["sum", "count"]).reset_index()
    g[f"rate_{'_'.join(keys)}_{target}"] = (g["sum"] + SMOOTHING * prior) / (g["count"] + SMOOTHING)
    return g.drop(columns=["sum", "count"])


def route_carrier_medians(hist: pd.DataFrame) -> list:
    """Median Distance / CRSElapsedTime by (Origin, Dest, Reporting_Airline), then by (Origin, Dest) as fallback."""
    return [(keys, hist.groupby(keys, observed=True)[["Distance", "CRSElapsedTime"]].median())
            for keys in (["Origin", "Dest", "Reporting_Airline"], ["Origin", "Dest"])]


class BookingTables:
    """Rate tables computed once from `hist` (months strictly earlier); `features` looks rows up in them."""

    def __init__(self, hist: pd.DataFrame, lagged_route_stats: bool = False):
        self.lagged = lagged_route_stats
        if lagged_route_stats:
            self.route_stats = route_carrier_medians(hist)
        flown = hist[~hist["cancelled"]]
        self.p_delay = flown["delayed"].mean()
        self.p_cancel = hist["cancelled"].mean()
        self.delay = [(keys, smoothed_rate(flown, keys, "delayed", self.p_delay)) for keys in DELAY_KEYS]
        hc = hist.assign(c=hist["cancelled"].astype(float))
        self.cancel = [(keys, smoothed_rate(hc, keys, "c", self.p_cancel)) for keys in CANCEL_KEYS]
        days = hist["FlightDate"].nunique()
        self.volume = {col: (hist.groupby(col).size() / days).rename(f"{col}_daily_flights")
                       for col in ("Origin", "Dest")}

    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.lagged:  # serving parity: schedule stats from lagged history, never from the row itself
            out = df[["dep_hour", "dow", "month"]].copy()
            for col, r in self.route_stats:
                keys = list(r.index.names)
                m = df[keys].merge(r.reset_index(), on=keys, how="left").set_index(df.index)
                for c in ("Distance", "CRSElapsedTime"):
                    out[c] = m[c] if c not in out else out[c].fillna(m[c])
        else:
            out = df[["dep_hour", "dow", "month", "Distance", "CRSElapsedTime"]].copy()
        out = out[["dep_hour", "dow", "month", "Distance", "CRSElapsedTime"]]
        out["days_to_holiday"] = days_to_holiday(df["FlightDate"])
        for keys, r in self.delay + self.cancel:
            out = out.join(df[keys].merge(r, on=keys, how="left").set_index(df.index).iloc[:, len(keys):])
        for col, vol in self.volume.items():
            out[vol.name] = df[col].map(vol)

        # Unseen groups get the global rate (what a new route would honestly get)
        rate_cols = [c for c in out.columns if c.startswith("rate_")]
        for c in rate_cols:
            out[c] = out[c].fillna(self.p_cancel if c.endswith("_c") else self.p_delay)
        return out.astype(float)


def booking_features(df: pd.DataFrame, hist: pd.DataFrame, lagged_route_stats: bool = False) -> pd.DataFrame:
    """Booking-time features. Every rate comes from `hist` (months strictly earlier)."""
    return BookingTables(hist, lagged_route_stats).features(df)


def serve_row(row, tables: BookingTables) -> pd.Series:
    """Features for one flight (a dict or Series with the columns of `df`), via the vectorised code path."""
    frame = pd.DataFrame([dict(row)])
    return tables.features(frame).iloc[0]
