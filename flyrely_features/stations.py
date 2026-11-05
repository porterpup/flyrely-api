"""Airport code <-> METAR station mapping (BTS/IATA code, ICAO code, IEM archive station id).

Never "prepend K" or "take the last 3 letters": Alaska, Hawaii, Puerto Rico and Guam have 4-letter ICAO codes
that are not K plus the IATA code (PANC, PHNL, TJSJ), and some airports are listed under a new code (KPBI is
KDJT / DJT in the table).
Reference table: ``data/ref/airports.csv`` (OurAirports), columns ident, icao_code, iata_code, gps_code.
Without that file (serving) a built-in table of the non-K airports is used.
"""

from pathlib import Path

import pandas as pd

REF = Path(__file__).resolve().parent.parent / "data" / "ref" / "airports.csv"

# BTS/IATA code -> ICAO code, for airports whose ICAO code is not K + code (and KPBI, listed as KDJT).
BUILTIN_ICAO = {"ANC": "PANC", "HNL": "PHNL", "OGG": "PHOG", "KOA": "PHKO", "LIH": "PHLI", "SJU": "TJSJ",
                "PBI": "KDJT", "GUM": "PGUM", "STT": "TIST", "STX": "TISX", "ITO": "PHTO", "FAI": "PAFA",
                "JNU": "PAJN", "BET": "PABE", "ADQ": "PADQ"}


def load_table(ref: Path = None):
    """The reference table, or None when the file is not there."""
    f = ref or REF
    if not f.exists():
        return None
    return pd.read_csv(f, dtype=str, usecols=["ident", "icao_code", "iata_code", "gps_code"])


def icao_for(code: str, table: pd.DataFrame = None) -> str:
    """ICAO code of the airport with BTS/IATA code `code`: matched by IATA code, else by ident K + code."""
    code = code.upper()
    if table is None:
        table = load_table()
    if table is None:
        return BUILTIN_ICAO.get(code, "K" + code)
    rows = table[table["iata_code"] == code]
    if rows.empty:
        rows = table[table["ident"] == "K" + code]
    if rows.empty:
        return BUILTIN_ICAO.get(code, "K" + code)
    r = rows.iloc[0]
    for c in ("icao_code", "gps_code", "ident"):
        if isinstance(r[c], str) and r[c]:
            return r[c]
    return "K" + code


def iem_station(code: str, table: pd.DataFrame = None) -> str:
    """Station id in the IEM ASOS archive: 3 letters for K-codes (KATL -> ATL), the ICAO code otherwise (PANC)."""
    icao = icao_for(code, table)
    return icao[1:] if len(icao) == 4 and icao.startswith("K") else icao


def icao_to_code(icao: str) -> str:
    """Live METAR id (aviationweather.gov icaoId) -> BTS/IATA code: TJSJ -> SJU, KATL -> ATL, KDJT -> PBI."""
    icao = icao.upper()
    for code, v in BUILTIN_ICAO.items():
        if v == icao:
            return code
    return icao[1:] if len(icao) == 4 and icao.startswith("K") else icao
