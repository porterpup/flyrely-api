"""Carrier and airport helpers for the linkage store (stdlib only).

IATA->ICAO callsign table (operating carriers, mainline and regional partners that report to BTS), and airport
code normalisation. SWIM feeds mix 4-letter ICAO (KORD) and 3-letter FAA/IATA (ORD) codes; the store keeps the
3-letter form for the contiguous US (K prefix stripped) and a small table for the other US ICAO codes.
"""
import re

IATA_TO_ICAO = {
    "AA": "AAL", "UA": "UAL", "DL": "DAL", "WN": "SWA", "B6": "JBU", "AS": "ASA", "NK": "NKS", "F9": "FFT",
    "HA": "HAL", "G4": "AAY", "SY": "SCX", "XP": "VXP", "MX": "MXY",
    "OO": "SKW", "YX": "RPA", "MQ": "ENY", "9E": "EDV", "PT": "PDT", "OH": "JIA", "C5": "UCA", "G7": "GJS",
    "YV": "ASH", "QX": "QXE", "ZW": "AWI", "CP": "CPZ", "KS": "PEN", "EV": "ASQ", "9K": "KAP", "MU": "MKU",
    "OY": "PSA", "ZG": "PSA",
}
ICAO_TO_IATA = {}
for _i, _c in IATA_TO_ICAO.items():
    ICAO_TO_IATA.setdefault(_c, _i)

CALLSIGN_RE = re.compile(r"^([A-Z]{3})(\d{1,4})([A-Z]{0,2})$")

# US airports whose ICAO code does not start with K
ICAO_US_OTHER = {
    "PHNL": "HNL", "PHOG": "OGG", "PHKO": "KOA", "PHLI": "LIH", "PHTO": "ITO", "PHMK": "MKK", "PHNY": "LNY",
    "PHJR": "JRF", "PANC": "ANC", "PAFA": "FAI", "PAJN": "JNU", "PAKT": "KTN", "PASI": "SIT", "PABE": "BET",
    "PABR": "BRW", "PAOM": "OME", "PAOT": "OTZ", "PACD": "CDV", "PAKN": "AKN", "PAYA": "YAK", "PADQ": "ADQ",
    "PAGS": "GST", "PASX": "SOV", "PAWG": "WRG", "PAPG": "PSG", "PAHO": "HOM", "PAEN": "ENA", "PADL": "DLG",
    "PGUM": "GUM", "PGSN": "SPN", "TJSJ": "SJU", "TJBQ": "BQN", "TJPS": "PSE", "TJMZ": "MAZ", "TIST": "STT",
    "TISX": "STX", "NSTU": "PPG",
}
_US_ICAO_PREFIXES = ("K", "PA", "PH", "PG", "PF", "PO", "PP", "PK", "PM", "PW", "TJ", "TI", "NS")


def split_callsign(callsign: str):
    """'AAL123' -> ('AAL', '123'); None when it is not an airline-style callsign."""
    m = CALLSIGN_RE.match((callsign or "").upper())
    return (m.group(1), str(int(m.group(2)))) if m else None


def callsign_for(iata_carrier: str, flight_no) -> str:
    """Operating callsign for an IATA carrier + number ('OO', 5331 -> 'SKW5331'); '' if the carrier is unknown."""
    icao = IATA_TO_ICAO.get((iata_carrier or "").upper())
    return f"{icao}{int(flight_no)}" if icao else ""


def norm_airport(code: str) -> str:
    c = (code or "").strip().upper()
    if len(c) == 4:
        if c in ICAO_US_OTHER:
            return ICAO_US_OTHER[c]
        if c.startswith("K"):
            return c[1:]
    return c


def is_us_airport(code: str) -> bool:
    """3-letter (FAA/IATA) and US-prefixed 4-letter codes are domestic; other 4-letter ICAO codes are foreign."""
    c = (code or "").strip().upper()
    if len(c) == 3:
        return c.isalnum()
    if len(c) == 4:
        return c in ICAO_US_OTHER or c.startswith(_US_ICAO_PREFIXES)
    return False


def is_us_normalised(code: str) -> bool:
    """For codes already passed through norm_airport: 3 letters are domestic; unmapped 4-letter codes only when
    they carry a US territory prefix."""
    c = code or ""
    return len(c) == 3 or (len(c) == 4 and c.startswith(_US_ICAO_PREFIXES[1:]))
