"""Offline report over archived SWIM messages (tfms, sfdps, stdds).

Usage:  python -m swim.inspect_tfms --source tfms [--dir data/swim/tfms]

Prints counts and field NAMES only, never full messages or property values
other than a small allow-list of categorical ones.
"""
import argparse
import collections
import glob
import gzip
import json
import re
import statistics
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

REG_NAMES = {"aircraftregistration", "registration", "tailnumber", "registrationnumber"}
ICAO_NAMES = {"aircraftaddress", "modescode", "icaoaddress", "aircraftaddresshex", "modeaddress"}
REG_TEXT = re.compile(r"\bREG/\s*([A-Z0-9-]{2,10})")
ICAO_TEXT = re.compile(r"\bCODE/\s*([0-9A-F]{6})\b")
# role -> element/attribute local names (lower case) carrying an ACTUAL time
TIME_FIELDS = {
    "out": {"airlineouttime", "gatedeparture", "actualoffblocktime", "actualtimeofoffblock"},
    "off": {"airlineofftime", "runwaydeparture", "actualtimeofdeparture", "actualoff", "actualtakeofftime"},
    "on": {"airlineontime", "runwayarrival", "actualtimeofarrival", "actuallandingtime"},
    "in": {"airlineintime", "gatearrival", "actualonblocktime", "actualtimeofonblock"},
}
SKIP_PROP_WORDS = ("dest", "queue", "user", "uuid", "id", "time", "host", "breadcrumb", "codes", "count")


def local(name: str) -> str:
    return name.rsplit("}", 1)[-1].split(":")[-1]


def parse(body: str):
    try:
        return ET.fromstring(body)
    except ET.ParseError:
        return None


def root_tag(root) -> str:
    return local(root.tag)


def walk(root):
    """Yield (localname, attrs{local:value}, text) for every element."""
    for el in root.iter():
        yield local(el.tag), {local(k): v for k, v in el.attrib.items()}, (el.text or "").strip()


def find_registration(root):
    for tag, attrs, text in walk(root):
        if tag.lower() in REG_NAMES and text:
            return text
        for k, v in attrs.items():
            if k.lower() in REG_NAMES and v:
                return v
            m = REG_TEXT.search(v)
            if m:
                return m.group(1)
        m = REG_TEXT.search(text)
        if m:
            return m.group(1)
    return None


def find_icao(root):
    for tag, attrs, text in walk(root):
        if tag.lower() in ICAO_NAMES and text:
            return text
        for k, v in attrs.items():
            if k.lower() in ICAO_NAMES and v:
                return v
            m = ICAO_TEXT.search(v)
            if m:
                return m.group(1)
        m = ICAO_TEXT.search(text)
        if m:
            return m.group(1)
    return None


def find_times(root) -> set:
    """Roles (out/off/on/in) for which a non-empty actual time is present."""
    found = set()
    for tag, attrs, text in walk(root):
        for role, names in TIME_FIELDS.items():
            if tag.lower() in names and text:
                found.add(role)
            if any(k.lower() in names and v for k, v in attrs.items()):
                found.add(role)
        # TFMS departureInformation/arrivalInformation: timeOfDeparture/Arrival estimated="false"
        if tag == "timeOfDeparture" and attrs.get("estimated", "").lower() == "false":
            found.add("off")
        if tag == "timeOfArrival" and attrs.get("estimated", "").lower() == "false":
            found.add("on")
    # FIXM (SFDPS): departure/.../runwayTime/actual@time = wheels off; arrival/.../actual@time = wheels on
    def visit(el, ancestors):
        name = local(el.tag)
        if name == "actual" and el.attrib.get("time"):
            if "departure" in ancestors:
                found.add("off")
            elif "arrival" in ancestors:
                found.add("on")
        for ch in el:
            visit(ch, ancestors | {name})
    visit(root, frozenset())
    return found


def flight_key(root):
    """Stable per-flight id: gufi / flightRef, else aircraft id + departure/arrival."""
    acid = dep = arr = None
    for tag, attrs, text in walk(root):
        if tag.lower() == "gufi" and text:
            return "g:" + text
        if "flightRef" in attrs and attrs["flightRef"]:
            return "r:" + attrs["flightRef"]
        acid = acid or attrs.get("acid")
        dep = dep or attrs.get("depArpt")
        arr = arr or attrs.get("arrArpt")
    return f"a:{acid}|{dep}|{arr}" if acid else None


def field_names(root) -> set:
    out = set()
    for tag, attrs, _ in walk(root):
        out.add(tag)
        out.update(f"{tag}@{k}" for k in attrs if not k.startswith("xmlns"))
    return out


def is_flight(props: dict, root) -> bool:
    cls = str(props.get("TFMDataClass", "")).lower()
    if cls:
        return cls == "flightdata"
    return any(k in root_tag(root).lower() for k in ("flight", "fixm", "message")) and \
        "status" not in root_tag(root).lower()


def iter_records(directory: Path):
    for f in sorted(glob.glob(str(directory / "*" / "*.jsonl.gz"))):
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def pct(a, b):
    return f"{100.0 * a / b:.1f}%" if b else "n/a"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="tfms")
    ap.add_argument("--dir", type=Path, default=None)
    args = ap.parse_args(argv)
    d = args.dir or Path(__file__).resolve().parent.parent / "data" / "swim" / args.source

    total = unparsed = flights = with_reg = with_icao = 0
    per_min = collections.Counter()
    roots, services = collections.Counter(), collections.Counter()
    prop_counts = collections.defaultdict(collections.Counter)
    fields = collections.Counter()
    fl_keys, fl_reg, fl_icao = set(), set(), set()
    fl_roles = collections.defaultdict(set)
    reg_samples = collections.Counter()
    msg_roles = collections.Counter()

    for rec in iter_records(d):
        total += 1
        props = rec.get("props", {})
        try:
            per_min[rec["recv"][:16]] += 1
        except (KeyError, TypeError):
            pass
        for k, v in props.items():
            if not any(w in k.lower() for w in SKIP_PROP_WORDS) and isinstance(v, (str, int, bool)):
                prop_counts[k][str(v)[:40]] += 1
        root = parse(rec.get("body", ""))
        if root is None:
            unparsed += 1
            continue
        roots[root_tag(root)] += 1
        flight = is_flight(props, root)
        services["flight" if flight else str(props.get("TFMDataClass") or root_tag(root))] += 1
        fields.update(field_names(root))
        if not flight:
            continue
        flights += 1
        key = flight_key(root)
        reg, icao, roles = find_registration(root), find_icao(root), find_times(root)
        with_reg += bool(reg)
        with_icao += bool(icao)
        for r in roles:
            msg_roles[r] += 1
        if key:
            fl_keys.add(key)
            if reg:
                fl_reg.add(key)
            if icao:
                fl_icao.add(key)
            fl_roles[key] |= roles

    print(f"source={args.source} dir={'data/swim/' + args.source if args.dir is None else d.name}")
    print(f"total messages: {total} (unparseable: {unparsed})")
    if per_min:
        counts = list(per_min.values())
        print(f"minutes spanned: {len(counts)}  messages/min mean={statistics.mean(counts):.0f} "
              f"median={statistics.median(counts):.0f} max={max(counts)}")
    size = sum(Path(f).stat().st_size for f in glob.glob(str(d / "*" / "*.jsonl.gz")))
    if per_min:
        gb_day = size / len(per_min) * 1440 / 1e9
        print(f"archive size: {size / 1e6:.1f} MB gz; projected {gb_day:.2f} GB/day at this rate")
    print("\nservice breakdown:", dict(services))
    print("root elements:", dict(roots.most_common(10)))
    for k, c in sorted(prop_counts.items()):
        if len(c) <= 15:
            print(f"property {k}:", dict(c.most_common()))
    print(f"\nflight messages: {flights}; distinct flights (by key): {len(fl_keys)}")
    print(f"messages with registration: {with_reg} ({pct(with_reg, flights)}); "
          f"flights with registration: {len(fl_reg)} ({pct(len(fl_reg), len(fl_keys))})")
    print(f"messages with ICAO 24-bit address: {with_icao} ({pct(with_icao, flights)}); "
          f"flights with ICAO address: {len(fl_icao)} ({pct(len(fl_icao), len(fl_keys))})")
    print("messages carrying actual times:", dict(msg_roles) or "none")
    for role, label in (("out", "gate OUT"), ("off", "wheels OFF"), ("on", "wheels ON"), ("in", "gate IN")):
        n = sum(1 for r in fl_roles.values() if role in r)
        print(f"flights with {label}: {n} ({pct(n, len(fl_keys))})")
    print("\nfield-name sample (most common, names only):")
    print(", ".join(f"{k}({v})" for k, v in fields.most_common(40)))
    reg_like = [k for k in fields if re.search(r"regist|tail|address|modes|icao", k, re.I)]
    print("registration/address-like field names present:", reg_like or "none")


if __name__ == "__main__":
    main()
