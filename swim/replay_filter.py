"""Offline replay: run the receive-time filter over archived SWIM messages and measure the reduction.

Reads archives only (never connects to SWIM, never modifies them); output archives go to a temp dir.

Usage:  python -m swim.replay_filter --source tfms [--dir data/swim/tfms] [--config filter.json]
                                     [--codec auto|zstd|gzip] [--track-interval-min N] [--blocklist FILE]
"""
import argparse
import json
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path

from swim import archive, consumer, filters


def replay(source, directory, cfg=None, codec="auto", blocklist=None, limit=None):
    """Returns a dict of measurements. Timestamps in the archive (`recv`) drive thinning and rates."""
    files = archive.archive_files(directory)
    in_gz = sum(f.stat().st_size for f in files)
    flt = filters.MessageFilter(source, cfg, filters.Blocklist(blocklist))
    tmp = Path(tempfile.mkdtemp(prefix="swim_replay_"))
    try:
        w_all = archive.ArchiveWriter(tmp / "all", source, "all", codec)
        w_kept = archive.ArchiveWriter(tmp / "kept", source, "kept", codec)
        n = kept = raw_all = raw_kept = 0
        first = last = None
        by_type_in = by_type_kept = None
        t0 = time.time()
        pend_all, pend_kept = [], []
        for f in files:
            for line in archive.read_lines(f):
                try:
                    rec = json.loads(line)
                    ts = datetime.fromisoformat(rec["recv"])
                except (ValueError, KeyError):
                    continue
                n += 1
                first = first or ts
                last = ts
                body = rec.get("body")
                raw = (line + "\n").encode()
                raw_all += len(raw)
                pend_all.append((ts, raw))
                if body is None:  # binary body: always keep
                    out, rec_out = None, raw
                else:
                    out, _ = flt.apply(rec.get("props", {}), body, ts.timestamp())
                    if out is None:
                        rec_out = None
                    elif out is body:
                        rec_out = raw
                    else:
                        rec["body"] = out
                        rec_out = (json.dumps(rec, default=str) + "\n").encode()
                if rec_out is not None:
                    kept += 1
                    raw_kept += len(rec_out)
                    pend_kept.append((ts, rec_out))
                if len(pend_all) >= 500:
                    w_all.write_batch(pend_all); w_kept.write_batch(pend_kept)
                    pend_all, pend_kept = [], []
                if limit and n >= limit:
                    break
        w_all.write_batch(pend_all); w_kept.write_batch(pend_kept)
        w_all.close(); w_kept.close()
        size = lambda d: sum(p.stat().st_size for p in (tmp / d).rglob("*") if p.is_file())
        span = max((last - first).total_seconds(), 1.0) if first else 1.0
        per_day = lambda b: b / span * 86400 / 1e9
        return {
            "source": source, "codec": w_kept.codec, "messages_in": n, "messages_kept": kept,
            "span_minutes": round(span / 60, 1), "counts": dict(flt.counts.most_common()),
            "raw_mb_in": raw_all / 1e6, "raw_mb_kept": raw_kept / 1e6,
            "gz_mb_existing": in_gz / 1e6,
            "mb_unfiltered_new_codec": size("all") / 1e6, "mb_filtered_new_codec": size("kept") / 1e6,
            "gb_day_existing": per_day(in_gz), "gb_day_unfiltered_new_codec": per_day(size("all")),
            "gb_day_filtered": per_day(size("kept")), "replay_seconds": round(time.time() - t0, 1),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=consumer.SOURCES, required=True)
    ap.add_argument("--dir", type=Path, default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--codec", default="auto")
    ap.add_argument("--track-interval-min", type=float, default=None)
    ap.add_argument("--blocklist", default=None)
    ap.add_argument("--limit", type=int, default=None, help="stop after N messages (quick test)")
    args = ap.parse_args(argv)
    cfg = filters.load_config(args.config)[args.source]
    if args.track_interval_min is not None:
        cfg["track_interval_min"] = args.track_interval_min
    d = args.dir or consumer.DATA_ROOT / args.source
    r = replay(args.source, d, cfg, args.codec, args.blocklist, args.limit)
    pct = lambda a, b: f"{100 * a / b:.1f}%" if b else "n/a"
    print(f"source={r['source']} codec={r['codec']} sample span={r['span_minutes']} min")
    print(f"messages: {r['messages_in']} in -> {r['messages_kept']} kept ({pct(r['messages_kept'], r['messages_in'])})")
    print(f"uncompressed MB: {r['raw_mb_in']:.1f} -> {r['raw_mb_kept']:.1f} ({pct(r['raw_mb_kept'], r['raw_mb_in'])})")
    print(f"compressed MB: existing {r['gz_mb_existing']:.1f} | unfiltered/new codec {r['mb_unfiltered_new_codec']:.1f}"
          f" | filtered/new codec {r['mb_filtered_new_codec']:.1f}")
    print(f"projected GB/day (sample rate x 24h): existing {r['gb_day_existing']:.2f} | "
          f"new codec only {r['gb_day_unfiltered_new_codec']:.2f} | filtered {r['gb_day_filtered']:.2f}")
    print("decisions:")
    for k, v in r["counts"].items():
        print(f"  {k:40s} {v}")


if __name__ == "__main__":
    main()
