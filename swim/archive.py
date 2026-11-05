"""Crash-safe compressed archive streams for SWIM records.

One long-lived compressed stream per (source, hour) per process, in its own file
(`<source>_<YYYYmmddTHH>_<runid>.jsonl.zst|gz`). After every batch the stream is
flushed to a decodable boundary (zstd FLUSH_BLOCK / zlib Z_SYNC_FLUSH) and fsynced,
so everything acked to the broker can be read back even if the process is killed
afterwards. A kill mid-write leaves a torn tail; readers stop cleanly at it. Each
process run uses its own files, so a torn tail can never hide later data.

zstd is used when the `zstandard` module is importable (SWIM_ARCHIVE_CODEC=auto),
otherwise gzip. Only the standard library is needed for gzip.
"""
import os
import zlib
from datetime import datetime
from pathlib import Path

EXTS = (".jsonl.zst", ".jsonl.gz")


def zstd_module():
    try:
        import zstandard
        return zstandard
    except ImportError:
        return None


def resolve_codec(name: str = "auto") -> str:
    name = (name or "auto").lower()
    if name not in ("auto", "zstd", "gzip"):
        raise ValueError("SWIM_ARCHIVE_CODEC must be auto, zstd or gzip")
    if name == "gzip":
        return "gzip"
    if zstd_module() is None:
        if name == "zstd":
            raise ValueError("SWIM_ARCHIVE_CODEC=zstd but the zstandard module is not installed")
        return "gzip"
    return "zstd"


def default_level(codec: str) -> int:
    # zstd 10 costs about 0.2% of one core at the filtered rate and compresses ~13% better than 6
    return 10 if codec == "zstd" else 6


def _fsync_dir(path: Path):
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


class _Stream:
    def __init__(self, path: Path, codec: str, level: int):
        self.path, self.codec = path, codec
        new_dirs = not path.parent.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.raw = open(path, "ab")
        self.bytes_in = 0
        if codec == "zstd":
            zstd = zstd_module()
            self.c = zstd.ZstdCompressor(level=level)
            self.obj = self.c.compressobj()
            self.FLUSH = zstd.COMPRESSOBJ_FLUSH_BLOCK
            self.FINISH = zstd.COMPRESSOBJ_FLUSH_FINISH
        else:
            # wbits 31 = gzip container; a Z_SYNC_FLUSH after each batch keeps it decodable
            self.obj = zlib.compressobj(level, zlib.DEFLATED, 31)
        _fsync_dir(path.parent)
        if new_dirs:
            _fsync_dir(path.parent.parent)

    def write(self, data: bytes):
        self.bytes_in += len(data)
        self.raw.write(self.obj.compress(data))

    def sync(self):
        if self.codec == "zstd":
            self.raw.write(self.obj.flush(self.FLUSH))
        else:
            self.raw.write(self.obj.flush(zlib.Z_SYNC_FLUSH))
        self.raw.flush()
        os.fsync(self.raw.fileno())

    def close(self):
        try:
            tail = self.obj.flush(self.FINISH) if self.codec == "zstd" else self.obj.flush(zlib.Z_FINISH)
            self.raw.write(tail)
            self.raw.flush()
            os.fsync(self.raw.fileno())
        finally:
            self.raw.close()


class ArchiveWriter:
    """write_batch([(received_at, line_bytes), ...]) appends and fsyncs; close() finishes streams."""

    def __init__(self, root: Path, source: str, run_id: str, codec: str = "auto", level=None):
        self.root, self.source, self.run_id = Path(root), source, run_id
        self.codec = resolve_codec(codec)
        self.level = int(level) if level else default_level(self.codec)
        self._streams: dict = {}  # hour key -> _Stream
        self.bytes_written = 0

    @property
    def ext(self) -> str:
        return ".jsonl.zst" if self.codec == "zstd" else ".jsonl.gz"

    def path_for(self, ts: datetime) -> Path:
        d = self.root / ts.strftime("%Y-%m-%d")
        return d / f"{self.source}_{ts.strftime('%Y%m%dT%H')}_{self.run_id}{self.ext}"

    def write_batch(self, records: list) -> None:
        if not records:
            return
        by_hour = {}
        for ts, line in records:
            by_hour.setdefault(ts.strftime("%Y%m%dT%H"), []).append((ts, line))
        newest = max(by_hour)
        for key in sorted(by_hour):
            lines = by_hour[key]
            st = self._streams.get(key)
            if st is None:
                st = self._streams[key] = _Stream(self.path_for(lines[0][0]), self.codec, self.level)
            st.write(b"".join(line for _, line in lines))
            st.sync()
        # finish streams for hours that are over (hour keys sort chronologically)
        for key in [k for k in self._streams if k < newest]:
            self._streams.pop(key).close()
        self.bytes_written = sum(s.raw.tell() for s in self._streams.values())

    def close(self):
        for key in list(self._streams):
            try:
                self._streams.pop(key).close()
            except OSError:
                pass


# ------------------------------------------------------------------ reading
def _split_lines(chunks):
    buf = b""
    for out in chunks:
        buf += out
        *lines, buf = buf.split(b"\n")
        for ln in lines:
            yield ln.decode("utf-8", "replace")
    # a partial trailing line (torn write) is dropped on purpose


def _gzip_chunks(path):
    """Decompressed chunks from a (multi-member, possibly truncated or damaged) gzip file."""
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                return
            while chunk:
                snap = d.copy()
                try:
                    out = d.decompress(chunk)
                except zlib.error:
                    # damaged bytes inside this chunk: salvage the valid prefix in small steps
                    d = snap
                    for i in range(0, len(chunk), 1024):
                        try:
                            out = d.decompress(chunk[i:i + 1024])
                        except zlib.error:
                            return
                        if out:
                            yield out
                    return
                if out:
                    yield out
                if d.eof:  # member finished; start the next one
                    chunk = d.unused_data
                    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
                else:
                    chunk = b""


def _zstd_chunks(path):
    zstd = zstd_module()
    if zstd is None:
        raise RuntimeError(f"{path.name}: the zstandard module is needed to read .zst archives")
    d = zstd.ZstdDecompressor().decompressobj(read_across_frames=True)
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 12)  # small slices: little valid data is lost next to damage
            if not chunk:
                return
            try:
                out = d.decompress(chunk)
            except zstd.ZstdError:
                return
            if out:
                yield out


def read_lines(path):
    """Yield text lines from a .jsonl.gz or .jsonl.zst archive; a torn tail is skipped."""
    path = Path(path)
    chunks = _zstd_chunks(path) if path.name.endswith(".zst") else _gzip_chunks(path)
    yield from _split_lines(chunks)


def archive_files(directory: Path) -> list:
    """All archive files under <directory>/<day>/, oldest first."""
    out = []
    for ext in EXTS:
        out += list(Path(directory).glob("*/*" + ext))
    return sorted(out)
