"""Read single members of remote Zenodo zip archives with HTTP Range requests (official Zenodo API URLs only).
Nothing is written to disk by name (members are returned as bytes), so there is no zip-slip risk; member names are only parsed.
list_members(url) -> list of dict(name, header_offset, compress_size, file_size, compress_type) read from the central directory (zipfile on a seekable range file).
read_member(url, info) -> bytes of the uncompressed member (stored or deflate)."""
import io, struct, time, urllib.request, zlib, zipfile

BASE = "https://zenodo.org/api/records/10868257/files/{}/content"
UA = "Mozilla/5.0 (research data access; python urllib)"   # plain library identification, not a browser spoof of a bot check


def _get(url, start, end, tries=8):
    err = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "python-urllib"})
            with urllib.request.urlopen(req, timeout=180) as r:
                data = r.read()
            if len(data) != end - start + 1:
                raise IOError(f"short read {len(data)} != {end - start + 1}")
            return data
        except Exception as e:      # retry on transient errors (rate limits, resets); polite exponential back-off
            err = e
            wait = 5 * (2 ** min(k, 4))
            ra = getattr(e, "headers", None) and e.headers.get("Retry-After")
            if ra and str(ra).isdigit():
                wait = max(wait, int(ra))
            time.sleep(wait)
    raise RuntimeError("range request failed after retries: " + repr(err)[:300])


def total_size(url):
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": "python-urllib"})
    with urllib.request.urlopen(req, timeout=120) as r:
        cr = r.headers.get("Content-Range")
    return int(cr.split("/")[1])


class RangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests with a one-block cache."""

    def __init__(self, url, block=2 * 1024 * 1024):
        self.url, self.block, self.pos = url, block, 0
        self.size = total_size(url)
        self._blk_start, self._blk = -1, b""

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def read(self, n=-1):
        if n < 0: n = self.size - self.pos
        n = min(n, self.size - self.pos)
        out = []
        while n > 0:
            b0 = (self.pos // self.block) * self.block
            if b0 != self._blk_start:
                self._blk = _get(self.url, b0, min(b0 + self.block, self.size) - 1)
                self._blk_start = b0
            o = self.pos - b0
            chunk = self._blk[o:o + n]
            if not chunk: break
            out.append(chunk); self.pos += len(chunk); n -= len(chunk)
        return b"".join(out)

    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)


def list_members(archive):
    url = BASE.format(archive)
    z = zipfile.ZipFile(RangeFile(url))
    return [dict(name=i.filename, header_offset=i.header_offset, compress_size=i.compress_size, file_size=i.file_size,
                 compress_type=i.compress_type) for i in z.infolist() if not i.is_dir()]


def read_member(archive, info):
    """Uncompressed bytes of one member (no files written)."""
    url = BASE.format(archive)
    off, cs = info["header_offset"], info["compress_size"]
    head = _get(url, off, off + 29)
    sig, = struct.unpack("<I", head[:4])
    assert sig == 0x04034B50, "bad local header"
    nlen, elen = struct.unpack("<HH", head[26:30])
    start = off + 30 + nlen + elen
    raw = _get(url, start, start + cs - 1) if cs > 0 else b""
    if info["compress_type"] == 0:
        data = raw
    elif info["compress_type"] == 8:
        data = zlib.decompress(raw, -15)
    else:
        raise ValueError("unsupported compression %s" % info["compress_type"])
    assert len(data) == info["file_size"], "size mismatch"
    return data


def safe_name(name):
    """Reject absolute paths and parent references (names are only parsed, never used as paths)."""
    n = name.replace("\\", "/")
    return not (n.startswith("/") or ".." in n.split("/") or ":" in n.split("/")[0])
