"""Fetch the 18 faulty-bearing vibration members of the KAIST bearing dataset, part 1 (results/kaist/PREDECLARED_KAIST.txt).
The record is public (CC BY 4.0, Mendeley Data ztmf3m7h5x version 6). The 2.7 GB archive vibration.zip is not downloaded as a whole: its central
directory and the 18 needed members are read with HTTP Range requests on the pre-signed URL that the public file link redirects to (resolved with curl,
an honest client string, no bot check is bypassed). Member names are only matched, never used as paths. Each member is checked against the CRC-32 of the
central directory. Output: data/kaist/raw/<member>.mat
Usage: python -I fetch_kaist.py"""
import pathlib, struct, subprocess, sys, time, urllib.request, zlib

HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parents[1]
OUT = WORK / "data" / "kaist" / "raw"
FILE_URL = "https://data.mendeley.com/public-files/datasets/ztmf3m7h5x/files/ee98c5d9-1052-4448-84b2-ed57711b658d/file_downloaded"
WANT = [f"{l}Nm_{r}_{s}.mat" for l in (0, 2, 4) for r in ("BPFI", "BPFO") for s in ("03", "10", "30")]


def resolve():
    out = subprocess.run(["curl", "-s", "-A", "research-data-access curl", "-o", "NUL" if sys.platform == "win32" else "/dev/null",
                          "-w", "%{redirect_url}", FILE_URL], capture_output=True, text=True, timeout=120).stdout.strip()
    assert out.startswith("https://"), out[:200]
    return out


def rng(url, a, b, tries=6):
    err = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
            with urllib.request.urlopen(req, timeout=300) as r:
                d = r.read()
            if len(d) != b - a + 1:
                raise IOError("short read")
            return d
        except Exception as e:
            err = e
            time.sleep(5 * (2 ** min(k, 3)))
    raise RuntimeError(repr(err)[:300])


def central_directory(url):
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        total = int(r.headers["Content-Range"].split("/")[1])
    tail = rng(url, total - 200000, total - 1)
    i = tail.rfind(b"PK\x05\x06")
    _, _, _, _, n_tot, cd_size, cd_off, _ = struct.unpack("<IHHHHIIH", tail[i:i + 22])
    cd = rng(url, cd_off, cd_off + cd_size - 1)
    ent, p = {}, 0
    while p < len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        (_, _, _, _, comp, _, _, crc, csize, usize, nlen, elen, clen, _, _, _, loff) = struct.unpack("<IHHHHHHIIIHHHHHII", cd[p:p + 46])
        name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace").rsplit("/", 1)[-1]
        ent[name] = (comp, crc, csize, usize, loff)
        p += 46 + nlen + elen + clen
    return ent


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in WANT:
        dst = OUT / name
        if dst.exists():
            print("have", name, dst.stat().st_size, flush=True)
            continue
        url = resolve()
        ent = central_directory(url)
        assert name in ent, name
        comp, crc, csize, usize, loff = ent[name]
        hdr = rng(url, loff, loff + 29)
        n2, e2 = struct.unpack("<HH", hdr[26:30])
        data = rng(url, loff + 30 + n2 + e2, loff + 30 + n2 + e2 + csize - 1)
        raw = zlib.decompress(data, -15) if comp == 8 else data
        assert len(raw) == usize and (zlib.crc32(raw) & 0xFFFFFFFF) == crc, ("crc mismatch", name)
        dst.write_bytes(raw)
        print("saved", name, len(raw), flush=True)


if __name__ == "__main__":
    main()
