"""List the central directories of all 48 Zenodo zip archives (range reads only) -> data/rtf/manifest.csv. Label-free."""
import sys, json, pathlib, urllib.request, re
HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parents[1]
sys.path.insert(0, str(HERE))
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import rangezip as R

OUT = WORK / "data" / "rtf"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    rec = json.load(urllib.request.urlopen(urllib.request.Request("https://zenodo.org/api/records/10868257", headers={"User-Agent": "python-urllib"}), timeout=120))
    arch = sorted(f["key"] for f in rec["files"] if f["key"].endswith(".zip"))
    print(len(arch), "archives", flush=True)

    def one(a):
        ms = R.list_members(a)
        for m in ms: m["archive"] = a
        return ms

    rows = []
    with ThreadPoolExecutor(6) as ex:
        for a, ms in zip(arch, ex.map(one, arch)):
            print(a, len(ms), flush=True)
            rows += ms
    df = pd.DataFrame(rows)
    assert all(R.safe_name(n) for n in df.name), "unsafe member name"
    df.to_csv(OUT / "manifest.csv", index=False)
    print("members", len(df))


if __name__ == "__main__":
    main()
