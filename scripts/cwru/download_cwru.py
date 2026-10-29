"""Build the waveform_id -> raw file mapping from the official CWRU 12k tables (link targets) and download the 72 needed files.
Works with `python -I`. Only official engineering.case.edu URLs; sequential; no spoofing. Output: data/cwru/raw/<n>.mat, data/cwru/file_map.csv
"""
import glob, json, os, re, sys, time, urllib.request
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); WORK = os.path.abspath(os.path.join(HERE, "..", ".."))
links = json.load(open(os.path.join(WORK, "data", "cwru", "pages", "cwru_links.json")))
SPL = os.path.join(WORK, "bearing-data-leakage", "data", "splits", "cwru", "eval")
ids = set()
for f in glob.glob(os.path.join(SPL, "*.parquet")):
    ids |= set(pd.read_parquet(f, columns=["waveform_id"])["waveform_id"])
rows = {}
for w in sorted(ids):
    m = re.fullmatch(r"(IR|B|OR@6|OR@3)_(\d+)_(DE|FE)_(DE|FE)_12_(\d)HP", w)
    typ, size, fl, sl, load = m.group(1), int(m.group(2)), m.group(3), m.group(4), int(m.group(5))
    lab = {"IR": "IR%03d_%d", "B": "B%03d_%d"}.get(typ)
    lab = (lab % (size, load)) if lab else "OR%03d%s_%d" % (size, typ[2:], load)
    url = links[fl][lab]
    num = int(re.search(r"/(\d+)\.mat$", url).group(1))
    rows[(fl, typ, size, load)] = dict(fault_loc=fl, typ=typ, size=size, load=load, label=lab, url=url, file=num)
fm = pd.DataFrame(rows.values()).sort_values(["fault_loc", "typ", "size", "load"])
assert len(fm) == 72 and fm["file"].nunique() == 72, (len(fm), fm["file"].nunique())
fm.to_csv(os.path.join(WORK, "data", "cwru", "file_map.csv"), index=False)
print(fm.to_string())
for _, r in fm.iterrows():
    dst = os.path.join(WORK, "data", "cwru", "raw", f"{r.file}.mat")
    if os.path.exists(dst) and os.path.getsize(dst) > 10000:
        continue
    urllib.request.urlretrieve(r.url, dst)
    time.sleep(0.3)
print("downloaded", len(glob.glob(os.path.join(WORK, "data", "cwru", "raw", "*.mat"))))
