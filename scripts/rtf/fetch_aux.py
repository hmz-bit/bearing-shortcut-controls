"""Download the small auxiliary files (operatingConditions.csv, log.pdf, meanTemperatures.csv) of every experiment (label-free) -> data/rtf/aux/."""
import sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parents[1]
sys.path.insert(0, str(HERE))
import pandas as pd
import rangezip as R
m = pd.read_csv(WORK / "data" / "rtf" / "manifest.csv")
m["base"] = m.name.str.replace("\\", "/").str.split("/").str[-1]
aux = m[m.base.str.contains(r"_(operatingConditions\.csv|log\.pdf|meanTemperatures\.csv)$")]
out = WORK / "data" / "rtf" / "aux"; out.mkdir(exist_ok=True)
for _, r in aux.iterrows():
    assert R.safe_name(r["name"])
    data = R.read_member(r["archive"], dict(header_offset=r.header_offset, compress_size=r.compress_size, file_size=r.file_size, compress_type=r.compress_type))
    (out / r["base"]).write_bytes(data)      # base name checked by regex above (no separators)
print(len(aux), "files", sum(f.stat().st_size for f in out.iterdir()) / 1e6, "MB")
