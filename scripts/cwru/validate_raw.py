import os, re, sys
import numpy as np, pandas as pd
from scipy.io import loadmat
WORK = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
fm = pd.read_csv(os.path.join(WORK, "data", "cwru", "file_map.csv"))
RPM = {0: 1797, 1: 1772, 2: 1750, 3: 1730}
bad = 0; rows = []
for _, r in fm.iterrows():
    m = loadmat(os.path.join(WORK, "data", "cwru", "raw", f"{r.file}.mat"))
    keys = [k for k in m if not k.startswith("__")]
    n = r.file
    de, fe = m.get(f"X{n:03d}_DE_time"), m.get(f"X{n:03d}_FE_time")
    rpm = m.get(f"X{n:03d}RPM")
    ok = de is not None and fe is not None and rpm is not None
    rp = float(np.ravel(rpm)[0]) if rpm is not None else np.nan
    ok = ok and abs(rp - RPM[r.load]) <= 15 and len(de) > 100000 and len(de) == len(fe)
    rows.append(dict(file=n, label=r.label, fault_loc=r.fault_loc, load=r.load, n_de=len(de) if de is not None else 0, rpm_file=rp, ok=ok))
    bad += (not ok)
d = pd.DataFrame(rows); print(d.describe(include="all").to_string()); print("not ok:", bad)
print(d[~d.ok].to_string())
