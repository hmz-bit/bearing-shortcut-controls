import os
import numpy as np, pandas as pd
from scipy.io import loadmat
from scipy import signal
WORK = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
fm = pd.read_csv(os.path.join(WORK, "data", "cwru", "file_map.csv"))
NOM = {0: 1797, 1: 1772, 2: 1750, 3: 1730}
rows = []
for _, r in fm.iterrows():
    m = loadmat(os.path.join(WORK, "data", "cwru", "raw", f"{r.file}.mat")); n = r.file
    rpm = float(np.ravel(m[f"X{n:03d}RPM"])[0])
    pk = []
    for ch in ("DE", "FE"):
        x = np.ravel(m[f"X{n:03d}_{ch}_time"]); x = x - x.mean()
        fr, P = signal.welch(x, fs=12000, nperseg=60000, window="hann")       # 0.2 Hz resolution
        sel = (fr >= 25) & (fr <= 33)
        pk.append(fr[sel][np.argmax(P[sel])])
    rows.append(dict(file=n, label=r.label, loc=r.fault_loc, load=r.load, rpm_file_hz=rpm / 60, nominal_hz=NOM[r.load] / 60, peak_de=pk[0], peak_fe=pk[1]))
d = pd.DataFrame(rows)
d["dev_file_%"] = 100 * (d[["peak_de", "peak_fe"]].median(axis=1) - d.rpm_file_hz) / d.rpm_file_hz
d["dev_nom_%"] = 100 * (d[["peak_de", "peak_fe"]].median(axis=1) - d.nominal_hz) / d.nominal_hz
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 100)
print(d.round(2).to_string())
print("abs dev vs file rpm: median %.2f max %.2f | vs nominal: median %.2f max %.2f" % (d["dev_file_%"].abs().median(), d["dev_file_%"].abs().max(), d["dev_nom_%"].abs().median(), d["dev_nom_%"].abs().max()))
