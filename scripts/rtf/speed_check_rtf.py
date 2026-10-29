"""Label-free check of the speed columns against the shaft line of the vibration spectrum (channel A, first 1 s), 6 snapshots per experiment
(evenly spaced over the snapshots whose SET speed lies inside the Table 2 range). Writes results/rtf/speed_check.txt."""
import sys, io, pathlib
HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parents[1]
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import scipy.io as sio
from scipy import signal
from concurrent.futures import ThreadPoolExecutor
import rangezip as R

RANGE = dict(B01=(2400, 3000), B02=(2400, 3000), B03=(2400, 3000), B04=(1500, 3000), B05=(1500, 3000), B06=(1500, 3600), B07=(1500, 3000),
             B08=(1500, 3000), B09=(1500, 3000), B10=(1500, 2700), B11=(1500, 2700), B12=(1500, 2700), B13=(1500, 2700), B14=(2700, 2700),
             B15=(2700, 2700), B16=(2700, 2700), B17=(2700, 2700))
m = pd.read_csv(WORK / "data" / "rtf" / "manifest.csv")
m["base"] = m.name.str.replace("\\", "/").str.split("/").str[-1]
mat = m[m.base.str.match(r"data_B\d\d_M\d+\.mat$")].copy()
mat["exp"] = mat.base.str.extract(r"data_(B\d\d)_")[0]
mat["M"] = mat.base.str.extract(r"_M(\d+)\.mat")[0].astype(int)
idx = {(r.exp, r.M): r for r in mat.itertuples()}
rows, jobs = [], []
for e, (lo, hi) in RANGE.items():
    oc = pd.read_csv(WORK / "data" / "rtf" / "aux" / f"{e}_operatingConditions.csv")
    st = oc["setSpeed / rpm"].values
    ok = np.where((st >= lo - 1) & (st <= hi + 1))[0]
    for k in np.linspace(0, len(ok) - 1, 6).round().astype(int):
        i = int(ok[k]); jobs.append((e, i + 1, float(st[i]), float(oc["meanAbs_speed / rpm"].values[i])))


def work(j):
    e, M, sset, smeas = j
    r = idx[(e, M)]
    info = dict(header_offset=r.header_offset, compress_size=r.compress_size, file_size=r.file_size, compress_type=r.compress_type)
    d = sio.loadmat(io.BytesIO(R.read_member(r.archive, info)), squeeze_me=True)
    x = np.asarray(d["accHorizRear_A"], dtype=float)
    fs = round(len(x) / 1.6)
    x = x[: fs] - x[: fs].mean()
    f, P = signal.welch(x, fs=fs, nperseg=fs, window="hann")       # 1 Hz resolution
    med = np.median(P[(f >= 10) & (f <= 200)])
    def prom(rpm):
        c = rpm / 60.0; hw = max(0.02 * c, 1.5); w = (f >= c - hw) & (f <= c + hw)
        return float(P[w].max() / med), float(f[w][np.argmax(P[w])])
    ps, fs_pk = prom(sset); pm, fm_pk = prom(smeas)
    return dict(exp=e, M=M, fs=fs, set_rpm=sset, meas_rpm=smeas, prom_set=ps, peak_set_Hz=fs_pk, prom_meas=pm, peak_meas_Hz=fm_pk,
                prom_half_set=prom(sset / 2)[0], prom_double_set=prom(sset * 2)[0], keys=sorted(k for k in d if not k.startswith("__")))


with ThreadPoolExecutor(6) as ex:
    res = list(ex.map(work, jobs))
df = pd.DataFrame(res)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
out = df.drop(columns=["keys"]).round(1).to_string()
(WORK / "results" / "rtf" / "speed_check.txt").write_text(out + "\nvariables: " + str(res[0]["keys"]) + "\n", encoding="utf-8")
print(out); print(res[0]["keys"])
