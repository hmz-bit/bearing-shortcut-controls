"""Extract the pre-declared windows of the Paderborn run-to-failure data set (results/rtf/PREDECLARED_RTF.txt).
Downloads only the selected .mat members (HTTP Range, nothing written to disk by member name), takes the first 1 s of accHorizRear_A, x10 g/V,
resamples to 42 kHz, and computes the LF-PNCN order tensor (t_mad) and the B1 envelope spectrum.
Output: data/rtf/rtf_windows.npz (t_mad, exp, stage, M, f_r, Y [IR, OR], raw_n/sum/sumsq of the 42 kHz g-scaled window) and data/rtf/rtf_env.npy (N,21000) float32."""
import sys, io, pathlib, time, re
HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(WORK / "scripts" / "lfpncn"))
sys.path.insert(0, str(WORK / "scripts" / "b1"))
import numpy as np, pandas as pd
import scipy.io as sio
from scipy import signal
from multiprocessing import Pool
import pymupdf
import rangezip as R
import features as F
import geometry as G
import envspec

RTF_61806 = G.Bearing("RTF_61806", 19, 3.18, 35.5, 0.0)
IR = set("B01 B02 B03 B04 B06 B07 B08 B11 B12 B13 B15 B16".split())
OR = set("B01 B09 B10 B11 B16".split())
ALL = [f"B{i:02d}" for i in range(1, 18)]
FS_RAW = {e: (128000 if int(e[1:]) <= 9 else 64000) for e in ALL}
DATA = WORK / "data" / "rtf"

_m = pd.read_csv(DATA / "manifest.csv")
_m["base"] = _m.name.str.replace("\\", "/").str.split("/").str[-1]
_m = _m[_m.base.str.match(r"data_B\d\d_M\d+\.mat$")].copy()
_m["key"] = _m.base.str.extract(r"data_(B\d\d)_M(\d+)\.mat").apply(lambda r: (r[0], int(r[1])), axis=1)
IDX = {k: r for k, r in zip(_m.key, _m.itertuples())}


def proc(job):
    e, M, stage, f_r = job
    r = IDX[(e, M)]
    assert R.safe_name(r.name)
    info = dict(header_offset=r.header_offset, compress_size=r.compress_size, file_size=r.file_size, compress_type=r.compress_type)
    d = sio.loadmat(io.BytesIO(R.read_member(r.archive, info)), squeeze_me=True)
    x = np.asarray(d["accHorizRear_A"], dtype=np.float64)
    fs = FS_RAW[e]
    assert abs(len(x) - 1.6 * fs) <= 2, (e, M, len(x))
    w = x[:fs] * 10.0                                                   # 10 g/V
    w = signal.resample_poly(w, 21, 64 if fs == 128000 else 32)
    assert len(w) == 42000, len(w)
    assert np.isfinite(w).all()
    t = F.window_to_array(w, f_r, RTF_61806)
    env = envspec.envelope_spectrum(w)
    return t, env, len(w), float(w.sum()), float((w ** 2).sum())


def main():
    # the log of every experiment must give 10 g/V and 0 dB gain
    for e in ALL:
        txt = "\n".join(p.get_text() for p in pymupdf.open(DATA / "aux" / f"{e}_log.pdf"))
        t = " ".join(txt.split())
        assert re.search(r"Conversion Factor\s*10 g/V", t) and re.search(r"Gain Factor\s*0 dB", t), (e, t[:300])
    sel = pd.read_csv(DATA / "selection.csv")
    jobs = [(r.exp, int(r.M), r.stage, float(r.f_r)) for r in sel.itertuples()]
    t0 = time.time()
    outs = []
    with Pool(4) as pool:
        for i, o in enumerate(pool.imap(proc, jobs, chunksize=4)):
            outs.append(o)
            if i % 200 == 0:
                print(i, len(jobs), round(time.time() - t0), "s", flush=True)
    T = np.stack([o[0] for o in outs]).astype(np.float32)
    E = np.stack([o[1] for o in outs]).astype(np.float32)
    exp = sel.exp.values
    faulty = (sel.stage.values == "faulty")
    Y = np.stack([faulty & np.isin(exp, list(IR)), faulty & np.isin(exp, list(OR))], 1).astype(np.float32)
    np.savez(DATA / "rtf_windows.npz", t_mad=T, exp=exp.astype(str), stage=sel.stage.values.astype(str), M=sel.M.values, f_r=sel.f_r.values,
             Y=Y, raw_n=np.array([o[2] for o in outs]), raw_sum=np.array([o[3] for o in outs]), raw_sumsq=np.array([o[4] for o in outs]))
    np.save(DATA / "rtf_env.npy", E)
    print("windows", len(T), "finite", bool(np.isfinite(T).all() and np.isfinite(E).all()), "max env", float(E.max()), "seconds", round(time.time() - t0))


if __name__ == "__main__":
    main()
