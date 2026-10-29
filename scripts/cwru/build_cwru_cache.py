"""Window cache for CWRU (pre-declared in results/cwru/PREDECLARED_CWRU.txt). Works with `python -I`.
144 waveforms (72 official 12 kHz files x DE/FE channel), 1-s windows (12000 samples) with 75% overlap (hop 3000), fs = 12 kHz,
f_r = nominal speed of the load condition, geometry by signal location (DE 6205-2RS, FE 6203-2RS).
Output: data/cwru/cwru_windows.npz with t_mad (N,3,15,64), wid, sig_loc, group, Y (N,2) [inner, outer], load.
"""
import glob, os, re, sys, time
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy.io import loadmat

HERE = os.path.dirname(os.path.abspath(__file__)); WORK = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(WORK, "scripts", "lfpncn"))
import features as F
import geometry as G

FS = 12000
WIN, HOP = 12000, 3000
NOM = {0: 1797, 1: 1772, 2: 1750, 3: 1730}
GEOM = {"DE": G.Bearing("CWRU_DE_6205", 9, 7.94, 39.04, 0.0), "FE": G.Bearing("CWRU_FE_6203", 8, 6.746, 28.50, 0.0)}
SPL = os.path.join(WORK, "bearing-data-leakage", "data", "splits", "cwru", "eval")
FM = pd.read_csv(os.path.join(WORK, "data", "cwru", "file_map.csv"))


def meta_universe():
    d = pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.join(SPL, "*.parquet"))]).drop_duplicates("waveform_id")
    allm = pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.join(SPL, "*.parquet"))])
    # labels, group and ratios must be unique per waveform across all split files
    for c in ("inner", "outer", "group", "bpfo", "bpfi", "rpm", "load", "fs"):
        assert (allm.groupby("waveform_id")[c].nunique() == 1).all(), c
    return d.sort_values("waveform_id").reset_index(drop=True)


def process(row):
    w = row["waveform_id"]
    m = re.fullmatch(r"(IR|B|OR@6|OR@3)_(\d+)_(DE|FE)_(DE|FE)_12_(\d)HP", w)
    typ, size, fl, sl, load = m.group(1), int(m.group(2)), m.group(3), m.group(4), int(m.group(5))
    f = FM[(FM.fault_loc == fl) & (FM.typ == typ) & (FM["size"] == size) & (FM.load == load)]
    assert len(f) == 1
    n = int(f.iloc[0]["file"])
    mat = loadmat(os.path.join(WORK, "data", "cwru", "raw", f"{n}.mat"))
    x = np.ravel(mat[f"X{n:03d}_{sl}_time"]).astype(np.float64)
    f_r = NOM[load] / 60.0
    assert NOM[load] == int(row["rpm"])
    ff = G.fault_frequencies(f_r, GEOM[sl])
    assert abs(ff["f_o"] / f_r - row["bpfo"]) < 1.5e-3 and abs(ff["f_i"] / f_r - row["bpfi"]) < 1.5e-3, (w, ff["f_o"] / f_r, row["bpfo"])
    T = np.stack([F.window_to_array(x[s:s + WIN], f_r, GEOM[sl], fs=FS) for s in range(0, len(x) - WIN + 1, HOP)])
    return w, sl, row["group"], int(row["inner"]), int(row["outer"]), load, T


if __name__ == "__main__":
    u = meta_universe()
    assert len(u) == 144 and (u.fs == 12000).all()
    t0 = time.time()
    with Pool(6) as pool:
        res = pool.map(process, [r for _, r in u.iterrows()], chunksize=4)
    T = np.concatenate([r[6] for r in res]); n = [len(r[6]) for r in res]
    rep = lambda k: np.concatenate([[r[k]] * len(r[6]) for r in res])
    out = dict(t_mad=T, wid=rep(0), sig_loc=rep(1), group=rep(2), Y=np.stack([rep(3), rep(4)], 1).astype(np.float32), load=rep(5).astype(np.int8))
    np.savez(os.path.join(WORK, "data", "cwru", "cwru_windows.npz"), **out)
    print("windows", len(T), "per signal min/max", min(n), max(n), "finite", bool(np.isfinite(T).all()), "seconds", round(time.time() - t0))
