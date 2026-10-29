"""Hand-crafted features for the random-forest baseline (results/rf/PREDECLARED_RF.txt) from the raw files, in the row order of the existing caches.
Usage: python -I build_features.py cwru|uored|xjtu
Output: data/rf/features_<dataset>.npz with feat (N,20) and the metadata arrays of the existing cache (copied and asserted equal).
Check: for a sample of windows the LF-PNCN tensor is rebuilt and compared with the stored one (so windows, speed and geometry are identical)."""
import glob, os, re, sys, time
import pathlib
from multiprocessing import Pool
import numpy as np, pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(WORK / "scripts" / "lfpncn")); sys.path.insert(0, str(WORK / "scripts" / "xjtu"))
import rfcommon as R
import features as F
import geometry as G

# ------------------------------------------------------------------ CWRU
NOM = {0: 1797, 1: 1772, 2: 1750, 3: 1730}
CW_GEOM = {"DE": G.Bearing("CWRU_DE_6205", 9, 7.94, 39.04, 0.0), "FE": G.Bearing("CWRU_FE_6203", 8, 6.746, 28.50, 0.0)}
CW_FS, CW_WIN, CW_HOP = 12000, 12000, 3000


def cwru_universe():
    spl = WORK / "bearing-data-leakage" / "data" / "splits" / "cwru" / "eval"
    d = pd.concat([pd.read_parquet(f) for f in glob.glob(str(spl / "*.parquet"))]).drop_duplicates("waveform_id")
    return d.sort_values("waveform_id").reset_index(drop=True)


def cwru_proc(args):
    from scipy.io import loadmat
    row, check = args
    FM = pd.read_csv(WORK / "data" / "cwru" / "file_map.csv")
    w = row["waveform_id"]
    m = re.fullmatch(r"(IR|B|OR@6|OR@3)_(\d+)_(DE|FE)_(DE|FE)_12_(\d)HP", w)
    typ, size, fl, sl, load = m.group(1), int(m.group(2)), m.group(3), m.group(4), int(m.group(5))
    f = FM[(FM.fault_loc == fl) & (FM.typ == typ) & (FM["size"] == size) & (FM.load == load)]
    assert len(f) == 1
    n = int(f.iloc[0]["file"])
    x = np.ravel(loadmat(str(WORK / "data" / "cwru" / "raw" / f"{n}.mat"))[f"X{n:03d}_{sl}_time"]).astype(np.float64)
    f_r = NOM[load] / 60.0
    starts = list(range(0, len(x) - CW_WIN + 1, CW_HOP))
    feats = np.stack([R.window_features(x[s:s + CW_WIN], f_r, CW_FS, CW_GEOM[sl]) for s in starts])
    chk = None
    if check:
        k = len(starts) // 2
        chk = (k, F.window_to_array(x[starts[k]:starts[k] + CW_WIN], f_r, CW_GEOM[sl], fs=CW_FS))
    return feats, chk


def build_cwru():
    z = np.load(WORK / "data" / "cwru" / "cwru_windows.npz", allow_pickle=False)
    u = cwru_universe()
    jobs = [(r, i % 4 == 0) for i, (_, r) in enumerate(u.iterrows())]
    with Pool(6) as pool:
        res = pool.map(cwru_proc, jobs, chunksize=4)
    counts = [len(r[0]) for r in res]
    wid = np.concatenate([[u.waveform_id[i]] * c for i, c in enumerate(counts)])
    assert len(wid) == len(z["wid"]) and (wid == z["wid"]).all(), "row order differs from the cache"
    feat = np.concatenate([r[0] for r in res])
    off = np.r_[0, np.cumsum(counts)]
    diffs = [float(np.abs(r[1][1] - z["t_mad"][off[i] + r[1][0]]).max()) for i, r in enumerate(res) if r[1] is not None]
    meta = {k: z[k] for k in ("wid", "sig_loc", "group", "Y", "load")}
    return feat, meta, diffs


# ------------------------------------------------------------------ UORED
UO_GEOM = G.Bearing("UORED_6203", 8, 6.77, 28.50, 0.0)
UO_FR, UO_HOP = 29.75, 10500


def uored_proc(path):
    base = os.path.basename(path)
    x = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0,), encoding="utf-8-sig")
    assert len(x) == 420000
    starts = list(range(0, len(x) - F.WIN + 1, UO_HOP))
    feats = np.stack([R.window_features(x[s:s + F.WIN], UO_FR, F.FS_TARGET, UO_GEOM) for s in starts])
    k = len(starts) // 2
    chk = (k, F.window_to_array(x[starts[k]:starts[k] + F.WIN], UO_FR, UO_GEOM))
    return base, feats, chk


def build_uored():
    z = np.load(WORK / "data" / "uored" / "uored_windows.npz", allow_pickle=False)
    files = sorted(glob.glob(str(WORK / "data" / "uored" / "csv" / "*.csv")))
    assert len(files) == 60
    with Pool(6) as pool:
        res = pool.map(uored_proc, files, chunksize=2)
    counts = [len(r[1]) for r in res]
    fname = np.concatenate([[r[0]] * len(r[1]) for r in res])
    assert (fname == z["fname"]).all(), "row order differs from the cache"
    feat = np.concatenate([r[1] for r in res])
    off = np.r_[0, np.cumsum(counts)]
    diffs = [float(np.abs(r[2][1] - z["t_mad"][off[i] + r[2][0]]).max()) for i, r in enumerate(res)]
    meta = {k: z[k] for k in ("bearing", "state", "cls", "fname", "Y")}
    return feat, meta, diffs


# ------------------------------------------------------------------ XJTU-SY
def xjtu_proc(args):
    from scipy import signal
    import xjtu_common as C
    b, fileno, cond, check = args
    x = C.load_snapshot(b, fileno)
    w = signal.resample_poly(np.asarray(x[0:25600], dtype=np.float64), 105, 64)
    assert len(w) == 42000
    geom = G.Bearing("XJTU_UER204", 8, 7.92, 34.55, 0.0)
    f = R.window_features(w, C.NOMINAL_HZ[cond], F.FS_TARGET, geom)
    chk = F.window_to_array(w, C.NOMINAL_HZ[cond], geom) if check else None
    return f, chk


def build_xjtu():
    import xjtu_common as C
    z = C.load_windows()
    sn = C.select_snapshots()
    jobs = [(r.bearing, r.fileno, r.cond, i % 10 == 0) for i, r in enumerate(sn.itertuples())]
    with Pool(6) as pool:
        res = pool.map(xjtu_proc, jobs, chunksize=8)
    assert (sn.bearing.to_numpy().astype(str) == z["exp"]).all() and (sn.fileno.to_numpy() == z["fileno"]).all() and (sn.stage.to_numpy().astype(str) == z["stage"]).all()
    feat = np.stack([r[0] for r in res])
    diffs = [float(np.abs(r[1] - z["t_mad"][i]).max()) for i, r in enumerate(res) if r[1] is not None]
    meta = {k: z[k] for k in ("exp", "stage", "fileno", "Y", "cond")}
    return feat, meta, diffs


if __name__ == "__main__":
    ds = sys.argv[1]
    t0 = time.time()
    feat, meta, diffs = {"cwru": build_cwru, "uored": build_uored, "xjtu": build_xjtu}[ds]()
    assert feat.shape[1] == 20 and np.isfinite(feat).all(), (feat.shape, int((~np.isfinite(feat)).sum()))
    print(f"{ds}: features {feat.shape}, finite True; t_mad check on {len(diffs)} windows, max abs diff {max(diffs):.3e}; seconds {time.time() - t0:.0f}", flush=True)
    assert max(diffs) < 1e-4, "windows differ from the stored LF-PNCN tensors"
    R.DATA.mkdir(parents=True, exist_ok=True)
    np.savez(R.DATA / f"features_{ds}.npz", feat=feat, **meta)
