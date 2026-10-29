"""Hand-crafted features for the random-forest baseline on the Paderborn run-to-failure windows (results/rf/PREDECLARED_RF.txt).
The raw members are fetched again with HTTP Range requests exactly as in scripts/rtf/build_rtf_windows.py (official Zenodo URLs, nothing written by member name);
the 1-s window (accHorizRear_A, x10 g/V, resampled to 42 kHz) is the same as for the LF-PNCN cache. Output: data/rf/features_rtf.npz and data/rf/rtf_windows42k.npy (float32).
Check: the LF-PNCN tensor of every 10th window is rebuilt and compared with the stored one. Usage: python -I build_features_rtf.py"""
import io, sys, pathlib, time
from multiprocessing import Pool
HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(WORK / "scripts" / "rtf")); sys.path.insert(0, str(WORK / "scripts" / "lfpncn"))
import numpy as np, pandas as pd
import scipy.io as sio
from scipy import signal
import rangezip as RZ
import features as F
import geometry as G
import rfcommon as R

RTF_61806 = G.Bearing("RTF_61806", 19, 3.18, 35.5, 0.0)
ALL = [f"B{i:02d}" for i in range(1, 18)]
FS_RAW = {e: (128000 if int(e[1:]) <= 9 else 64000) for e in ALL}
DATA = WORK / "data" / "rtf"


def load_index():
    m = pd.read_csv(DATA / "manifest.csv")
    m["base"] = m.name.str.replace("\\", "/").str.split("/").str[-1]
    m = m[m.base.str.match(r"data_B\d\d_M\d+\.mat$")].copy()
    m["key"] = m.base.str.extract(r"data_(B\d\d)_M(\d+)\.mat").apply(lambda r: (r[0], int(r[1])), axis=1)
    return {k: r for k, r in zip(m.key, m.itertuples())}


IDX = load_index()


def proc(job):
    e, M, f_r, check = job
    r = IDX[(e, M)]
    assert RZ.safe_name(r.name)
    info = dict(header_offset=r.header_offset, compress_size=r.compress_size, file_size=r.file_size, compress_type=r.compress_type)
    d = sio.loadmat(io.BytesIO(RZ.read_member(r.archive, info)), squeeze_me=True)
    x = np.asarray(d["accHorizRear_A"], dtype=np.float64)
    fs = FS_RAW[e]
    assert abs(len(x) - 1.6 * fs) <= 2, (e, M, len(x))
    w = signal.resample_poly(x[:fs] * 10.0, 21, 64 if fs == 128000 else 32)
    assert len(w) == 42000 and np.isfinite(w).all()
    f = R.window_features(w, f_r, F.FS_TARGET, RTF_61806)
    chk = F.window_to_array(w, f_r, RTF_61806) if check else None
    return f, chk, w.astype(np.float32)


def main():
    z = np.load(DATA / "rtf_windows.npz", allow_pickle=False)
    sel = pd.read_csv(DATA / "selection.csv")
    assert (sel.exp.to_numpy().astype(str) == z["exp"]).all() and (sel.M.to_numpy() == z["M"]).all() and (sel.stage.to_numpy().astype(str) == z["stage"]).all()
    jobs = [(r.exp, int(r.M), float(r.f_r), i % 10 == 0) for i, r in enumerate(sel.itertuples())]
    t0 = time.time()
    outs = []
    with Pool(4) as pool:
        for i, o in enumerate(pool.imap(proc, jobs, chunksize=4)):
            outs.append(o)
            if i % 200 == 0:
                print(i, len(jobs), round(time.time() - t0), "s", flush=True)
    feat = np.stack([o[0] for o in outs])
    diffs = [float(np.abs(o[1] - z["t_mad"][i]).max()) for i, o in enumerate(outs) if o[1] is not None]
    print(f"rtf: features {feat.shape}, finite {bool(np.isfinite(feat).all())}; t_mad check on {len(diffs)} windows, max abs diff {max(diffs):.3e}; seconds {time.time() - t0:.0f}", flush=True)
    assert np.isfinite(feat).all() and max(diffs) < 1e-4
    R.DATA.mkdir(parents=True, exist_ok=True)
    np.save(R.DATA / "rtf_windows42k.npy", np.stack([o[2] for o in outs]))
    np.savez(R.DATA / "features_rtf.npz", feat=feat, exp=z["exp"], stage=z["stage"], M=z["M"], Y=z["Y"], f_r=z["f_r"])


if __name__ == "__main__":
    main()
